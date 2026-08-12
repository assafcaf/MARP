import os
import random
import time
import numpy as np

from src.env.commons_env import HarvestCommonsEnv, MAP
from src.reward_model.preference_buffer import EpisodeRecord, PreferenceBuffer
from src.reward_model.reward_model import RewardModel
from src.reward_model.reward_trainer import RewardModelTrainer
from .config import TrainerConfig, save_config
from .logging_utils import ResultLogger
from .registry import build_algorithm
from .video_utils import VideoRecorder
from .metrics import compute_agent_step_metrics


class Trainer:
    def __init__(self, config: TrainerConfig):
        self.config = config
        # Generate random seed if None
        if self.config.seed is None:
            self.config.seed = random.randint(0, 2**31 - 1)
        # Seed all random number generators
        self._seed_rngs(self.config.seed)
        self.env = self._build_env()
        self.algorithm = build_algorithm(config.algorithm)
        self.algorithm.on_env_ready(self.env)
        self.logger = self._build_logger()
    
    def _seed_rngs(self, seed: int) -> None:
        """Seed all random number generators for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        # Try to seed torch if available
        try:
            import torch
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except ImportError:
            pass

    def _build_env(self) -> HarvestCommonsEnv:
        env_cfg = self.config.env
        ascii_map = MAP[env_cfg.map_type]
        return HarvestCommonsEnv(
            ascii_map=ascii_map,
            num_agents=env_cfg.num_agents,
            render=env_cfg.render,
            agent_view_range=env_cfg.agent_view_range,
            ep_length=env_cfg.ep_length,
            spawn_speed=env_cfg.spawn_speed,
            metric=env_cfg.metric,
            penalty=env_cfg.penalty,
        )

    def _build_logger(self) -> ResultLogger:
        log_cfg = self.config.logging
        algo = self.config.algorithm.name
        rm_cfg = self.config.reward_model
        run_name = log_cfg.run_name
        if not run_name:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            run_name = f"{timestamp}-{algo}-map={self.config.env.map_type}-agents={self.config.env.num_agents}"
        rm_suffix = f"rm={rm_cfg.mode}" if rm_cfg.enabled else "rm=off"
        run_name = f"{run_name}-{rm_suffix}"
        if self.config.seed is not None:
            run_name = f"{run_name}-seed={self.config.seed}"
        logger = ResultLogger(log_cfg.log_dir, run_name)
        config_path = os.path.join(logger.run_dir, "config.json")
        save_config(config_path, self.config)
        print(f"Using random seed: {self.config.seed}")
        return logger

    def _build_video_recorder(self) -> VideoRecorder:
        log_cfg = self.config.logging
        videos_dir = os.path.join(self.logger.run_dir, "videos")
        os.makedirs(videos_dir, exist_ok=True)
        return VideoRecorder(
            base_dir=videos_dir,
            enabled=log_cfg.video_enabled,
            every_n_episodes=log_cfg.video_every_n_episodes,
            max_steps=log_cfg.video_max_steps,
            fps=log_cfg.video_fps,
            keep_frames=log_cfg.video_keep_frames,
            total_episodes=self.config.episodes,
        )

    def _format_reward_obs(self, obs: dict, agent_id: str) -> np.ndarray:
        img = obs[agent_id]["curr_obs"]
        normalize = False
        if hasattr(self.config, "algorithm") and hasattr(self.config.algorithm, "dqn"):
            normalize = getattr(self.config.algorithm.dqn, "normalize_obs", False)
        if normalize:
            return (img / 255.0).astype(np.float32)
        return img.astype(np.float32)

    def train(self) -> None:
        # Set total episodes for algorithms that support entropy annealing
        if hasattr(self.algorithm, "set_total_episodes"):
            self.algorithm.set_total_episodes(self.config.episodes)

        video_recorder = self._build_video_recorder()
        rm_cfg = self.config.reward_model
        reward_model = None
        rm_trainer = None
        pref_buffer = None
        global_step = 0
        last_rm_update_step = 0
        if rm_cfg.enabled:
            obs_shape = self.env.observation_space["curr_obs"].shape
            num_actions = int(self.env.action_space.n)
            reward_model = RewardModel(obs_shape=obs_shape, num_actions=num_actions)
            rm_trainer = RewardModelTrainer(
                reward_model,
                lr=rm_cfg.lr,
                device=rm_cfg.device,
                use_amp=rm_cfg.use_amp,
                chunk_size=rm_cfg.chunk_size,
            )
            reward_model = rm_trainer.reward_model
            pref_buffer = PreferenceBuffer(
                rm_cfg.max_episodes_in_buffer,
                max_steps_per_sequence=rm_cfg.max_steps_per_sequence,
            )

        for episode in range(self.config.episodes):
            obs, infos = self.env.reset(seed=None)
            episode_rewards = {agent_id: 0.0 for agent_id in obs.keys()}
            episode_pred_rewards = {agent_id: 0.0 for agent_id in obs.keys()} if rm_cfg.enabled else None
            episode_agent_trajs = {agent_id: [] for agent_id in obs.keys()} if rm_cfg.enabled else None
            # Track detailed step-by-step data per agent for extended logging
            agent_episode_details = {agent_id: [] for agent_id in obs.keys()} if self.config.logging.log_agent_episode_details else None
            step_count = 0
            video_recorder.start(episode)
            for step in range(self.config.env.ep_length):
                actions = self.algorithm.act(obs, step)
                step_obs_imgs = {}
                if rm_cfg.enabled:
                    for agent_id in obs.keys():
                        obs_img = self._format_reward_obs(obs, agent_id)
                        step_obs_imgs[agent_id] = obs_img
                        episode_agent_trajs[agent_id].append((obs_img, actions[agent_id]))
                next_obs, rewards, dones, infos = self.env.step(actions)
                if rm_cfg.enabled:
                    pred_rewards = {
                        agent_id: reward_model.predict(step_obs_imgs[agent_id], actions[agent_id])
                        for agent_id in obs.keys()
                    }
                    self.algorithm.observe(obs, actions, pred_rewards, next_obs, dones, infos, step)
                    for agent_id, reward in pred_rewards.items():
                        episode_pred_rewards[agent_id] += reward
                else:
                    self.algorithm.observe(obs, actions, rewards, next_obs, dones, infos, step)
                video_recorder.record(self.env, step)
                for agent_id, reward in rewards.items():
                    episode_rewards[agent_id] += reward
                # Track detailed step data per agent
                if agent_episode_details is not None:
                    for agent_id in obs.keys():
                        # Check if an apple was eaten (reward > 0 indicates apple consumption)
                        apple_eaten = bool(rewards[agent_id] > 0)
                        
                        # Compute agent-specific metrics
                        agent = self.env.agents[agent_id]
                        metrics = compute_agent_step_metrics(
                            agent=agent,
                            env=self.env,
                            reward=rewards[agent_id],
                            apple_eaten=apple_eaten,
                            nearby_radius=2
                        )
                        
                        step_data = {
                            "step": step,
                            "action": int(actions[agent_id]),
                            "reward": float(rewards[agent_id]),
                            "done": bool(dones.get(agent_id, False)),
                            "apple_eaten": apple_eaten,
                            "nearby_apples": metrics["nearby_apples"],
                            "ate_last_apple_in_cluster": metrics["ate_last_apple_in_cluster"],
                        }
                        if rm_cfg.enabled and pred_rewards is not None:
                            step_data["predicted_reward"] = float(pred_rewards[agent_id])
                        agent_episode_details[agent_id].append(step_data)
                obs = next_obs
                step_count = step + 1
                global_step += 1
                if dones.get("__all__", False):
                    break

            self.env.compute_social_metrics()
            metrics = self.env.get_social_metrics()
            if rm_cfg.enabled:
                pref_buffer.add_episode(EpisodeRecord(agent_trajs=episode_agent_trajs, metrics=metrics))
            rm_metrics = {}
            if rm_cfg.enabled and (episode + 1) >= rm_cfg.warmup_episodes:
                if (global_step - last_rm_update_step) >= rm_cfg.update_every_env_steps:
                    rm_metrics = rm_trainer.train(
                        pref_buffer,
                        phi_key=rm_cfg.phi,
                        mode=rm_cfg.mode,
                        batch_pairs=rm_cfg.batch_pairs,
                        train_steps=rm_cfg.train_steps_per_update,
                    )
                    last_rm_update_step = global_step
            if rm_cfg.enabled and (episode + 1) % rm_cfg.save_every_episodes == 0:
                reward_model_path = os.path.join(self.logger.run_dir, "reward_model.pt")
                reward_model.save(reward_model_path)

            algo_metrics = self.algorithm.on_episode_end(episode)
            if algo_metrics is None:
                algo_metrics = {}
            if rm_metrics:
                algo_metrics = dict(algo_metrics)
                algo_metrics["reward_model"] = rm_metrics
            payload = {
                "episode": episode,
                "steps": step_count,
                "reward_sum": float(np.sum(list(episode_rewards.values()))),
                "reward_mean": float(np.mean(list(episode_rewards.values()))),
                "reward_per_agent": episode_rewards,
                "social_metrics": metrics,
                "algo_metrics": algo_metrics,
            }
            if rm_cfg.enabled and episode_pred_rewards is not None:
                payload["reward_pred_sum"] = float(np.sum(list(episode_pred_rewards.values())))
                payload["reward_pred_mean"] = float(np.mean(list(episode_pred_rewards.values())))
                payload["reward_pred_per_agent"] = episode_pred_rewards
            if episode % self.config.logging.log_interval == 0:
                self.logger.log_episode(payload)
            # Log detailed agent episode data to separate files
            if agent_episode_details is not None:
                for agent_id, details in agent_episode_details.items():
                    episode_summary = {
                        "total_steps": step_count,
                        "total_reward": float(episode_rewards[agent_id]),
                        "steps": details,
                    }
                    if rm_cfg.enabled and episode_pred_rewards is not None:
                        episode_summary["total_predicted_reward"] = float(episode_pred_rewards[agent_id])
                    if metrics:
                        episode_summary["social_metrics"] = metrics
                    self.logger.log_agent_episode_details(agent_id, episode, episode_summary)
            video_recorder.finish()

        if hasattr(self.algorithm, "save"):
            model_path = os.path.join(self.logger.run_dir, "model_last.pt")
            self.algorithm.save(model_path)
        if rm_cfg.enabled and reward_model is not None:
            reward_model_path = os.path.join(self.logger.run_dir, "reward_model_last.pt")
            reward_model.save(reward_model_path)

        video_recorder.finalize()
        self.logger.close()
