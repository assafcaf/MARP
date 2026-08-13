import os
import random
import numpy as np
from omegaconf import OmegaConf

from ..env.commons_env import HarvestCommonsEnv, MAP
from ..env.frame_stack import FrameStackEnv
from ..reward_model.preference_buffer import EpisodeRecord, PreferenceBuffer
from ..reward_model.reward_model import RewardModel
from ..reward_model.reward_trainer import RewardModelTrainer
from .config import TrainerConfig
from .console import TrainingConsole, format_metrics as _format_metrics
from .logging_utils import ResultLogger
from .registry import build_algorithm
from .run_info import write_run_info
from .run_paths import run_path
from .video_utils import VideoRecorder
from .metrics import compute_agent_step_metrics


class Trainer:
    # Above this projected resident size, the preference buffer gets a warning
    # at startup. PreferenceBuffer's docstring records a real OOM from this
    # arithmetic; widening the view and stacking frames both push straight
    # into it (view 7 + 2 frames is ~20 GB at the default buffer length).
    BUFFER_WARN_BYTES = 8 * 1024**3

    def __init__(self, config: TrainerConfig):
        self.config = config
        # Generate random seed if None
        if self.config.seed is None:
            self.config.seed = random.randint(0, 2**31 - 1)
        # Seed all random number generators
        self._seed_rngs(self.config.seed)
        self.console = TrainingConsole.from_config(config.logging)
        self.console.section("Setup")
        self.env = self._build_env()
        self.algorithm = build_algorithm(config.algorithm)
        self.algorithm.on_env_ready(self.env)
        self.logger = self._build_logger()
        self._announce_setup()

    def _announce_setup(self) -> None:
        """The parameters worth seeing at a glance when a run starts."""
        env_cfg = self.config.env
        # The algorithm resolves "auto" to a real device; fall back to the
        # configured value for policies that never build a torch module.
        device = getattr(
            self.algorithm, "device", getattr(self.config.algorithm, "device", "n/a")
        )
        self.console.info(
            f"algorithm      : {getattr(self.config.algorithm, 'name', 'unknown')} (device={device})"
        )
        self.console.info(f"episodes       : {self.config.episodes} x {env_cfg.ep_length} steps")
        stack = f" frames={env_cfg.num_frames}" if env_cfg.num_frames > 1 else ""
        self.console.info(
            f"environment    : map={env_cfg.map_type} agents={env_cfg.num_agents}"
            f" view={env_cfg.agent_view_range}{stack} spawn={env_cfg.spawn_speed}"
        )
        self.console.info(f"seed           : {self.config.seed}")
        log_cfg = self.config.logging
        video = (
            f"every {log_cfg.video_every_n_episodes} episodes" if log_cfg.video_enabled else "off"
        )
        self.console.info(f"video          : {video}")
        self.console.info(f"run directory  : {self.logger.run_dir}")
        self._warn_if_buffer_large()

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

    def _build_env(self):
        env_cfg = self.config.env
        ascii_map = MAP[env_cfg.map_type]
        env = HarvestCommonsEnv(
            ascii_map=ascii_map,
            num_agents=env_cfg.num_agents,
            render=env_cfg.render,
            agent_view_range=env_cfg.agent_view_range,
            ep_length=env_cfg.ep_length,
            spawn_speed=env_cfg.spawn_speed,
            metric=env_cfg.metric,
            penalty=env_cfg.penalty,
        )
        num_frames = int(getattr(env_cfg, "num_frames", 1))
        if num_frames > 1:
            return FrameStackEnv(env, num_frames)
        return env

    def _projected_buffer_bytes(self) -> int:
        """Resident size the preference buffer will reach once full.

        Frames are stored raw, so this is the frame size times the number of
        frames retained: one per agent per step, for every episode the ring
        buffer holds. `store_max_steps_per_agent` subsamples at insertion time
        and so replaces `ep_length` when it is set.
        """
        env_cfg = self.config.env
        rm_cfg = self.config.reward_model
        side = 2 * env_cfg.agent_view_range + 1
        bytes_per_frame = side * side * 3 * int(getattr(env_cfg, "num_frames", 1))
        steps = rm_cfg.store_max_steps_per_agent or env_cfg.ep_length
        return bytes_per_frame * rm_cfg.max_episodes_in_buffer * steps * env_cfg.num_agents

    def _warn_if_buffer_large(self) -> None:
        if not self.config.reward_model.enabled:
            return
        projected = self._projected_buffer_bytes()
        if projected <= self.BUFFER_WARN_BYTES:
            return
        self.console.warn(
            f"preference buffer will reach ~{projected / 1024**3:.1f} GB when full. "
            "Lower reward_model.max_episodes_in_buffer or set "
            "reward_model.store_max_steps_per_agent to bound it."
        )

    def _build_logger(self) -> ResultLogger:
        log_cfg = self.config.logging
        # Under Hydra the entry point hands us Hydra's own output directory, so
        # `.hydra/`, the job log and the artifacts below stay together. Outside
        # Hydra the same layout is derived from the config.
        run_dir = log_cfg.run_dir or os.path.join(log_cfg.log_dir, run_path(self.config))
        logger = ResultLogger(run_dir)
        # Saved after the seed is resolved, so the snapshot records the seed
        # actually used even when it was drawn at random.
        config_path = os.path.join(logger.run_dir, "config.yaml")
        OmegaConf.save(OmegaConf.structured(self.config), config_path)
        write_run_info(self.config, logger.run_dir)
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

    def _reward_obs_scale(self) -> float:
        """Scale the reward model applies to stored observations.

        Normalization follows the selected algorithm's setting so the reward
        model sees inputs on the same scale as the policy does. Every algorithm
        node declares `normalize_obs`, including `random`.
        """
        algorithm_cfg = getattr(self.config, "algorithm", None)
        normalize = getattr(algorithm_cfg, "normalize_obs", False)
        return 1.0 / 255.0 if normalize else 1.0

    def _format_reward_obs(self, obs: dict, agent_id: str) -> np.ndarray:
        """Format an observation for the reward model.

        Returns the frame in its native `uint8` dtype. Scaling happens inside
        `RewardModel.forward`, on device, via `obs_scale` -- so the preference
        buffer holds a quarter of the bytes a float32 copy would need, and the
        same saving applies to every host-to-device transfer. The effective
        input scale is unchanged: `_reward_obs_scale() * frame`.
        """
        return np.ascontiguousarray(obs[agent_id]["curr_obs"])

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
            reward_model = RewardModel(
                obs_shape=obs_shape,
                num_actions=num_actions,
                obs_scale=self._reward_obs_scale(),
            )
            rm_trainer = RewardModelTrainer(
                reward_model,
                lr=rm_cfg.lr,
                device=rm_cfg.device,
                use_amp=rm_cfg.use_amp,
                chunk_size=rm_cfg.chunk_size,
                weight_decay=rm_cfg.weight_decay,
                max_grad_norm=rm_cfg.max_grad_norm,
                delta_temperature=rm_cfg.delta_temperature,
                grad_checkpoint=rm_cfg.grad_checkpoint,
                tie_tolerance=rm_cfg.tie_tolerance,
            )
            reward_model = rm_trainer.reward_model
            pref_buffer = PreferenceBuffer(
                rm_cfg.max_episodes_in_buffer,
                max_steps_per_sequence=rm_cfg.max_steps_per_sequence,
                store_max_steps_per_agent=rm_cfg.store_max_steps_per_agent,
            )
            self.console.section("Reward model")
            self.console.info(f"mode / phi     : {rm_cfg.mode} / {rm_cfg.phi}")
            self.console.info(f"input          : obs{tuple(obs_shape)} x {num_actions} actions")
            self.console.info(f"device         : {rm_trainer.device}")
            self.console.info(
                f"warmup         : collecting preferences for {rm_cfg.warmup_episodes}"
                " episodes before the first update"
            )
            self.console.info(
                f"updates        : every {rm_cfg.update_every_env_steps} env steps,"
                f" {rm_cfg.train_steps_per_update} steps of {rm_cfg.batch_pairs} pairs"
            )
            self.console.warn("policies learn from predicted rewards, not environment rewards")
        else:
            self.console.section("Reward model")
            self.console.info("disabled -- policies learn from environment rewards")

        self.console.section("Training")
        self.console.start_episodes(self.config.episodes)
        recent_rewards: list = []

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
                step_agent_ids = []
                step_obs_imgs = []
                step_actions = []
                if rm_cfg.enabled:
                    for agent_id in obs.keys():
                        obs_img = self._format_reward_obs(obs, agent_id)
                        step_agent_ids.append(agent_id)
                        step_obs_imgs.append(obs_img)
                        step_actions.append(actions[agent_id])
                        episode_agent_trajs[agent_id].append((obs_img, actions[agent_id]))
                next_obs, rewards, dones, infos = self.env.step(actions)
                if rm_cfg.enabled:
                    # One forward pass and one device sync for the whole step,
                    # instead of one per agent.
                    if step_obs_imgs:
                        predicted = reward_model.predict_batch(
                            np.stack(step_obs_imgs, axis=0), step_actions
                        )
                        pred_rewards = {
                            agent_id: float(value)
                            for agent_id, value in zip(step_agent_ids, predicted)
                        }
                    else:
                        pred_rewards = {}
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
                self.console.info_once(
                    "rm-warmup-done",
                    f"warmup complete after episode {episode + 1}"
                    f" -- reward model training starts, buffer holds {len(pref_buffer)} episodes",
                )
                if (global_step - last_rm_update_step) >= rm_cfg.update_every_env_steps:
                    rm_metrics = rm_trainer.train(
                        pref_buffer,
                        phi_key=rm_cfg.phi,
                        mode=rm_cfg.mode,
                        batch_pairs=rm_cfg.batch_pairs,
                        train_steps=rm_cfg.train_steps_per_update,
                    )
                    last_rm_update_step = global_step
                    # Later updates are visible in the per-episode stats; the
                    # first one is worth calling out because it is the moment
                    # the predicted rewards stop being random.
                    self.console.info_once(
                        "rm-first-update",
                        "first reward model update: "
                        + (_format_metrics(rm_metrics) or "no metrics returned"),
                    )
            if rm_cfg.enabled and (episode + 1) % rm_cfg.save_every_episodes == 0:
                reward_model_path = os.path.join(self.logger.run_dir, "reward_model.pt")
                reward_model.save(reward_model_path)
                self.console.info(f"reward model checkpoint saved at episode {episode + 1}")

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
            video_path = video_recorder.finish()
            if video_path is not None:
                self.console.info(f"video saved: {video_path}")
            self.console.episode_end(episode, self._episode_stats(payload))
            recent_rewards.append(payload["reward_mean"])

        self.console.close()
        self.console.section("Finished")
        if hasattr(self.algorithm, "save"):
            model_path = os.path.join(self.logger.run_dir, "model_last.pt")
            self.algorithm.save(model_path)
            self.console.info(f"policy saved   : {model_path}")
        if rm_cfg.enabled and reward_model is not None:
            reward_model_path = os.path.join(self.logger.run_dir, "reward_model_last.pt")
            reward_model.save(reward_model_path)
            self.console.info(f"reward model   : {reward_model_path}")

        video_recorder.finalize()
        self.logger.close()
        if recent_rewards:
            # The last tenth of the run, but never fewer than 10 episodes --
            # one episode's reward is too noisy to summarize a run with.
            window = min(len(recent_rewards), max(10, len(recent_rewards) // 10))
            tail = recent_rewards[-window:]
            self.console.info(f"mean reward over the last {len(tail)} episodes: {np.mean(tail):.2f}")
        self.console.info(f"env steps      : {global_step}")
        self.console.info(f"artifacts      : {self.logger.run_dir}")

    def _episode_stats(self, payload: dict) -> dict:
        """The handful of numbers worth watching live, out of everything logged."""
        stats = {"reward": payload["reward_mean"]}
        if "reward_pred_mean" in payload:
            stats["pred"] = payload["reward_pred_mean"]
        social = payload.get("social_metrics")
        if isinstance(social, dict):
            for name in ("efficiency", "equality", "sustainability", "peace"):
                if isinstance(social.get(name), (int, float)):
                    stats[name[:3]] = float(social[name])
        elif isinstance(social, (list, tuple)) and len(social) == 4:
            for name, value in zip(("eff", "equ", "sus", "pea"), social):
                stats[name] = float(value)
        rm_metrics = payload.get("algo_metrics", {}).get("reward_model") or {}
        if isinstance(rm_metrics.get("loss"), (int, float)):
            stats["rm_loss"] = float(rm_metrics["loss"])
        return stats
