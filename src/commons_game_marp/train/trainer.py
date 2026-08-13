import os
import random
import numpy as np
from omegaconf import OmegaConf

from ..env.commons_env import HarvestCommonsEnv, MAP
from ..env.frame_stack import FrameStackEnv
from ..env.vec_env import VecCommonsEnv
from ..reward_model.preference_buffer import EpisodeRecord, PreferenceBuffer
from ..reward_model.reward_model import RewardModel
from ..reward_model.reward_trainer import RewardModelTrainer
from .config import TrainerConfig, resolve_iterations
from .console import TrainingConsole, format_metrics as _format_metrics
from .logging_utils import ResultLogger
from .registry import build_algorithm
from .run_info import write_run_info
from .run_paths import run_path
from .video_utils import VideoRecorder
from .metrics import compute_agent_step_metrics


def _average_social_metrics(per_env: list) -> dict:
    """Mean of each social metric across the environments of one iteration."""
    if not per_env:
        return {}
    return {
        key: float(np.mean([metrics[key] for metrics in per_env]))
        for key in per_env[0]
    }


class Trainer:
    # Above this projected resident size, the preference buffer gets a warning
    # at startup. PreferenceBuffer's docstring records a real OOM from this
    # arithmetic; widening the view and stacking frames both push straight
    # into it (view 7 + 2 frames is ~20 GB at the default buffer length).
    BUFFER_WARN_BYTES = 8 * 1024**3

    # Consecutive logged episodes of a saturated, wrong-side entropy coefficient
    # before the run says so out loud.
    SATURATION_WARN_EPISODES = 20

    def __init__(self, config: TrainerConfig):
        self.config = config
        # Generate random seed if None
        if self.config.seed is None:
            self.config.seed = random.randint(0, 2**31 - 1)
        # Seed all random number generators
        self._seed_rngs(self.config.seed)
        self.console = TrainingConsole.from_config(config.logging)
        self.console.section("Setup")
        # Resolved before any environment is constructed, so a bad
        # episodes/num_envs pair fails immediately rather than after setup.
        self.iterations = resolve_iterations(
            self.config.episodes, int(self.config.env.num_envs)
        )
        self.env = self._build_env()
        self.algorithm = build_algorithm(config.algorithm)
        self.algorithm.on_env_ready(self.env)
        self.logger = self._build_logger()
        self._saturated_episodes = 0
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
        if env_cfg.num_envs > 1:
            self.console.info(
                f"parallel envs  : {env_cfg.num_envs}"
                f" ({self.iterations} iterations x {env_cfg.num_envs} episodes)"
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

    def _make_single_env(self):
        """Build one environment copy, exactly as configured."""
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
        num_frames = int(env_cfg.num_frames)
        if num_frames < 1:
            raise ValueError(f"env.num_frames must be >= 1, got {num_frames}")
        if num_frames > 1:
            return FrameStackEnv(env, num_frames)
        return env

    def _build_env(self) -> VecCommonsEnv:
        return VecCommonsEnv(self._make_single_env, int(self.config.env.num_envs))

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

    def _watch_entropy_saturation(self, algo_metrics: dict) -> None:
        """Say out loud when the entropy controller has stopped controlling.

        A coefficient pinned at a clamp while entropy sits on the wrong side of
        target is an open loop. It looks exactly like a healthy steady value in
        `metrics.jsonl`, which is how the schedule this controller replaced went
        unnoticed for 350 episodes.
        """
        saturated = algo_metrics.get("ent_coef_saturated")
        entropy = algo_metrics.get("entropy")
        target = algo_metrics.get("target_entropy")
        if not saturated or entropy is None or target is None:
            self._saturated_episodes = 0
            return
        if entropy >= target:
            self._saturated_episodes = 0
            return
        self._saturated_episodes += 1
        if self._saturated_episodes == self.SATURATION_WARN_EPISODES:
            self.console.warn(
                f"entropy coefficient has been pinned at a clamp for "
                f"{self.SATURATION_WARN_EPISODES} episodes while entropy "
                f"({entropy:.2f}) sits below target ({target:.2f}) -- the controller "
                "has stopped responding. Widen algorithm.ent_coef_max or raise "
                "algorithm.ent_coef_lr."
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

    def _format_reward_obs(self, observations: np.ndarray, row: int) -> np.ndarray:
        """Format one row's observation for the reward model.

        Returns the frame in its native `uint8` dtype. Scaling happens inside
        `RewardModel.forward`, on device, via `obs_scale` -- so the preference
        buffer holds a quarter of the bytes a float32 copy would need, and the
        same saving applies to every host-to-device transfer. The effective
        input scale is unchanged: `_reward_obs_scale() * frame`.
        """
        return np.ascontiguousarray(observations[row])

    def train(self) -> None:
        # Set total episodes for algorithms that support entropy annealing.
        # Counted in iterations, because `on_episode_end` -- which advances the
        # controllers' schedule -- fires once per iteration.
        if hasattr(self.algorithm, "set_total_episodes"):
            self.algorithm.set_total_episodes(self.iterations)

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

        vec = self.env
        num_envs = vec.num_envs
        num_agents = vec.num_agents
        agent_ids = vec.agent_ids
        num_rows = vec.num_rows

        for iteration in range(self.iterations):
            # `episode` stays an episode count, not an iteration count, so runs
            # with different num_envs overlay on one x-axis.
            episode = (iteration + 1) * num_envs - 1

            obs, infos = vec.reset()
            episode_rewards = np.zeros(num_rows, dtype=np.float64)
            episode_pred_rewards = (
                np.zeros(num_rows, dtype=np.float64) if rm_cfg.enabled else None
            )
            # One trajectory set per environment: each becomes its own
            # EpisodeRecord, so the preference buffer keeps exact episodes.
            episode_agent_trajs = (
                [{agent_id: [] for agent_id in agent_ids} for _ in range(num_envs)]
                if rm_cfg.enabled
                else None
            )
            # Per-step detail is env 0 only: logging all num_envs would multiply
            # the JSON volume for data already summarized in metrics.jsonl.
            agent_episode_details = (
                {agent_id: [] for agent_id in agent_ids}
                if self.config.logging.log_agent_episode_details
                else None
            )
            step_count = 0
            video_recorder.start(episode)

            for step in range(self.config.env.ep_length):
                actions = self.algorithm.act(obs, step)
                next_obs, rewards, dones, infos = vec.step(actions)

                if rm_cfg.enabled:
                    # One forward pass and one device sync for the whole step,
                    # covering every environment as well as every agent.
                    obs_frames = [
                        self._format_reward_obs(obs, row) for row in range(num_rows)
                    ]
                    predicted = reward_model.predict_batch(
                        np.stack(obs_frames, axis=0), [int(a) for a in actions]
                    )
                    pred_rewards = np.asarray(predicted, dtype=np.float32).reshape(num_rows)
                    for row in range(num_rows):
                        env_idx, agent_idx = divmod(row, num_agents)
                        episode_agent_trajs[env_idx][agent_ids[agent_idx]].append(
                            (obs_frames[row], int(actions[row]))
                        )
                    episode_pred_rewards += pred_rewards
                    self.algorithm.observe(
                        obs, actions, pred_rewards, next_obs, dones, infos, step
                    )
                else:
                    pred_rewards = None
                    self.algorithm.observe(
                        obs, actions, rewards, next_obs, dones, infos, step
                    )

                video_recorder.record(vec.envs[0], step)
                episode_rewards += rewards

                # Track detailed step data per agent, for environment 0 only.
                if agent_episode_details is not None:
                    for row, agent_id in enumerate(agent_ids):
                        # Check if an apple was eaten (reward > 0 indicates apple consumption)
                        apple_eaten = bool(rewards[row] > 0)

                        # Compute agent-specific metrics
                        agent = vec.envs[0].agents[agent_id]
                        metrics = compute_agent_step_metrics(
                            agent=agent,
                            env=vec.envs[0],
                            reward=float(rewards[row]),
                            apple_eaten=apple_eaten,
                            nearby_radius=2,
                        )

                        step_data = {
                            "step": step,
                            "action": int(actions[row]),
                            "reward": float(rewards[row]),
                            "done": bool(dones[row]),
                            "apple_eaten": apple_eaten,
                            "nearby_apples": metrics["nearby_apples"],
                            "ate_last_apple_in_cluster": metrics["ate_last_apple_in_cluster"],
                        }
                        if pred_rewards is not None:
                            step_data["predicted_reward"] = float(pred_rewards[row])
                        agent_episode_details[agent_id].append(step_data)

                obs = next_obs
                step_count = step + 1
                # num_envs transitions happened, not one.
                global_step += num_envs

            per_env_metrics = vec.compute_social_metrics()
            metrics = _average_social_metrics(per_env_metrics)
            if rm_cfg.enabled:
                for env_idx in range(num_envs):
                    pref_buffer.add_episode(
                        EpisodeRecord(
                            agent_trajs=episode_agent_trajs[env_idx],
                            metrics=per_env_metrics[env_idx],
                        )
                    )
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

            algo_metrics = self.algorithm.on_episode_end(iteration)
            if algo_metrics is None:
                algo_metrics = {}
            if rm_metrics:
                algo_metrics = dict(algo_metrics)
                algo_metrics["reward_model"] = rm_metrics
            self._watch_entropy_saturation(algo_metrics)

            # Averaged across the num_envs episodes this iteration completed.
            by_env_agent = episode_rewards.reshape(num_envs, num_agents)
            payload = {
                "episode": episode,
                "num_envs": num_envs,
                "steps": step_count,
                "reward_sum": float(by_env_agent.sum(axis=1).mean()),
                "reward_mean": float(episode_rewards.mean()),
                "reward_per_agent": {
                    agent_id: float(by_env_agent[:, i].mean())
                    for i, agent_id in enumerate(agent_ids)
                },
                "social_metrics": metrics,
                "algo_metrics": algo_metrics,
            }
            if rm_cfg.enabled and episode_pred_rewards is not None:
                pred_by_env_agent = episode_pred_rewards.reshape(num_envs, num_agents)
                payload["reward_pred_sum"] = float(pred_by_env_agent.sum(axis=1).mean())
                payload["reward_pred_mean"] = float(episode_pred_rewards.mean())
                payload["reward_pred_per_agent"] = {
                    agent_id: float(pred_by_env_agent[:, i].mean())
                    for i, agent_id in enumerate(agent_ids)
                }
            if iteration % self.config.logging.log_interval == 0:
                self.logger.log_episode(payload)
            # Log detailed agent episode data to separate files, environment 0.
            if agent_episode_details is not None:
                for i, agent_id in enumerate(agent_ids):
                    episode_summary = {
                        "total_steps": step_count,
                        "total_reward": float(by_env_agent[0, i]),
                        "steps": agent_episode_details[agent_id],
                    }
                    if rm_cfg.enabled and episode_pred_rewards is not None:
                        episode_summary["total_predicted_reward"] = float(
                            episode_pred_rewards.reshape(num_envs, num_agents)[0, i]
                        )
                    if metrics:
                        episode_summary["social_metrics"] = metrics
                    self.logger.log_agent_episode_details(agent_id, episode, episode_summary)
            video_path = video_recorder.finish()
            if video_path is not None:
                self.console.info(f"video saved: {video_path}")
            self.console.episode_end(
                episode, self._episode_stats(payload), advance=num_envs
            )
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
