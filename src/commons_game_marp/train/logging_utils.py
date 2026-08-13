import csv
import json
import math
import os
import time
from typing import Any, Dict, Sequence, Tuple

import numpy as np
from torch.utils.tensorboard import SummaryWriter


def _is_number(value: Any) -> bool:
    """A real, finite scalar.

    Bools are excluded even though `True` is an `int`: a flag logged as a 1.0
    scalar reads as a measurement, and non-finite values are dropped because a
    single NaN makes TensorBoard's y-axis autoscale useless for the rest of the
    run.
    """
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


class ResultLogger:
    def __init__(self, run_dir: str):
        # Under Hydra this is `hydra.runtime.output_dir`, which already exists
        # and already holds `.hydra/` and the job log.
        self.run_dir = run_dir
        os.makedirs(self.run_dir, exist_ok=True)
        self.metrics_path = os.path.join(self.run_dir, "metrics.jsonl")
        self._metrics_file = open(self.metrics_path, "a", encoding="utf-8")
        self.start_time = time.time()
        self._writer = SummaryWriter(os.path.join(self.run_dir, "tensorboard"))
        # Agent-specific episode detail files (CSV writers and file handles)
        self._agent_episode_files: Dict[str, Any] = {}
        self._agent_csv_writers: Dict[str, csv.DictWriter] = {}
        self._agent_headers_written: Dict[str, bool] = {}

    def log_config(self, config_payload: Dict[str, Any]) -> None:
        config_path = os.path.join(self.run_dir, "config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config_payload, f, indent=2, sort_keys=True)

    def log_episode(self, payload: Dict[str, Any]) -> None:
        payload = dict(payload)
        payload["wall_time_sec"] = round(time.time() - self.start_time, 3)
        self._metrics_file.write(json.dumps(payload) + "\n")
        self._metrics_file.flush()
        self._log_tensorboard(payload)

    def _log_tensorboard(self, payload: Dict[str, Any]) -> None:
        step = int(payload.get("episode", 0))
        scalar_pairs: list[Tuple[str, float]] = []
        if "steps" in payload:
            scalar_pairs.append(("train/steps", float(payload["steps"])))
        if "reward_sum" in payload:
            scalar_pairs.append(("train/reward_sum", float(payload["reward_sum"])))
        if "reward_mean" in payload:
            scalar_pairs.append(("train/reward_mean", float(payload["reward_mean"])))
        if "reward_pred_sum" in payload:
            scalar_pairs.append(("train/reward_pred_sum", float(payload["reward_pred_sum"])))
        if "reward_pred_mean" in payload:
            scalar_pairs.append(("train/reward_pred_mean", float(payload["reward_pred_mean"])))
        if "reward_env_sum" in payload:
            scalar_pairs.append(("train/reward_env_sum", float(payload["reward_env_sum"])))
        if "reward_env_mean" in payload:
            scalar_pairs.append(("train/reward_env_mean", float(payload["reward_env_mean"])))
        for tag, value in scalar_pairs:
            self._writer.add_scalar(tag, value, step)

        reward_per_agent = payload.get("reward_per_agent")
        if isinstance(reward_per_agent, dict):
            for agent_id, reward in reward_per_agent.items():
                self._writer.add_scalar(f"reward/agent_{agent_id}", float(reward), step)
        pred_reward_per_agent = payload.get("reward_pred_per_agent")
        if isinstance(pred_reward_per_agent, dict):
            for agent_id, reward in pred_reward_per_agent.items():
                self._writer.add_scalar(f"reward_pred/agent_{agent_id}", float(reward), step)
        env_reward_per_agent = payload.get("reward_env_per_agent")
        if isinstance(env_reward_per_agent, dict):
            for agent_id, reward in env_reward_per_agent.items():
                self._writer.add_scalar(f"reward_env/agent_{agent_id}", float(reward), step)

        social_metrics = payload.get("social_metrics")
        if isinstance(social_metrics, (list, tuple)) and len(social_metrics) == 4:
            names = ("efficiency", "equality", "sustainability", "peace")
            for name, value in zip(names, social_metrics):
                self._writer.add_scalar(f"social/{name}", float(value), step)
        elif isinstance(social_metrics, dict):
            for name, value in social_metrics.items():
                if _is_number(value):
                    self._writer.add_scalar(f"social/{name}", float(value), step)

        algo_metrics = payload.get("algo_metrics")
        if isinstance(algo_metrics, dict):
            if "avg_loss" in algo_metrics and _is_number(algo_metrics["avg_loss"]):
                self._writer.add_scalar("train/loss", float(algo_metrics["avg_loss"]), step)
            if "loss" in algo_metrics and _is_number(algo_metrics["loss"]):
                self._writer.add_scalar("train/loss", float(algo_metrics["loss"]), step)
            reward_model_metrics = algo_metrics.get("reward_model")
            if isinstance(reward_model_metrics, dict):
                for name, value in reward_model_metrics.items():
                    if _is_number(value):
                        self._writer.add_scalar(f"reward_model/{name}", float(value), step)
            for name, value in algo_metrics.items():
                if _is_number(value):
                    self._writer.add_scalar(f"algo/{name}", float(value), step)
                elif name != "reward_model" and isinstance(value, dict):
                    # Per-agent algorithm metrics -- IPPO's `ent_coef_per_agent`
                    # is the one that matters, because a single averaged
                    # coefficient hides one agent's controller running away.
                    # These used to be dropped for not being scalars.
                    for agent_id, agent_value in value.items():
                        if _is_number(agent_value):
                            self._writer.add_scalar(
                                f"algo/{name}/{agent_id}", float(agent_value), step
                            )

        self._log_sections(payload.get("sections"), step)
        self._log_sections({"time": payload.get("time")}, step)

    def _log_sections(self, sections: Any, step: int) -> None:
        """Route `{section: {tag: value}}` to `section/tag` scalars.

        The accumulator in `episode_stats` already decides *which* statistics
        exist this iteration -- conditional ones vanish when their subset is
        empty -- so this deliberately does not fill gaps with zeros. A gap in
        the series is the honest rendering of "no harvest with five apples
        nearby happened in these 600 steps".
        """
        if not isinstance(sections, dict):
            return
        for section, values in sections.items():
            if not isinstance(values, dict):
                continue
            for name, value in values.items():
                if _is_number(value):
                    self._writer.add_scalar(f"{section}/{name}", float(value), step)

    def log_histograms(self, step: int, histograms: Dict[str, Sequence[float]]) -> None:
        """Write distribution histograms for the given step.

        Separate from `log_episode` because histograms cost orders of magnitude
        more disk than a scalar and are written on their own, coarser interval.
        """
        for tag, values in histograms.items():
            array = [float(v) for v in values if _is_number(v)]
            if array:
                self._writer.add_histogram(tag, np.asarray(array), step)

    def log_agent_episode_details(self, agent_id: str, episode: int, episode_data: Dict[str, Any]) -> None:
        """Log detailed episode data for a specific agent to a separate CSV file.
        
        Each step in the episode becomes a row in the CSV, with episode-level
        information repeated in each row.
        
        Args:
            agent_id: The agent identifier
            episode: Episode number
            episode_data: Dictionary containing step-by-step data (actions, rewards, etc.)
        """
        if agent_id not in self._agent_episode_files:
            # Create extended_info subdirectory for CSV files
            extended_info_dir = os.path.join(self.run_dir, "extended_info")
            os.makedirs(extended_info_dir, exist_ok=True)
            agent_log_path = os.path.join(extended_info_dir, f"agent_{agent_id}_episodes.csv")
            self._agent_episode_files[agent_id] = open(agent_log_path, "a", encoding="utf-8", newline="")
            self._agent_headers_written[agent_id] = False
        
        # Extract episode-level data
        wall_time_sec = round(time.time() - self.start_time, 3)
        total_steps = episode_data.get("total_steps", 0)
        total_reward = episode_data.get("total_reward", 0.0)
        total_predicted_reward = episode_data.get("total_predicted_reward")
        social_metrics = episode_data.get("social_metrics", {})
        steps_data = episode_data.get("steps", [])
        
        # Build CSV fieldnames
        fieldnames = [
            "episode", "step", "action", "reward", "env_reward", "apple_eaten",
            "nearby_apples", "ate_last_apple_in_cluster"
        ]
        # Check if any step has predicted_reward (step-level predicted reward)
        has_predicted_reward = any("predicted_reward" in step_info for step_info in steps_data)
        if has_predicted_reward:
            # Insert after "reward" column
            reward_idx = fieldnames.index("reward")
            fieldnames.insert(reward_idx + 1, "predicted_reward")
        
        # Initialize CSV writer if needed
        if agent_id not in self._agent_csv_writers:
            writer = csv.DictWriter(self._agent_episode_files[agent_id], fieldnames=fieldnames)
            self._agent_csv_writers[agent_id] = writer
        
        writer = self._agent_csv_writers[agent_id]
        
        # Write header if this is the first time
        if not self._agent_headers_written[agent_id]:
            writer.writeheader()
            self._agent_headers_written[agent_id] = True
        
        # Write one row per step
        for step_info in steps_data:
            row = {}
            # Only include fields that are in the writer's fieldnames
            for fieldname in writer.fieldnames:
                if fieldname == "episode":
                    row[fieldname] = episode
                elif fieldname == "step":
                    row[fieldname] = step_info.get("step", "")
                elif fieldname == "action":
                    row[fieldname] = step_info.get("action", "")
                elif fieldname == "reward":
                    row[fieldname] = step_info.get("reward", "")
                elif fieldname == "env_reward":
                    row[fieldname] = step_info.get("env_reward", "")
                elif fieldname == "predicted_reward":
                    row[fieldname] = step_info.get("predicted_reward", "")
                elif fieldname == "apple_eaten":
                    row[fieldname] = step_info.get("apple_eaten", "")
                elif fieldname == "nearby_apples":
                    row[fieldname] = step_info.get("nearby_apples", "")
                elif fieldname == "ate_last_apple_in_cluster":
                    row[fieldname] = step_info.get("ate_last_apple_in_cluster", "")
            
            writer.writerow(row)
        
        self._agent_episode_files[agent_id].flush()

    def close(self) -> None:
        if self._metrics_file:
            self._metrics_file.close()
        for agent_file in self._agent_episode_files.values():
            if agent_file:
                agent_file.close()
        if self._writer:
            self._writer.close()
