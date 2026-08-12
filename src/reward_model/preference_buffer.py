import random
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class EpisodeRecord:
    agent_trajs: Dict[str, List[Tuple[np.ndarray, int]]]
    metrics: Dict[str, float]


# Default max steps for temporal subsampling (None = no limit)
DEFAULT_MAX_STEPS: Optional[int] = 256


class PreferenceBuffer:
    def __init__(self, max_episodes: int, max_steps_per_sequence: Optional[int] = DEFAULT_MAX_STEPS):
        self._episodes: Deque[EpisodeRecord] = deque(maxlen=max_episodes)
        self.max_steps_per_sequence = max_steps_per_sequence

    def add_episode(self, record: EpisodeRecord) -> None:
        self._episodes.append(record)

    def __len__(self) -> int:
        return len(self._episodes)

    def _as_list(self) -> List[EpisodeRecord]:
        return list(self._episodes)

    def sample_episode_pairs(self, batch_pairs: int) -> List[Tuple[EpisodeRecord, EpisodeRecord]]:
        episodes = self._as_list()
        n = len(episodes)
        if n < 2:
            return []
        pairs = []
        for _ in range(batch_pairs):
            i, j = random.sample(range(n), 2)
            pairs.append((episodes[i], episodes[j]))
        return pairs

    def _subsample(
        self, sequence: List[Tuple[np.ndarray, int]], max_steps: Optional[int]
    ) -> List[Tuple[np.ndarray, int]]:
        """Subsample a sequence to at most max_steps using uniform spacing."""
        if max_steps is None or len(sequence) <= max_steps:
            return sequence
        indices = np.linspace(0, len(sequence) - 1, max_steps, dtype=np.int64)
        return [sequence[i] for i in indices]

    def aggregate_episode(
        self, record: EpisodeRecord, max_steps: Optional[int] = None
    ) -> List[Tuple[np.ndarray, int]]:
        """Aggregate all agent trajectories, optionally subsampling to max_steps."""
        if max_steps is None:
            max_steps = self.max_steps_per_sequence
        merged: List[Tuple[np.ndarray, int]] = []
        for agent_id in sorted(record.agent_trajs.keys()):
            merged.extend(record.agent_trajs.get(agent_id, []))
        return self._subsample(merged, max_steps)

    def sample_agent_trajectory(
        self, record: EpisodeRecord, max_steps: Optional[int] = None
    ) -> List[Tuple[np.ndarray, int]]:
        """Sample a single agent's trajectory, optionally subsampling to max_steps."""
        if max_steps is None:
            max_steps = self.max_steps_per_sequence
        if not record.agent_trajs:
            return []
        agent_ids = list(record.agent_trajs.keys())
        agent_id = random.choice(agent_ids)
        traj = record.agent_trajs.get(agent_id, [])
        return self._subsample(traj, max_steps)

    def sample_narrow_view_pairs(
        self, batch_pairs: int
    ) -> List[Tuple[List[Tuple[np.ndarray, int]], List[Tuple[np.ndarray, int]], EpisodeRecord, EpisodeRecord]]:
        pairs = self.sample_episode_pairs(batch_pairs)
        output = []
        for ep_i, ep_j in pairs:
            traj_i = self.sample_agent_trajectory(ep_i)
            traj_j = self.sample_agent_trajectory(ep_j)
            output.append((traj_i, traj_j, ep_i, ep_j))
        return output
