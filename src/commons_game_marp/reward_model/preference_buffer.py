import random
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class EpisodeRecord:
    agent_trajs: Dict[str, List[Tuple[np.ndarray, int]]]
    metrics: Dict[str, float]

    def nbytes(self) -> int:
        """Approximate resident size of the stored observations, in bytes."""
        total = 0
        for traj in self.agent_trajs.values():
            if traj:
                total += traj[0][0].nbytes * len(traj)
        return total


# Default max steps for temporal subsampling (None = no limit)
DEFAULT_MAX_STEPS: Optional[int] = 256


def _subsample(
    sequence: List[Tuple[np.ndarray, int]], max_steps: Optional[int]
) -> List[Tuple[np.ndarray, int]]:
    """Subsample a sequence to at most max_steps using uniform spacing."""
    if max_steps is None or max_steps <= 0 or len(sequence) <= max_steps:
        return sequence
    indices = np.linspace(0, len(sequence) - 1, max_steps, dtype=np.int64)
    return [sequence[i] for i in indices]


class PreferenceBuffer:
    """Ring buffer of episodes, each holding every agent's `(obs, action)` steps.

    Sizing matters more than it looks. With the shipped defaults
    (`max_episodes_in_buffer: 5000`, 600-step episodes, 5 agents, 15x15x3
    frames) the buffer holds 5000 * 600 * 5 = 15M frames. At `uint8` that is
    ~10 GB; at `float32` -- what the trainer used to hand it -- it was ~40 GB,
    i.e. an OOM well before the buffer ever filled. Two knobs bound it:
    observations are stored in their source `uint8` dtype (see
    `Trainer._format_reward_obs`), and `store_max_steps_per_agent` subsamples
    each agent's trajectory at *insertion* time rather than keeping full
    resolution in memory only to subsample it away at every training step.
    """

    def __init__(
        self,
        max_episodes: int,
        max_steps_per_sequence: Optional[int] = DEFAULT_MAX_STEPS,
        store_max_steps_per_agent: Optional[int] = None,
    ):
        self._episodes: Deque[EpisodeRecord] = deque(maxlen=max_episodes)
        self.max_steps_per_sequence = max_steps_per_sequence
        self.store_max_steps_per_agent = store_max_steps_per_agent
        self._snapshot: Optional[List[EpisodeRecord]] = None

    def add_episode(self, record: EpisodeRecord) -> None:
        if self.store_max_steps_per_agent is not None:
            record = EpisodeRecord(
                agent_trajs={
                    agent_id: _subsample(traj, self.store_max_steps_per_agent)
                    for agent_id, traj in record.agent_trajs.items()
                },
                metrics=record.metrics,
            )
        self._episodes.append(record)
        self._snapshot = None

    def __len__(self) -> int:
        return len(self._episodes)

    def nbytes(self) -> int:
        """Approximate resident size of all stored observations, in bytes."""
        return sum(record.nbytes() for record in self._episodes)

    def _as_list(self) -> List[EpisodeRecord]:
        # Cached: `sample_episode_pairs` is called once per training step
        # (50 per update by default) and rebuilding a 5000-element list each
        # time is pure overhead. Invalidated on insert.
        if self._snapshot is None:
            self._snapshot = list(self._episodes)
        return self._snapshot

    def sample_episode_pairs(self, batch_pairs: int) -> List[Tuple[EpisodeRecord, EpisodeRecord]]:
        episodes = self._as_list()
        n = len(episodes)
        if n < 2 or batch_pairs <= 0:
            return []
        # Two draws per pair, vectorised; the offset trick keeps the second
        # index distinct from the first without a rejection loop.
        first = np.random.randint(0, n, size=batch_pairs)
        offset = np.random.randint(1, n, size=batch_pairs)
        second = (first + offset) % n
        return [(episodes[i], episodes[j]) for i, j in zip(first, second)]

    def _subsample(
        self, sequence: List[Tuple[np.ndarray, int]], max_steps: Optional[int]
    ) -> List[Tuple[np.ndarray, int]]:
        return _subsample(sequence, max_steps)

    def aggregate_episode(
        self, record: EpisodeRecord, max_steps: Optional[int] = None
    ) -> List[Tuple[np.ndarray, int]]:
        """Aggregate all agent trajectories, optionally subsampling to max_steps."""
        if max_steps is None:
            max_steps = self.max_steps_per_sequence
        merged: List[Tuple[np.ndarray, int]] = []
        for agent_id in sorted(record.agent_trajs.keys()):
            merged.extend(record.agent_trajs.get(agent_id, []))
        return _subsample(merged, max_steps)

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
        return _subsample(traj, max_steps)

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
