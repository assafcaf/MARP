from typing import Any, Dict, List

import numpy as np

from .base import Algorithm


class RandomAlgorithm(Algorithm):
    def __init__(self, config: Any):
        super().__init__(config)
        self._env = None

    def on_env_ready(self, env) -> None:
        self._env = env

    def act(self, observations: np.ndarray, step: int) -> np.ndarray:
        return np.array(
            [self._env.action_space.sample() for _ in range(observations.shape[0])],
            dtype=np.int64,
        )

    def observe(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_observations: np.ndarray,
        dones: np.ndarray,
        infos: List[Dict[str, Any]],
        step: int,
    ) -> None:
        return None

    def on_episode_end(self, episode: int) -> Dict[str, Any]:
        return {}
