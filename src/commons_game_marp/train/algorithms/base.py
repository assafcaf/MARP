from abc import ABC, abstractmethod
from typing import Any, Dict, List

import numpy as np


class Algorithm(ABC):
    """Policy interface over a flat batch of `num_envs * num_agents` rows.

    Every array is ordered env-major, agent-minor -- row = env_idx *
    num_agents + agent_idx -- matching `VecCommonsEnv`. There is no separate
    single-environment path: `num_envs=1` is the same code, so the path every
    existing config uses is the path the tests cover.
    """

    def __init__(self, config: Any):
        self.config = config

    @abstractmethod
    def on_env_ready(self, env) -> None:
        """Build networks and buffers.

        `env` is a `VecCommonsEnv`, exposing `agent_ids`, `num_envs`,
        `num_agents`, `observation_space` and `action_space`.
        """

    @abstractmethod
    def act(self, observations: np.ndarray, step: int) -> np.ndarray:
        """(N, ...) observations -> (N,) int64 actions."""

    @abstractmethod
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
        """Record one transition for every row."""

    @abstractmethod
    def on_episode_end(self, episode: int) -> Dict[str, Any]:
        pass

    def uses_external_loop(self) -> bool:
        return True

    def save(self, path: str) -> None:
        return None
