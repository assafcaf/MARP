from abc import ABC, abstractmethod
from typing import Dict, Any


class Algorithm(ABC):
    def __init__(self, config: Any):
        self.config = config

    @abstractmethod
    def on_env_ready(self, env) -> None:
        pass

    @abstractmethod
    def act(self, observations: Dict[str, Any], step: int) -> Dict[str, int]:
        pass

    @abstractmethod
    def observe(
        self,
        observations: Dict[str, Any],
        actions: Dict[str, int],
        rewards: Dict[str, float],
        next_observations: Dict[str, Any],
        dones: Dict[str, bool],
        infos: Dict[str, Any],
        step: int,
    ) -> None:
        pass

    @abstractmethod
    def on_episode_end(self, episode: int) -> Dict[str, Any]:
        pass

    def uses_external_loop(self) -> bool:
        return True

    def train(self, env, logger, config) -> None:
        raise NotImplementedError

    def save(self, path: str) -> None:
        return None
