from typing import Any, Dict

from .base import Algorithm


class RandomAlgorithm(Algorithm):
    def __init__(self, config: Any):
        super().__init__(config)
        self._env = None

    def on_env_ready(self, env) -> None:
        self._env = env

    def act(self, observations: Dict[str, Any], step: int) -> Dict[str, int]:
        return {agent_id: self._env.action_space.sample() for agent_id in observations.keys()}

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
        return None

    def on_episode_end(self, episode: int) -> Dict[str, Any]:
        return {}
