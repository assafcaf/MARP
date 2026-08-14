"""Several environment copies stepped in lockstep, presented as flat rows.

The layout is SuperSuit's, so this repo and `DanfoaTestSOT` describe the same
thing the same way. `pettingzoo_env_to_vec_env_v1` maps the A agents of one
env onto A vector slots in `possible_agents` order, and `concat_vec_envs_v1`
stacks k of those along axis 0 -- which fixes the ordering as env-major,
agent-minor:

    row = env_idx * num_agents + agent_idx

SuperSuit is flat because SB3 trains one shared policy over every row. IPPO and
MAPPO here keep per-agent networks, so they need per-agent slices instead -- and
`rows.reshape(num_envs, num_agents, ...)` is a zero-copy view that provides
them. The flat array stays canonical; the reshape is how per-agent consumers
read it.

Deliberately *not* copied from SuperSuit: auto-reset. Episodes here are
fixed-length and lockstep, and both the social metrics and the preference
buffer's episode records depend on exact episode boundaries, so the trainer
resets explicitly.

Stepping is serial. The goal is decorrelated samples per update, not wall-clock
throughput, so there are no worker processes.
"""

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np


def rows_to_agents(
    rows: np.ndarray, num_envs: int, agent_ids: List[str]
) -> Dict[str, np.ndarray]:
    """View a flat (N, ...) batch as {agent_id: (num_envs, ...)}.

    The reshape is a view, so no data is copied; each agent's entry is a
    strided view into `rows`.
    """
    num_agents = len(agent_ids)
    reshaped = rows.reshape(num_envs, num_agents, *rows.shape[1:])
    return {agent_id: reshaped[:, i] for i, agent_id in enumerate(agent_ids)}


def agents_to_rows(
    per_agent: Dict[str, np.ndarray], num_envs: int, agent_ids: List[str]
) -> np.ndarray:
    """Inverse of `rows_to_agents`: {agent_id: (num_envs, ...)} -> (N, ...)."""
    stacked = np.stack([per_agent[agent_id] for agent_id in agent_ids], axis=1)
    return stacked.reshape(num_envs * len(agent_ids), *stacked.shape[2:])


class VecCommonsEnv:
    """`num_envs` independent environments stepped serially in one process."""

    def __init__(self, env_fn: Callable[[], Any], num_envs: int) -> None:
        if num_envs < 1:
            raise ValueError(f"num_envs must be >= 1, got {num_envs}")
        self.num_envs = int(num_envs)
        self.envs = [env_fn() for _ in range(self.num_envs)]

        # No probe reset: `MapEnv.__init__` already calls `setup_agents`, so the
        # agents and both spaces are readable immediately. Resetting here would
        # draw from the global RNG stream and desync `num_envs=1` from an
        # unwrapped environment given the same seed.
        probe = self.envs[0]
        self.agent_ids: List[str] = list(probe.agents.keys())
        self.num_agents = len(self.agent_ids)
        self.observation_space = probe.observation_space
        self.action_space = probe.action_space
        # Read by algorithms that align their update period to the episode.
        self.ep_length = int(probe.ep_length)

    @property
    def num_rows(self) -> int:
        return self.num_envs * self.num_agents

    def rows_to_agents(self, rows: np.ndarray) -> Dict[str, np.ndarray]:
        return rows_to_agents(rows, self.num_envs, self.agent_ids)

    def agents_to_rows(self, per_agent: Dict[str, np.ndarray]) -> np.ndarray:
        return agents_to_rows(per_agent, self.num_envs, self.agent_ids)

    def reset(self, seeds: Optional[Sequence[Optional[int]]] = None
              ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """Reset every environment.

        `seeds` gives one seed per copy. Passing them is what makes a run
        reproducible *and* independent of where the environments live: worker
        processes have their own RNG state, so the old approach -- reset with
        `seed=None` and let the copies decorrelate by drawing sequentially from
        the one global stream seeded in `Trainer._seed_rngs` -- cannot produce
        the same episodes once the copies are in different processes.

        `seeds=None` keeps that old behaviour, which is still what an
        unseeded, single-process run wants.
        """
        if seeds is not None and len(seeds) != self.num_envs:
            raise ValueError(
                f"expected {self.num_envs} seeds, got {len(seeds)}"
            )
        obs_rows: List[np.ndarray] = []
        infos: List[Dict[str, Any]] = []
        for env_idx, env in enumerate(self.envs):
            observations, env_infos = env.reset(
                seed=None if seeds is None else seeds[env_idx]
            )
            for agent_id in self.agent_ids:
                obs_rows.append(observations[agent_id]["curr_obs"])
                infos.append(env_infos.get(agent_id, {}))
        return np.stack(obs_rows, axis=0), infos

    def step(
        self, actions: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        actions = np.asarray(actions, dtype=np.int64)
        if actions.shape != (self.num_rows,):
            raise ValueError(
                f"expected actions of shape ({self.num_rows},), got {actions.shape}"
            )
        per_env_actions = actions.reshape(self.num_envs, self.num_agents)

        obs_rows: List[np.ndarray] = []
        rewards: List[float] = []
        dones: List[bool] = []
        infos: List[Dict[str, Any]] = []
        for env_idx, env in enumerate(self.envs):
            action_dict = {
                agent_id: int(per_env_actions[env_idx, i])
                for i, agent_id in enumerate(self.agent_ids)
            }
            observations, env_rewards, env_dones, env_infos = env.step(action_dict)
            all_done = bool(env_dones.get("__all__", False))
            for agent_id in self.agent_ids:
                obs_rows.append(observations[agent_id]["curr_obs"])
                rewards.append(float(env_rewards[agent_id]))
                dones.append(bool(env_dones.get(agent_id, False)) or all_done)
                infos.append(env_infos.get(agent_id, {}))

        return (
            np.stack(obs_rows, axis=0),
            np.asarray(rewards, dtype=np.float32),
            np.asarray(dones, dtype=bool),
            infos,
        )

    def render_frame(self, env_idx: int = 0) -> np.ndarray:
        """RGB frame of one environment, for video capture.

        Returned as an array rather than written to disk by the environment, so
        the subprocess implementation can ship the same thing back over a pipe.
        """
        return self.envs[env_idx].render()

    def close(self) -> None:
        """Present so callers can close either implementation uniformly."""

    def compute_social_metrics(self) -> List[Dict[str, float]]:
        """Per-env social metrics for the episode just finished.

        Copied, not aliased: `HarvestCommonsEnv.get_social_metrics` returns the
        env's own `self.metrics` dict, which it mutates on the next episode.
        """
        metrics = []
        for env in self.envs:
            env.compute_social_metrics()
            metrics.append(dict(env.get_social_metrics()))
        return metrics
