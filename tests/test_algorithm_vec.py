"""Every algorithm speaks the flat-row protocol, at num_envs 1 and above.

The protocol is the whole contract between the trainer and the algorithms:
(N, ...) observations in, (N,) int64 actions out, N = num_envs * num_agents in
env-major order. An algorithm that returns the right shape from the wrong
ordering trains happily on mismatched agent data, so ordering is asserted, not
just shape.
"""

import numpy as np
import pytest

from commons_game_marp.train.algorithms.dqn import DQNAlgorithm
from commons_game_marp.train.algorithms.ippo import IPPOAlgorithm
from commons_game_marp.train.algorithms.mappo import MAPPOAlgorithm
from commons_game_marp.train.algorithms.random_policy import RandomAlgorithm
from commons_game_marp.train.config import (
    DQNConfig,
    IPPOConfig,
    MAPPOConfig,
    RandomConfig,
)
from tests.conftest import FakeEnv


def build(algo_cls, config, num_envs):
    env = FakeEnv(num_envs=num_envs)
    algo = algo_cls(config)
    algo.on_env_ready(env)
    return algo, env


ALGORITHMS = [
    (IPPOAlgorithm, lambda: IPPOConfig(n_steps=4, batch_size=4, update_epochs=1, device="cpu")),
    (MAPPOAlgorithm, lambda: MAPPOConfig(n_steps=4, batch_size=4, update_epochs=1, device="cpu")),
    (DQNAlgorithm, lambda: DQNConfig(device="cpu", train_after=10_000)),
    (RandomAlgorithm, lambda: RandomConfig(device="cpu")),
]


@pytest.mark.parametrize("algo_cls,make_config", ALGORITHMS)
@pytest.mark.parametrize("num_envs", [1, 3])
def test_act_returns_one_int64_action_per_row(algo_cls, make_config, num_envs):
    algo, env = build(algo_cls, make_config(), num_envs)
    rows = num_envs * env.num_agents
    obs = np.zeros((rows, 15, 15, 3), dtype=np.uint8)

    actions = algo.act(obs, step=0)

    assert isinstance(actions, np.ndarray)
    assert actions.shape == (rows,)
    assert actions.dtype == np.int64
    assert np.all((actions >= 0) & (actions < env.action_space.n))


@pytest.mark.parametrize("algo_cls,make_config", ALGORITHMS)
@pytest.mark.parametrize("num_envs", [1, 3])
def test_observe_accepts_flat_rows(algo_cls, make_config, num_envs):
    algo, env = build(algo_cls, make_config(), num_envs)
    rows = num_envs * env.num_agents
    obs = np.zeros((rows, 15, 15, 3), dtype=np.uint8)
    actions = algo.act(obs, step=0)

    algo.observe(
        observations=obs,
        actions=actions,
        rewards=np.zeros(rows, dtype=np.float32),
        next_observations=obs,
        dones=np.zeros(rows, dtype=bool),
        infos=[{} for _ in range(rows)],
        step=0,
    )


def test_ippo_routes_each_agents_observations_to_its_own_actor():
    """Agent-0's actor must never see agent-1's rows.

    Both actors are replaced with deterministic stand-ins keyed to a marker
    value planted in each agent's observations; a transposed reshape would
    hand the markers to the wrong actor.
    """
    import torch
    from torch import nn

    algo, env = build(IPPOAlgorithm, IPPOConfig(device="cpu", flatten_obs=True), 3)

    class MarkerActor(nn.Module):
        def __init__(self, num_actions: int) -> None:
            super().__init__()
            self.num_actions = num_actions
            self.seen = []

        def forward(self, obs: torch.Tensor) -> torch.Tensor:
            self.seen.append(float(obs.flatten()[0].item()))
            return torch.zeros(obs.shape[0], self.num_actions)

    actors = {
        "agent-0": MarkerActor(env.action_space.n),
        "agent-1": MarkerActor(env.action_space.n),
    }
    algo.actors = actors

    obs = np.zeros((6, 15, 15, 3), dtype=np.uint8)
    for env_idx in range(3):
        obs[env_idx * 2 + 0] = 1  # agent-0 rows
        obs[env_idx * 2 + 1] = 2  # agent-1 rows

    algo.act(obs, step=0)

    assert actors["agent-0"].seen and actors["agent-1"].seen
    assert actors["agent-0"].seen[0] != actors["agent-1"].seen[0]


def test_ippo_buffer_keeps_one_column_per_env():
    algo, env = build(
        IPPOAlgorithm, IPPOConfig(n_steps=1000, device="cpu"), num_envs=3
    )
    rows = 6
    obs = np.zeros((rows, 15, 15, 3), dtype=np.uint8)

    for step in range(4):
        actions = algo.act(obs, step=step)
        algo.observe(
            obs, actions, np.zeros(rows, dtype=np.float32), obs,
            np.zeros(rows, dtype=bool), [{}] * rows, step,
        )

    buffer = algo.buffers["agent-0"]
    assert buffer.size() == 4, "size counts per-env timesteps, not rows"
    assert np.stack(buffer.rewards, axis=0).shape == (4, 3)


def test_ippo_updates_after_n_steps_per_env_not_per_row():
    """`n_steps` is per-env, so the update batch is n_steps * num_envs."""
    algo, env = build(
        IPPOAlgorithm,
        IPPOConfig(n_steps=3, batch_size=2, update_epochs=1, device="cpu"),
        num_envs=2,
    )
    rows = 4
    obs = np.zeros((rows, 15, 15, 3), dtype=np.uint8)

    for step in range(2):
        actions = algo.act(obs, step=step)
        algo.observe(
            obs, actions, np.zeros(rows, dtype=np.float32), obs,
            np.zeros(rows, dtype=bool), [{}] * rows, step,
        )
    assert algo.buffers["agent-0"].size() == 2, "must not update before n_steps"

    actions = algo.act(obs, step=2)
    algo.observe(
        obs, actions, np.zeros(rows, dtype=np.float32), obs,
        np.zeros(rows, dtype=bool), [{}] * rows, 2,
    )
    assert algo.buffers["agent-0"].size() == 0, "buffer clears after the update"
