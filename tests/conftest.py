"""Shared test fixtures."""

import random
from types import SimpleNamespace

import pytest


class FakeEnv:
    """Minimal stand-in for `VecCommonsEnv`.

    Exists so algorithm construction (`Algorithm.on_env_ready`) can be tested
    without building the full env stack (no PettingZoo/gym env, no rendering,
    no GPU). `on_env_ready` touches exactly these duck-typed attributes:

    - `observation_space["curr_obs"].shape`
    - `action_space.n`
    - `agent_ids`, `num_envs`, `num_agents`

    If `on_env_ready` starts reading anything else from the env, add it here
    -- do not grow this into a general-purpose mock environment.
    """

    observation_space = {"curr_obs": SimpleNamespace(shape=(15, 15, 3))}
    # `sample()` as well as `n`: the random policy draws from the space.
    action_space = SimpleNamespace(n=8, sample=lambda: random.randrange(8))
    agent_ids = ["agent-0", "agent-1"]
    num_agents = 2
    num_envs = 1

    def __init__(self, num_envs: int = 1) -> None:
        self.num_envs = num_envs


@pytest.fixture
def fake_env() -> FakeEnv:
    return FakeEnv()
