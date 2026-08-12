"""Shared test fixtures."""

from types import SimpleNamespace

import pytest


class FakeEnv:
    """Minimal stand-in for the real Harvest environment.

    Exists so algorithm construction (`Algorithm.on_env_ready`) can be tested
    without building the full env stack (no PettingZoo/gym env, no rendering,
    no GPU). `IPPOAlgorithm.on_env_ready` and `MAPPOAlgorithm.on_env_ready`
    touch exactly three duck-typed attributes on the env, and this stub
    satisfies all three:

    - `observation_space["curr_obs"].shape`
    - `action_space.n`
    - `agents.keys()`

    If `on_env_ready` starts reading anything else from the env, add it here
    -- do not grow this into a general-purpose mock environment.
    """

    observation_space = {"curr_obs": SimpleNamespace(shape=(15, 15, 3))}
    action_space = SimpleNamespace(n=8)
    agents = {"agent-0": None, "agent-1": None}


@pytest.fixture
def fake_env() -> FakeEnv:
    return FakeEnv()
