"""The vec layer owns the dict<->flat-row translation, and nothing else does.

Row ordering is env-major/agent-minor -- row = env_idx * num_agents +
agent_idx -- copied from SuperSuit's ConcatVecEnv so the two codebases stay
comparable. A transposed ordering would still train; it would just attribute
every observation to the wrong agent, which no shape assertion would catch.
"""

import random as _random

import numpy as np
import pytest

from commons_game_marp.env.commons_env import HarvestCommonsEnv, MAP
from commons_game_marp.env.vec_env import (
    VecCommonsEnv,
    agents_to_rows,
    rows_to_agents,
)


def make_env():
    return HarvestCommonsEnv(
        ascii_map=MAP["small"],
        num_agents=2,
        render=False,
        agent_view_range=3,
        ep_length=20,
        spawn_speed="slow",
        metric="Efficiency",
        penalty=False,
    )


@pytest.fixture
def vec():
    return VecCommonsEnv(make_env, num_envs=3)


def test_rows_to_agents_uses_env_major_ordering():
    agent_ids = ["agent-0", "agent-1"]
    # 3 envs x 2 agents; value encodes (env, agent) as env * 10 + agent
    rows = np.array([0, 1, 10, 11, 20, 21], dtype=np.int64)

    per_agent = rows_to_agents(rows, num_envs=3, agent_ids=agent_ids)

    np.testing.assert_array_equal(per_agent["agent-0"], [0, 10, 20])
    np.testing.assert_array_equal(per_agent["agent-1"], [1, 11, 21])


def test_rows_to_agents_preserves_trailing_dimensions():
    rows = np.zeros((4, 5, 5, 3), dtype=np.uint8)
    per_agent = rows_to_agents(rows, num_envs=2, agent_ids=["a", "b"])
    assert per_agent["a"].shape == (2, 5, 5, 3)


def test_agents_to_rows_is_the_inverse_of_rows_to_agents():
    agent_ids = ["agent-0", "agent-1"]
    rows = np.arange(6, dtype=np.int64)
    per_agent = rows_to_agents(rows, num_envs=3, agent_ids=agent_ids)

    np.testing.assert_array_equal(
        agents_to_rows(per_agent, num_envs=3, agent_ids=agent_ids), rows
    )


def test_reset_returns_flat_rows_and_per_row_infos(vec):
    obs, infos = vec.reset()

    assert vec.num_rows == 6
    assert obs.shape == (6, *vec.observation_space["curr_obs"].shape)
    assert obs.dtype == np.uint8
    assert isinstance(infos, list) and len(infos) == 6


def test_step_returns_the_documented_shapes_and_dtypes(vec):
    vec.reset()
    actions = np.zeros(vec.num_rows, dtype=np.int64)

    obs, rewards, dones, infos = vec.step(actions)

    assert obs.shape == (6, *vec.observation_space["curr_obs"].shape)
    assert obs.dtype == np.uint8
    assert rewards.shape == (6,) and rewards.dtype == np.float32
    assert dones.shape == (6,) and dones.dtype == np.bool_
    assert isinstance(infos, list) and len(infos) == 6


def test_actions_are_dispatched_to_the_right_env_and_agent():
    """Each row's action must reach exactly one (env, agent) pair.

    Action 7 is FIRE, which the env records per agent in infos[...]['fire'].
    Firing from a single row must show up in that row's info and no other.
    """
    vec = VecCommonsEnv(make_env, num_envs=3)
    vec.reset()
    actions = np.zeros(vec.num_rows, dtype=np.int64)
    fire_row = 1 * vec.num_agents + 1  # env 1, agent 1
    actions[fire_row] = 7

    _, _, _, infos = vec.step(actions)

    assert infos[fire_row]["fire"] is True
    assert all(
        infos[r]["fire"] is False for r in range(vec.num_rows) if r != fire_row
    )


def test_single_env_reproduces_the_unwrapped_environment():
    """num_envs=1 must be the same trajectory, not merely a valid one."""
    actions = [np.array([3, 5], dtype=np.int64) for _ in range(5)]

    np.random.seed(1234)
    _random.seed(1234)
    plain = make_env()
    plain.reset(seed=None)
    plain_obs = []
    for step_actions in actions:
        obs, _, _, _ = plain.step(
            {"agent-0": int(step_actions[0]), "agent-1": int(step_actions[1])}
        )
        plain_obs.append(
            np.stack([obs["agent-0"]["curr_obs"], obs["agent-1"]["curr_obs"]])
        )

    np.random.seed(1234)
    _random.seed(1234)
    vec = VecCommonsEnv(make_env, num_envs=1)
    vec.reset()
    vec_obs = []
    for step_actions in actions:
        obs, _, _, _ = vec.step(step_actions)
        vec_obs.append(obs)

    for a, b in zip(plain_obs, vec_obs):
        np.testing.assert_array_equal(a, b)


def test_parallel_envs_diverge_under_identical_actions():
    """The shared global RNG stream is what decorrelates the copies.

    The env has no per-instance RNG (MapEnv.reset reseeds the global modules),
    so the copies differ only because they draw sequentially from one stream.
    If that ever stopped being true, every env would be a duplicate and the
    whole feature would be a no-op -- which this test is here to catch.
    """
    vec = VecCommonsEnv(make_env, num_envs=2)
    vec.reset()
    actions = np.zeros(vec.num_rows, dtype=np.int64)

    diverged = False
    for _ in range(20):
        obs, _, _, _ = vec.step(actions)
        env0 = obs[: vec.num_agents]
        env1 = obs[vec.num_agents :]
        if not np.array_equal(env0, env1):
            diverged = True
            break

    assert diverged, "parallel envs produced identical observations"


def test_compute_social_metrics_returns_one_dict_per_env(vec):
    vec.reset()
    actions = np.zeros(vec.num_rows, dtype=np.int64)
    for _ in range(3):
        vec.step(actions)

    metrics = vec.compute_social_metrics()

    assert len(metrics) == 3
    assert all("efficiency" in m for m in metrics)


def test_social_metrics_are_snapshots_not_live_references(vec):
    """`HarvestCommonsEnv.get_social_metrics` returns its own mutable dict."""
    vec.reset()
    actions = np.zeros(vec.num_rows, dtype=np.int64)
    vec.step(actions)
    first = vec.compute_social_metrics()
    first_efficiency = first[0]["efficiency"]

    for _ in range(5):
        vec.step(actions)
    vec.compute_social_metrics()

    assert first[0]["efficiency"] == first_efficiency


def test_rejects_num_envs_below_one():
    with pytest.raises(ValueError, match="num_envs"):
        VecCommonsEnv(make_env, num_envs=0)
