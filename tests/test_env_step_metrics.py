"""The environment computes its own per-step agent metrics.

The trainer used to do this by reaching into `agent.grid`, `agent.get_pos()`,
`env.world_map` and `env.apple_points`. None of that survives a process
boundary, so the environment computes it and returns two scalars per agent on
the info dict instead. These tests pin that the values are identical to what
the old reach-in computation produced.
"""

import numpy as np
import pytest

from commons_game_marp.env.commons_env import HarvestCommonsEnv
from commons_game_marp.env.maps import SMALL_HARVEST_MAP
from commons_game_marp.env.step_metrics import (
    check_ate_last_apple_in_cluster,
    count_apples_around,
    disc_offsets,
)

STAND_STILL = 4


def _make(step_metrics=True, radius=2):
    return HarvestCommonsEnv(
        ascii_map=SMALL_HARVEST_MAP,
        num_agents=3,
        agent_view_range=3,
        ep_length=40,
        step_metrics=step_metrics,
        nearby_apple_radius=radius,
    )


def test_absent_when_disabled():
    env = _make(step_metrics=False)
    env.reset(seed=0)
    _, _, _, infos = env.step({a: STAND_STILL for a in env.agents})
    for info in infos.values():
        assert "nearby_apples" not in info
        assert "ate_last_apple_in_cluster" not in info


def test_present_when_enabled():
    env = _make()
    env.reset(seed=0)
    _, _, _, infos = env.step({a: STAND_STILL for a in env.agents})
    for info in infos.values():
        assert isinstance(info["nearby_apples"], int)
        assert isinstance(info["ate_last_apple_in_cluster"], bool)


def test_matches_the_reach_in_computation_over_an_episode():
    """The values must equal what the trainer used to compute itself."""
    env = _make()
    env.reset(seed=1)
    offsets = disc_offsets(2)
    rng = np.random.default_rng(0)
    checked_harvests = 0

    for _ in range(40):
        actions = {a: int(x) for a, x in zip(env.agents, rng.integers(0, 8, 3))}
        _, _, _, infos = env.step(actions)
        for agent_id, agent in env.agents.items():
            position = agent.get_pos()
            expected_nearby = count_apples_around(agent.grid, position, offsets)
            assert infos[agent_id]["nearby_apples"] == expected_nearby

            if infos[agent_id]["r"] > 0:
                checked_harvests += 1
                expected_cluster = check_ate_last_apple_in_cluster(
                    position, env.apple_points, env.world_map
                )
                assert (
                    infos[agent_id]["ate_last_apple_in_cluster"] == expected_cluster
                )
    assert checked_harvests > 0, "no harvest happened; the cluster check went untested"


def test_cluster_flag_is_false_without_a_harvest():
    env = _make()
    env.reset(seed=0)
    for _ in range(10):
        _, _, _, infos = env.step({a: STAND_STILL for a in env.agents})
        for agent_id, info in infos.items():
            if info["r"] <= 0:
                assert info["ate_last_apple_in_cluster"] is False


@pytest.mark.parametrize("radius", [1, 2, 4])
def test_radius_is_honoured(radius):
    env = _make(radius=radius)
    env.reset(seed=2)
    offsets = disc_offsets(radius)
    _, _, _, infos = env.step({a: STAND_STILL for a in env.agents})
    for agent_id, agent in env.agents.items():
        assert infos[agent_id]["nearby_apples"] == count_apples_around(
            agent.grid, agent.get_pos(), offsets
        )


def test_wider_radius_never_counts_fewer_apples():
    counts = []
    for radius in (1, 2, 4):
        env = _make(radius=radius)
        env.reset(seed=3)
        _, _, _, infos = env.step({a: STAND_STILL for a in env.agents})
        counts.append(sum(i["nearby_apples"] for i in infos.values()))
    assert counts == sorted(counts), counts
