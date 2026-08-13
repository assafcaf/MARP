"""Per-episode resource and fairness metrics computed by the environment.

These land on `EpisodeRecord.metrics` as well as in TensorBoard, so the tests
also pin the one property that makes adding them safe: the oracle's preference
labels do not move.
"""

import math

import numpy as np
import pytest

from commons_game_marp.env.commons_env import EMPTY_METRICS, HarvestCommonsEnv
from commons_game_marp.env.maps import SMALL_HARVEST_MAP
from commons_game_marp.reward_model.oracle import compute_phi

PHI_KEYS = [
    "efficiency",
    "efficiency_x_peace",
    "efficiency_x_peace_x_equality",
    "efficiency_x_equality",
    "efficiency_x_sustainability",
    "equality_x_peace",
    "efficiency_x_peace_x_equality_x_sustainability",
]

NEW_KEYS = [
    "fire_hit_rate",
    "apples_eaten",
    "apples_spawned",
    "apple_stock_mean",
    "apple_stock_min",
    "apple_stock_final",
    "depletion_fraction",
    "timeout_steps",
    "reward_min_agent",
    "reward_max_agent",
    "reward_std_agent",
]


def _run_episode(steps: int = 30, num_agents: int = 3, seed: int = 0):
    rng = np.random.default_rng(seed)
    env = HarvestCommonsEnv(
        ascii_map=SMALL_HARVEST_MAP, num_agents=num_agents, ep_length=steps
    )
    env.reset()
    total_positive = 0
    for _ in range(steps):
        actions = {
            agent_id: int(rng.integers(0, 8)) for agent_id in env.agents
        }
        _, _, _, infos = env.step(actions)
        total_positive += sum(1 for info in infos.values() if info["r"] > 0)
    env.compute_social_metrics()
    return env, env.get_social_metrics(), total_positive


def test_new_keys_are_present_and_finite():
    _, metrics, _ = _run_episode()
    for key in NEW_KEYS:
        assert key in metrics, key
        assert math.isfinite(float(metrics[key])), key


def test_apples_eaten_counts_positive_reward_events():
    _, metrics, total_positive = _run_episode()
    assert metrics["apples_eaten"] == float(total_positive)


def test_apple_stock_is_recorded_once_per_step():
    steps = 12
    env = HarvestCommonsEnv(ascii_map=SMALL_HARVEST_MAP, num_agents=1, ep_length=steps)
    env.reset()
    for _ in range(steps):
        env.step({"agent-0": 4})
    assert len(env.apple_stock_record) == steps
    env.compute_social_metrics()
    metrics = env.get_social_metrics()
    # A single agent standing still on the small map never empties it.
    assert metrics["apple_stock_min"] > 0
    assert metrics["depletion_fraction"] == 0.0
    assert metrics["apple_stock_mean"] >= metrics["apple_stock_min"]


def test_counters_reset_between_episodes():
    env = HarvestCommonsEnv(ascii_map=SMALL_HARVEST_MAP, num_agents=1, ep_length=5)
    env.reset()
    for _ in range(5):
        env.step({"agent-0": 4})
    env.compute_social_metrics()
    assert env.apple_stock_record == []
    assert env.apples_spawned == 0
    env.reset()
    assert env.apple_stock_record == []
    assert env.get_social_metrics() == EMPTY_METRICS


def test_fire_hit_rate_is_zero_when_nobody_fires():
    _, metrics, _ = _run_episode(steps=5, num_agents=1)
    env = HarvestCommonsEnv(ascii_map=SMALL_HARVEST_MAP, num_agents=1, ep_length=5)
    env.reset()
    for _ in range(5):
        env.step({"agent-0": 4})
    env.compute_social_metrics()
    assert env.get_social_metrics()["fire_attempts"] == 0
    assert env.get_social_metrics()["fire_hit_rate"] == 0.0


def test_reward_spread_brackets_the_mean():
    _, metrics, _ = _run_episode(num_agents=4)
    assert metrics["reward_min_agent"] <= metrics["reward_max_agent"]
    assert metrics["reward_std_agent"] >= 0.0


@pytest.mark.parametrize("phi_key", PHI_KEYS)
def test_phi_is_unaffected_by_the_new_keys(phi_key):
    """Adding keys must not move any oracle label."""
    _, metrics, _ = _run_episode()
    legacy = {
        key: metrics[key]
        for key in ("efficiency", "equality", "sustainability", "peace")
    }
    assert compute_phi(metrics, phi_key) == compute_phi(legacy, phi_key)


def test_empty_metrics_covers_every_computed_key():
    """A reset env and a finished env must report the same schema."""
    _, metrics, _ = _run_episode(steps=5, num_agents=1)
    assert set(metrics) == set(EMPTY_METRICS)
