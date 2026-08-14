"""The per-iteration statistics accumulator.

The load-bearing property is the last group: a statistic conditioned on an event
that did not happen must be absent, never NaN.
"""

import math

import numpy as np
import pytest

from commons_game_marp.train.episode_stats import (
    ACTION_GROUPS,
    NEARBY_BUCKETS,
    EpisodeStats,
)

NUM_ACTIONS = 8


def _stats(num_rows=4, track_reward_model=False):
    return EpisodeStats(
        num_actions=NUM_ACTIONS, num_rows=num_rows, track_reward_model=track_reward_model
    )


def _record(stats, actions, env_rewards, nearby=None, last_in_cluster=None, pred=None):
    n = len(actions)
    stats.record_step(
        actions=actions,
        env_rewards=env_rewards,
        nearby_apples=nearby if nearby is not None else [0] * n,
        last_in_cluster=last_in_cluster if last_in_cluster is not None else [False] * n,
        pred_rewards=pred,
    )


class TestActionSection:
    def test_fractions_sum_to_one(self):
        stats = _stats(num_rows=4)
        _record(stats, [0, 1, 2, 3], [0, 0, 0, 0])
        _record(stats, [4, 5, 6, 7], [0, 0, 0, 0])
        action = stats.result()["action"]
        total = sum(action[name] for name in ACTION_GROUPS)
        assert total == pytest.approx(1.0)

    def test_turn_merges_both_rotations(self):
        stats = _stats(num_rows=4)
        _record(stats, [5, 6, 5, 6], [0, 0, 0, 0])
        assert stats.result()["action"]["turn"] == pytest.approx(1.0)
        assert stats.result()["action"]["stay"] == 0.0

    def test_fire_fraction(self):
        stats = _stats(num_rows=4)
        _record(stats, [7, 7, 4, 4], [0, 0, 0, 0])
        assert stats.result()["action"]["fire"] == pytest.approx(0.5)

    def test_entropy_of_uniform_raw_actions_is_log_num_actions(self):
        """Entropy is over raw actions, so it lines up with algo/entropy."""
        stats = _stats(num_rows=NUM_ACTIONS)
        _record(stats, list(range(NUM_ACTIONS)), [0] * NUM_ACTIONS)
        assert stats.result()["action"]["entropy"] == pytest.approx(math.log(NUM_ACTIONS))

    def test_entropy_of_a_collapsed_policy_is_zero(self):
        stats = _stats(num_rows=4)
        _record(stats, [4, 4, 4, 4], [0, 0, 0, 0])
        assert stats.result()["action"]["entropy"] == pytest.approx(0.0)


class TestHarvestSection:
    def test_apples_per_agent_and_rate(self):
        stats = _stats(num_rows=2)
        _record(stats, [0, 0], [1.0, 0.0])
        _record(stats, [0, 0], [1.0, 1.0])
        harvest = stats.result()["harvest"]
        # 3 harvests over 2 rows, 2 steps each.
        assert harvest["apples_per_agent"] == pytest.approx(1.5)
        assert harvest["harvest_rate"] == pytest.approx(3 / 4)

    def test_nearby_apples_means(self):
        stats = _stats(num_rows=2)
        _record(stats, [0, 0], [1.0, 0.0], nearby=[4, 0])
        _record(stats, [0, 0], [0.0, 0.0], nearby=[2, 2])
        harvest = stats.result()["harvest"]
        assert harvest["nearby_apples_mean"] == pytest.approx(2.0)
        # Only the single harvest step counts here.
        assert harvest["nearby_apples_on_harvest"] == pytest.approx(4.0)

    def test_last_in_cluster_rate_counts_only_harvests(self):
        stats = _stats(num_rows=2)
        _record(stats, [0, 0], [1.0, 1.0], last_in_cluster=[True, False])
        harvest = stats.result()["harvest"]
        assert harvest["last_in_cluster_rate"] == pytest.approx(0.5)

    def test_no_harvest_omits_conditional_statistics(self):
        stats = _stats(num_rows=2)
        _record(stats, [0, 0], [0.0, 0.0], nearby=[3, 3])
        harvest = stats.result()["harvest"]
        assert harvest["apples_per_agent"] == 0.0
        assert "last_in_cluster_rate" not in harvest
        assert "nearby_apples_on_harvest" not in harvest


class TestRewardModelSections:
    def test_absent_when_not_tracking(self):
        stats = _stats(num_rows=2, track_reward_model=False)
        _record(stats, [0, 0], [1.0, 0.0])
        sections = stats.result()
        assert not [name for name in sections if name.startswith("rm_")]

    def test_pred_rewards_required_when_tracking(self):
        stats = _stats(num_rows=2, track_reward_model=True)
        with pytest.raises(ValueError):
            _record(stats, [0, 0], [1.0, 0.0])

    def test_mean_predicted_reward_per_action(self):
        stats = _stats(num_rows=4, track_reward_model=True)
        _record(stats, [0, 1, 7, 5], [0, 0, 0, 0], pred=[1.0, 2.0, 3.0, 4.0])
        _record(stats, [0, 1, 7, 6], [0, 0, 0, 0], pred=[3.0, 4.0, 5.0, 6.0])
        on_action = stats.result()["rm_on_action"]
        assert on_action["move_left"] == pytest.approx(2.0)
        assert on_action["move_right"] == pytest.approx(3.0)
        assert on_action["fire"] == pytest.approx(4.0)
        # turn merges the 4.0 (action 5) and 6.0 (action 6) samples.
        assert on_action["turn"] == pytest.approx(5.0)
        assert "move_up" not in on_action

    def test_outcome_split_and_delta(self):
        stats = _stats(num_rows=4, track_reward_model=True)
        _record(stats, [0] * 4, [1.0, 1.0, 0.0, 0.0], pred=[5.0, 7.0, 1.0, 3.0])
        result = stats.result()
        assert result["rm_outcome_avg"]["apple_eaten"] == pytest.approx(6.0)
        assert result["rm_outcome_avg"]["no_apple_eaten"] == pytest.approx(2.0)
        assert result["rm_outcome_avg"]["delta"] == pytest.approx(4.0)
        assert result["rm_outcome_std"]["apple_eaten"] == pytest.approx(1.0)
        assert result["rm_outcome_std"]["no_apple_eaten"] == pytest.approx(1.0)
        assert result["rm_outcome_std"]["delta"] == pytest.approx(0.0)
        # pooled std is 1.0, so separation equals delta here.
        assert result["rm_outcome"]["separation"] == pytest.approx(4.0)

    def test_separation_omitted_when_both_sides_are_constant(self):
        stats = _stats(num_rows=2, track_reward_model=True)
        _record(stats, [0, 0], [1.0, 0.0], pred=[5.0, 1.0])
        result = stats.result()
        assert result["rm_outcome_avg"]["delta"] == pytest.approx(4.0)
        assert "separation" not in result.get("rm_outcome", {})

    def test_nearby_buckets_partition_the_harvests(self):
        stats = _stats(num_rows=4, track_reward_model=True)
        _record(
            stats,
            [0] * 4,
            [1.0, 1.0, 1.0, 1.0],
            nearby=[0, 2, 3, 9],
            pred=[10.0, 20.0, 30.0, 40.0],
        )
        buckets = stats.result()["rm_by_nearby_apples"]
        assert buckets["0"] == pytest.approx(10.0)
        assert buckets["1-2"] == pytest.approx(20.0)
        assert buckets["3-4"] == pytest.approx(30.0)
        assert buckets["5+"] == pytest.approx(40.0)

    def test_empty_buckets_are_absent_not_nan(self):
        stats = _stats(num_rows=1, track_reward_model=True)
        _record(stats, [0], [1.0], nearby=[0], pred=[10.0])
        buckets = stats.result()["rm_by_nearby_apples"]
        assert set(buckets) == {"0"}
        for label, _, _ in NEARBY_BUCKETS:
            if label != "0":
                assert label not in buckets

    def test_bucket_labels_cover_every_count(self):
        """No harvest may fall between two buckets."""
        for count in range(0, 12):
            matched = [
                label
                for label, low, high in NEARBY_BUCKETS
                if count >= low and (high is None or count <= high)
            ]
            assert len(matched) == 1, (count, matched)

    def test_step_correlation_of_a_perfect_predictor(self):
        stats = _stats(num_rows=4, track_reward_model=True)
        _record(stats, [0] * 4, [1.0, 0.0, 1.0, 0.0], pred=[2.0, 0.0, 2.0, 0.0])
        assert stats.result()["rm_pred"]["step_corr"] == pytest.approx(1.0)

    def test_step_correlation_omitted_for_a_constant_predictor(self):
        stats = _stats(num_rows=4, track_reward_model=True)
        _record(stats, [0] * 4, [1.0, 0.0, 1.0, 0.0], pred=[2.0, 2.0, 2.0, 2.0])
        assert "step_corr" not in stats.result()["rm_pred"]

    def test_prediction_distribution(self):
        stats = _stats(num_rows=3, track_reward_model=True)
        _record(stats, [0] * 3, [0.0] * 3, pred=[-1.0, 0.0, 4.0])
        pred = stats.result()["rm_pred"]
        assert pred["mean"] == pytest.approx(1.0)
        assert pred["min"] == pytest.approx(-1.0)
        assert pred["max"] == pytest.approx(4.0)


class TestNoNaNsEverywhere:
    def test_no_section_value_is_nan_on_random_input(self):
        rng = np.random.default_rng(7)
        for _ in range(50):
            rows = int(rng.integers(1, 6))
            stats = _stats(num_rows=rows, track_reward_model=True)
            for _ in range(int(rng.integers(1, 5))):
                _record(
                    stats,
                    actions=rng.integers(0, NUM_ACTIONS, size=rows).tolist(),
                    # Mostly zeros, so harvest-conditioned subsets are often
                    # empty -- which is the case that used to produce NaN.
                    env_rewards=(rng.random(rows) > 0.85).astype(float).tolist(),
                    nearby=rng.integers(0, 3, size=rows).tolist(),
                    last_in_cluster=(rng.random(rows) > 0.5).tolist(),
                    pred=rng.normal(size=rows).tolist(),
                )
            for name, section in stats.result().items():
                for key, value in section.items():
                    assert np.isfinite(value), f"{name}/{key} = {value}"

    def test_empty_accumulator_returns_nothing(self):
        assert _stats().result() == {}
