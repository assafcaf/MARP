"""End-to-end: the diagnostics reach metrics.jsonl and the TensorBoard event file.

These assert on a real run rather than a mock, because the parts most likely to
break are the ones that read live environment state -- agent positions, the
world map, the info dicts -- and none of that is exercised by unit tests of the
accumulator.
"""

import csv
import glob
import json
import os

import pytest

from commons_game_marp.train.config import (
    EnvConfig,
    IPPOConfig,
    LoggingConfig,
    RandomConfig,
    RewardModelConfig,
    TrainerConfig,
)
from commons_game_marp.train.episode_stats import ACTION_GROUPS
from commons_game_marp.train.trainer import Trainer


def make_config(
    tmp_path,
    algorithm=None,
    num_envs=2,
    episodes=2,
    reward_model=False,
    detailed_metrics=True,
    penalty=False,
    histogram_every=0,
    agent_details=True,
):
    return TrainerConfig(
        episodes=episodes,
        seed=0,
        env=EnvConfig(
            map_type="small",
            num_agents=2,
            agent_view_range=3,
            ep_length=6,
            num_envs=num_envs,
            penalty=penalty,
        ),
        algorithm=algorithm or RandomConfig(device="cpu"),
        logging=LoggingConfig(
            log_dir=str(tmp_path),
            run_dir=str(tmp_path / "run"),
            console="quiet",
            video_enabled=False,
            log_agent_episode_details=agent_details,
            detailed_metrics=detailed_metrics,
            histogram_every_n_episodes=histogram_every,
        ),
        reward_model=RewardModelConfig(
            enabled=reward_model,
            warmup_episodes=1,
            update_every_env_steps=1,
            batch_pairs=2,
            train_steps_per_update=1,
            max_episodes_in_buffer=8,
            device="cpu",
        ),
    )


def read_records(tmp_path):
    with open(os.path.join(tmp_path, "run", "metrics.jsonl")) as handle:
        return [json.loads(line) for line in handle if line.strip()]


@pytest.fixture(scope="module")
def detailed_run(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("detailed")
    Trainer(make_config(tmp_path)).train()
    return read_records(tmp_path)


class TestBehaviourSections:
    def test_action_section_is_a_distribution(self, detailed_run):
        for record in detailed_run:
            action = record["sections"]["action"]
            assert set(ACTION_GROUPS).issubset(action)
            assert sum(action[name] for name in ACTION_GROUPS) == pytest.approx(1.0)
            assert action["entropy"] >= 0.0

    def test_harvest_section_is_present_and_bounded(self, detailed_run):
        for record in detailed_run:
            harvest = record["sections"]["harvest"]
            assert harvest["apples_per_agent"] >= 0.0
            assert 0.0 <= harvest["harvest_rate"] <= 1.0
            assert harvest["nearby_apples_mean"] >= 0.0

    def test_nearby_apples_are_actually_counted(self, detailed_run):
        """A run on the small map must see apples; a flat zero would mean the
        stencil is reading the wrong array or the wrong position."""
        assert any(
            record["sections"]["harvest"]["nearby_apples_mean"] > 0
            for record in detailed_run
        )

    def test_no_section_value_is_null_or_nan(self, detailed_run):
        for record in detailed_run:
            for name, section in record["sections"].items():
                for key, value in section.items():
                    assert value is not None, f"{name}/{key}"
                    assert value == value, f"{name}/{key} is NaN"


class TestEnvironmentReward:
    def test_reward_env_keys_are_logged(self, detailed_run):
        for record in detailed_run:
            assert "reward_env_sum" in record
            assert "reward_env_mean" in record
            assert set(record["reward_env_per_agent"]) == set(record["reward_per_agent"])

    def test_matches_policy_reward_without_a_penalty(self, detailed_run):
        for record in detailed_run:
            assert record["reward_env_mean"] == pytest.approx(record["reward_mean"])

    def test_diverges_from_policy_reward_under_a_penalty(self, tmp_path):
        """With the FIRE penalty on, the two must not be the same series.

        This is the whole point of logging both: `train/reward_mean` under a
        penalty is not the harvest.
        """
        Trainer(make_config(tmp_path, penalty=True, episodes=4)).train()
        records = read_records(tmp_path)
        fired = any(record["sections"]["action"]["fire"] > 0 for record in records)
        assert fired, "the random policy never fired; the test proves nothing"
        assert any(
            record["reward_env_mean"] != record["reward_mean"] for record in records
        )
        # The penalty only ever subtracts.
        assert all(
            record["reward_env_mean"] >= record["reward_mean"] for record in records
        )


class TestTimeSection:
    def test_throughput_is_logged(self, detailed_run):
        for record in detailed_run:
            assert record["time"]["fps"] > 0
            assert record["time"]["iteration_sec"] > 0
            assert record["time"]["elapsed_hours"] >= 0


class TestSocialMetricExtensions:
    def test_new_env_metrics_reach_the_payload(self, detailed_run):
        for record in detailed_run:
            social = record["social_metrics"]
            for key in (
                "apple_stock_mean",
                "depletion_fraction",
                "fire_hit_rate",
                "reward_std_agent",
            ):
                assert key in social, key


@pytest.fixture(scope="module")
def rm_run(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("rm")
    Trainer(
        make_config(
            tmp_path,
            algorithm=IPPOConfig(n_steps=4, batch_size=4, update_epochs=1, device="cpu"),
            reward_model=True,
            episodes=2,
        )
    ).train()
    return read_records(tmp_path)


class TestRewardModelDiagnostics:
    def test_step_level_sections_are_emitted(self, rm_run):
        for record in rm_run:
            sections = record["sections"]
            assert sections["rm_on_action"], "no per-action predicted reward"
            assert "apple_eaten" in sections["rm_outcome_avg"] or (
                "no_apple_eaten" in sections["rm_outcome_avg"]
            )
            assert "mean" in sections["rm_pred"]

    def test_predicted_reward_is_reported_per_action_taken(self, rm_run):
        """Every action the policy actually used gets a conditional mean."""
        for record in rm_run:
            action = record["sections"]["action"]
            on_action = record["sections"]["rm_on_action"]
            for name in ACTION_GROUPS:
                if action[name] > 0:
                    assert name in on_action, name


class TestDetailedMetricsSwitch:
    def test_off_removes_the_sections_but_keeps_the_rest(self, tmp_path):
        Trainer(make_config(tmp_path, detailed_metrics=False)).train()
        records = read_records(tmp_path)
        assert records
        for record in records:
            assert "sections" not in record
            # Everything that predates this feature must survive.
            assert "reward_mean" in record
            assert "social_metrics" in record
            assert "time" in record

    def test_agent_csv_still_written_with_detailed_metrics_off(self, tmp_path):
        Trainer(make_config(tmp_path, detailed_metrics=False)).train()
        files = glob.glob(str(tmp_path / "run" / "extended_info" / "*.csv"))
        assert files
        with open(files[0]) as handle:
            rows = list(csv.DictReader(handle))
        assert rows
        assert "nearby_apples" in rows[0]
        assert "env_reward" in rows[0]


class TestHistograms:
    def test_written_on_the_configured_interval(self, tmp_path):
        Trainer(
            make_config(tmp_path, num_envs=1, episodes=4, histogram_every=2)
        ).train()
        # Reading the event file back is the only way to see histograms; the
        # tag list is enough.
        from tensorboard.backend.event_processing.event_accumulator import (
            EventAccumulator,
        )

        accumulator = EventAccumulator(str(tmp_path / "run" / "tensorboard"))
        accumulator.Reload()
        tags = accumulator.Tags()
        histogram_tags = set(tags.get("histograms", [])) | set(
            tags.get("distributions", [])
        )
        assert "reward/agent_hist" in histogram_tags

    def test_absent_when_disabled(self, tmp_path):
        Trainer(make_config(tmp_path, num_envs=1, episodes=2, histogram_every=0)).train()
        from tensorboard.backend.event_processing.event_accumulator import (
            EventAccumulator,
        )

        accumulator = EventAccumulator(str(tmp_path / "run" / "tensorboard"))
        accumulator.Reload()
        tags = accumulator.Tags()
        assert not tags.get("histograms")


def test_tensorboard_receives_the_new_scalar_families(tmp_path):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    Trainer(make_config(tmp_path)).train()
    accumulator = EventAccumulator(str(tmp_path / "run" / "tensorboard"))
    accumulator.Reload()
    tags = set(accumulator.Tags()["scalars"])

    expected = {
        "action/fire",
        "action/entropy",
        "harvest/harvest_rate",
        "harvest/nearby_apples_mean",
        "social/apple_stock_mean",
        "social/depletion_fraction",
        "train/reward_env_mean",
        "reward_env/agent_agent-0",
        "time/fps",
        # Pre-existing tags must survive.
        "train/reward_mean",
        "social/efficiency",
    }
    assert expected.issubset(tags), sorted(expected - tags)
