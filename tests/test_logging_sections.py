"""ResultLogger's TensorBoard routing.

A fake writer stands in for `SummaryWriter` so the assertions are about tags and
values rather than about parsing event files.
"""

import json
import os

import pytest

from commons_game_marp.train.logging_utils import ResultLogger


class FakeWriter:
    def __init__(self):
        self.scalars = {}
        self.histograms = {}
        self.closed = False

    def add_scalar(self, tag, value, step):
        self.scalars.setdefault(tag, []).append((step, value))

    def add_histogram(self, tag, values, step):
        self.histograms.setdefault(tag, []).append((step, list(values)))

    def close(self):
        self.closed = True


@pytest.fixture
def logger(tmp_path):
    result = ResultLogger(str(tmp_path / "run"))
    result._writer = FakeWriter()
    yield result
    result.close()


def _tags(logger):
    return logger._writer.scalars


class TestExistingTagsStillEmitted:
    def test_core_training_tags(self, logger):
        logger.log_episode(
            {
                "episode": 3,
                "steps": 600,
                "reward_sum": 12.0,
                "reward_mean": 3.0,
                "reward_per_agent": {"agent-0": 3.0},
                "social_metrics": {"efficiency": 4.0, "peace": 1.0},
                "algo_metrics": {"avg_loss": 0.5, "epsilon": 0.1},
            }
        )
        tags = _tags(logger)
        assert tags["train/steps"] == [(3, 600.0)]
        assert tags["train/reward_sum"] == [(3, 12.0)]
        assert tags["reward/agent_agent-0"] == [(3, 3.0)]
        assert tags["social/efficiency"] == [(3, 4.0)]
        assert tags["train/loss"] == [(3, 0.5)]
        assert tags["algo/epsilon"] == [(3, 0.1)]

    def test_reward_env_payload_keys_are_wired_up(self, logger):
        """These branches existed but nothing produced the keys."""
        logger.log_episode(
            {
                "episode": 1,
                "reward_env_sum": 9.0,
                "reward_env_mean": 3.0,
                "reward_env_per_agent": {"agent-1": 3.0},
            }
        )
        tags = _tags(logger)
        assert tags["train/reward_env_sum"] == [(1, 9.0)]
        assert tags["train/reward_env_mean"] == [(1, 3.0)]
        assert tags["reward_env/agent_agent-1"] == [(1, 3.0)]


class TestSectionRouting:
    def test_sections_become_tag_families(self, logger):
        logger.log_episode(
            {
                "episode": 5,
                "sections": {
                    "action": {"fire": 0.25, "entropy": 1.7},
                    "harvest": {"harvest_rate": 0.01},
                    "rm_by_nearby_apples": {"3-4": 2.5},
                },
            }
        )
        tags = _tags(logger)
        assert tags["action/fire"] == [(5, 0.25)]
        assert tags["action/entropy"] == [(5, 1.7)]
        assert tags["harvest/harvest_rate"] == [(5, 0.01)]
        assert tags["rm_by_nearby_apples/3-4"] == [(5, 2.5)]

    def test_time_section_is_routed(self, logger):
        logger.log_episode({"episode": 2, "time": {"fps": 1234.0}})
        assert _tags(logger)["time/fps"] == [(2, 1234.0)]

    def test_non_numeric_and_non_finite_values_are_dropped(self, logger):
        logger.log_episode(
            {
                "episode": 1,
                "sections": {
                    "action": {
                        "fire": float("nan"),
                        "stay": float("inf"),
                        "turn": "0.5",
                        "flagged": True,
                        "move_up": 0.5,
                    }
                },
            }
        )
        tags = _tags(logger)
        assert set(tags) == {"action/move_up"}

    def test_missing_or_malformed_sections_are_ignored(self, logger):
        logger.log_episode({"episode": 1})
        logger.log_episode({"episode": 2, "sections": "not a dict"})
        logger.log_episode({"episode": 3, "sections": {"action": 5}})
        assert _tags(logger) == {}


class TestNestedAlgoMetrics:
    def test_per_agent_dicts_are_flattened(self, logger):
        logger.log_episode(
            {
                "episode": 4,
                "algo_metrics": {
                    "ent_coef": 0.05,
                    "ent_coef_per_agent": {"agent-0": 0.04, "agent-1": 0.06},
                },
            }
        )
        tags = _tags(logger)
        assert tags["algo/ent_coef"] == [(4, 0.05)]
        assert tags["algo/ent_coef_per_agent/agent-0"] == [(4, 0.04)]
        assert tags["algo/ent_coef_per_agent/agent-1"] == [(4, 0.06)]

    def test_reward_model_dict_keeps_its_own_prefix(self, logger):
        """It must not also be flattened under algo/."""
        logger.log_episode(
            {"episode": 4, "algo_metrics": {"reward_model": {"loss": 0.3}}}
        )
        tags = _tags(logger)
        assert tags["reward_model/loss"] == [(4, 0.3)]
        assert "algo/reward_model/loss" not in tags


class TestHistograms:
    def test_values_are_forwarded(self, logger):
        logger.log_histograms(7, {"rm_pred/hist": [1.0, 2.0, 3.0]})
        assert logger._writer.histograms["rm_pred/hist"] == [(7, [1.0, 2.0, 3.0])]

    def test_non_finite_values_are_filtered(self, logger):
        logger.log_histograms(1, {"rm_pred/hist": [1.0, float("nan"), 2.0]})
        assert logger._writer.histograms["rm_pred/hist"] == [(1, [1.0, 2.0])]

    def test_an_all_empty_histogram_is_skipped(self, logger):
        logger.log_histograms(1, {"rm_pred/hist": [float("nan")]})
        assert logger._writer.histograms == {}


def test_sections_are_persisted_to_metrics_jsonl(tmp_path):
    logger = ResultLogger(str(tmp_path / "run"))
    logger._writer = FakeWriter()
    logger.log_episode({"episode": 0, "sections": {"action": {"fire": 0.5}}})
    logger.close()
    with open(os.path.join(logger.run_dir, "metrics.jsonl"), encoding="utf-8") as handle:
        record = json.loads(handle.readline())
    assert record["sections"]["action"]["fire"] == 0.5
