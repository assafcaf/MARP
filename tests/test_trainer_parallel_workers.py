"""`env.num_workers` must change only where the work happens.

A full training run, not just the vector env: same seed, same config, workers
on and off, and every logged number has to match. If this passes, `num_workers`
is a pure performance knob and existing experiments stay comparable.
"""

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
from commons_game_marp.train.trainer import Trainer


def _config(tmp_path, num_workers, algorithm=None, reward_model=False, video=False):
    return TrainerConfig(
        episodes=4,
        seed=17,
        env=EnvConfig(
            map_type="small",
            num_agents=3,
            agent_view_range=3,
            ep_length=8,
            num_envs=2,
            num_workers=num_workers,
        ),
        algorithm=algorithm or RandomConfig(device="cpu"),
        logging=LoggingConfig(
            log_dir=str(tmp_path),
            run_dir=str(tmp_path / "run"),
            console="quiet",
            video_enabled=video,
            video_every_n_episodes=2,
            log_agent_episode_details=True,
            detailed_metrics=True,
        ),
        reward_model=RewardModelConfig(
            enabled=reward_model,
            warmup_episodes=1,
            batch_pairs=2,
            train_steps_per_update=1,
            max_episodes_in_buffer=8,
            device="cpu",
        ),
    )


def _records(tmp_path):
    with open(os.path.join(tmp_path, "run", "metrics.jsonl")) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _run(tmp_path, **kwargs):
    Trainer(_config(tmp_path, **kwargs)).train()
    return _records(tmp_path)


def _comparable(record):
    """Everything except wall-clock, which legitimately differs."""
    stripped = {k: v for k, v in record.items() if k not in ("time", "wall_time_sec")}
    return json.dumps(stripped, sort_keys=True)


class TestWorkersDoNotChangeResults:
    def test_random_policy_run_is_identical(self, tmp_path_factory):
        serial = _run(tmp_path_factory.mktemp("serial"), num_workers=0)
        parallel = _run(tmp_path_factory.mktemp("parallel"), num_workers=2)
        assert len(serial) == len(parallel)
        for a, b in zip(serial, parallel):
            assert _comparable(a) == _comparable(b)

    def test_one_worker_matches_in_process(self, tmp_path_factory):
        serial = _run(tmp_path_factory.mktemp("s1"), num_workers=0)
        parallel = _run(tmp_path_factory.mktemp("p1"), num_workers=1)
        for a, b in zip(serial, parallel):
            assert _comparable(a) == _comparable(b)

    def test_more_workers_than_envs_is_clamped_and_still_matches(self, tmp_path_factory):
        serial = _run(tmp_path_factory.mktemp("s8"), num_workers=0)
        parallel = _run(tmp_path_factory.mktemp("p8"), num_workers=8)
        for a, b in zip(serial, parallel):
            assert _comparable(a) == _comparable(b)

    def test_learning_run_with_reward_model_is_identical(self, tmp_path_factory):
        """The path that actually matters: a policy learning from a reward
        model, where the preference buffer holds per-env episode records."""
        kwargs = dict(
            algorithm=IPPOConfig(
                n_steps=4, batch_size=4, update_epochs=1, device="cpu"
            ),
            reward_model=True,
        )
        serial = _run(tmp_path_factory.mktemp("srm"), num_workers=0, **kwargs)
        parallel = _run(tmp_path_factory.mktemp("prm"), num_workers=2, **kwargs)
        for a, b in zip(serial, parallel):
            assert _comparable(a) == _comparable(b)


class TestRunsCompleteWithWorkers:
    def test_video_capture_works_through_workers(self, tmp_path):
        """Frames come back over the pipe rather than the worker writing them."""
        _run(tmp_path, num_workers=2, video=True)
        videos = os.path.join(tmp_path, "run", "videos")
        assert os.path.isdir(videos)
        assert os.listdir(videos)

    def test_agent_csv_is_written_with_workers(self, tmp_path):
        _run(tmp_path, num_workers=2)
        extended = os.path.join(tmp_path, "run", "extended_info")
        assert os.path.isdir(extended)
        assert os.listdir(extended)

    def test_detailed_sections_are_present(self, tmp_path):
        records = _run(tmp_path, num_workers=2)
        for record in records:
            assert "action" in record["sections"]
            assert "harvest" in record["sections"]


def test_reproducible_across_repeat_runs(tmp_path_factory):
    """Same seed twice must give the same run -- the property the seeding fix
    established, now asserted end to end through the trainer."""
    first = _run(tmp_path_factory.mktemp("r1"), num_workers=2)
    second = _run(tmp_path_factory.mktemp("r2"), num_workers=2)
    for a, b in zip(first, second):
        assert _comparable(a) == _comparable(b)
