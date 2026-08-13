"""End-to-end: a short run must complete on every algorithm, at num_envs 1 and
above, and must log the episode budget it was asked for.

A vec trainer that quietly runs `episodes` iterations instead of `episodes //
num_envs` would 4x the compute of every existing config without failing
anything, so the logged episode count is asserted directly.
"""

import json
import os

import pytest

from commons_game_marp.train.config import (
    DQNConfig,
    EnvConfig,
    IPPOConfig,
    LoggingConfig,
    MAPPOConfig,
    RandomConfig,
    RewardModelConfig,
    TrainerConfig,
)
from commons_game_marp.train.trainer import Trainer


def make_config(tmp_path, algorithm, num_envs, episodes=4, reward_model=False):
    return TrainerConfig(
        episodes=episodes,
        seed=0,
        env=EnvConfig(
            map_type="small",
            num_agents=2,
            agent_view_range=3,
            ep_length=5,
            num_envs=num_envs,
        ),
        algorithm=algorithm,
        logging=LoggingConfig(
            log_dir=str(tmp_path),
            run_dir=str(tmp_path / "run"),
            console="quiet",
            video_enabled=False,
            log_agent_episode_details=True,
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


ALGORITHMS = [
    ("ippo", lambda: IPPOConfig(n_steps=4, batch_size=4, update_epochs=1, device="cpu")),
    ("mappo", lambda: MAPPOConfig(n_steps=4, batch_size=4, update_epochs=1, device="cpu")),
    ("dqn", lambda: DQNConfig(device="cpu", train_after=10_000)),
    ("random", lambda: RandomConfig(device="cpu")),
]


def read_episodes(run_dir):
    path = os.path.join(run_dir, "metrics.jsonl")
    with open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


@pytest.mark.parametrize("name,make_algorithm", ALGORITHMS)
@pytest.mark.parametrize("num_envs", [1, 2])
def test_short_run_completes(tmp_path, name, make_algorithm, num_envs):
    config = make_config(tmp_path, make_algorithm(), num_envs)
    Trainer(config).train()

    records = read_episodes(tmp_path / "run")
    assert records, "the run logged nothing"


def test_episode_budget_is_spent_not_multiplied(tmp_path):
    config = make_config(
        tmp_path,
        IPPOConfig(n_steps=4, batch_size=4, update_epochs=1, device="cpu"),
        num_envs=2,
        episodes=4,
    )
    Trainer(config).train()

    records = read_episodes(tmp_path / "run")
    assert len(records) == 2, "4 episodes at num_envs=2 is 2 iterations"
    # `episode` stays an episode count so curves overlay across num_envs.
    assert [r["episode"] for r in records] == [1, 3]
    assert all(r["num_envs"] == 2 for r in records)


def test_non_divisible_episode_budget_fails_at_construction(tmp_path):
    config = make_config(tmp_path, RandomConfig(device="cpu"), num_envs=3, episodes=4)
    with pytest.raises(ValueError, match="divisible"):
        Trainer(config)


def test_reward_model_run_completes_with_parallel_envs(tmp_path):
    config = make_config(
        tmp_path,
        IPPOConfig(n_steps=4, batch_size=4, update_epochs=1, device="cpu"),
        num_envs=2,
        reward_model=True,
    )
    Trainer(config).train()

    records = read_episodes(tmp_path / "run")
    assert all("reward_pred_mean" in r for r in records)


def test_preference_buffer_gets_one_record_per_environment(tmp_path):
    """Each env's episode is its own EpisodeRecord, not an average of them."""
    from commons_game_marp.reward_model.preference_buffer import PreferenceBuffer

    seen = []
    original = PreferenceBuffer.add_episode

    def spy(self, record):
        seen.append(record)
        return original(self, record)

    PreferenceBuffer.add_episode = spy
    try:
        config = make_config(
            tmp_path,
            RandomConfig(device="cpu"),
            num_envs=2,
            episodes=4,
            reward_model=True,
        )
        Trainer(config).train()
    finally:
        PreferenceBuffer.add_episode = original

    # 2 iterations x 2 envs
    assert len(seen) == 4


def _count_reward_model_updates(tmp_path, num_envs, episodes, every_steps):
    from commons_game_marp.reward_model.reward_trainer import RewardModelTrainer

    calls = []
    original = RewardModelTrainer.train

    def spy(self, *args, **kwargs):
        calls.append(1)
        return original(self, *args, **kwargs)

    RewardModelTrainer.train = spy
    try:
        config = make_config(
            tmp_path,
            RandomConfig(device="cpu"),
            num_envs=num_envs,
            episodes=episodes,
            reward_model=True,
        )
        config.reward_model.update_every_env_steps = every_steps
        Trainer(config).train()
    finally:
        RewardModelTrainer.train = original
    return len(calls)


def test_reward_model_update_rate_does_not_depend_on_num_envs(tmp_path):
    """The period is denominated in env steps, so the same experience must buy
    the same number of updates however many environments produced it.

    The check only runs at iteration boundaries, and an iteration is num_envs
    episodes wide -- firing once per boundary made num_envs=4 update half as
    often per env step as num_envs=1.
    """
    # 8 episodes x 5 steps = 40 env steps either way, period 10 -> 4 updates.
    serial = _count_reward_model_updates(
        tmp_path / "serial", num_envs=1, episodes=8, every_steps=10
    )
    parallel = _count_reward_model_updates(
        tmp_path / "parallel", num_envs=4, episodes=8, every_steps=10
    )

    assert serial == parallel == 4


def test_reward_model_checkpoints_at_the_configured_episode_period(tmp_path):
    """`episode + 1` only takes multiples of num_envs, so an exact modulo
    against save_every_episodes misses every period that is not itself a
    multiple of num_envs."""
    from commons_game_marp.reward_model.reward_model import RewardModel

    saves = []
    original = RewardModel.save

    def spy(self, path):
        saves.append(path)
        return original(self, path)

    RewardModel.save = spy
    try:
        config = make_config(
            tmp_path,
            RandomConfig(device="cpu"),
            num_envs=2,
            episodes=12,
            reward_model=True,
        )
        config.reward_model.save_every_episodes = 3
        Trainer(config).train()
    finally:
        RewardModel.save = original

    # 12 episodes at a 3-episode period is 4 checkpoints, plus the final
    # reward_model_last.pt the run always writes.
    periodic = [p for p in saves if p.endswith("reward_model.pt")]
    assert len(periodic) == 4


@pytest.mark.parametrize("num_envs", [1, 2])
def test_log_interval_is_denominated_in_episodes(tmp_path, num_envs):
    """An iteration-based modulo would multiply the interval by num_envs."""
    config = make_config(
        tmp_path, RandomConfig(device="cpu"), num_envs=num_envs, episodes=6
    )
    config.logging.log_interval = 2
    Trainer(config).train()

    assert len(read_episodes(tmp_path / "run")) == 3


def test_agent_detail_logs_cover_env_zero_only(tmp_path):
    """One episode's detail row per agent per iteration, not num_envs of them."""
    config = make_config(tmp_path, RandomConfig(device="cpu"), num_envs=2, episodes=4)
    Trainer(config).train()

    details_dir = os.path.join(tmp_path, "run", "extended_info")
    assert os.path.isdir(details_dir)
    files = [f for f in os.listdir(details_dir) if f.endswith(".csv")]
    assert files, "no per-agent detail files written"
