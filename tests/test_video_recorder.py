"""Video capture cadence under parallel environments.

`Trainer.train` advances its episode counter one *iteration* at a time:

    episode = (iteration + 1) * num_envs - 1

so at `num_envs=4` it only ever takes the values 3, 7, 11, ... and never lands
on a round multiple of `video_every_n_episodes`. The recorder used to test
`episode % every_n == 0`, which is unsatisfiable whenever the stride and the
interval share no compatible residue -- at num_envs=4 every index is odd and
every 100-multiple is even, so a 10000-episode run produced exactly one video
(the `total_episodes - 1` special case) instead of a hundred.

These tests pin the cadence against the real strided schedule rather than
against a counter that increments by one.
"""

import pytest

from commons_game_marp.train.video_utils import VideoRecorder


def episode_schedule(episodes: int, num_envs: int):
    """The exact sequence of episode indices `Trainer.train` passes to the
    recorder, mirroring trainer.py's `(iteration + 1) * num_envs - 1`."""
    return [(i + 1) * num_envs - 1 for i in range(episodes // num_envs)]


def recorded(episodes: int, num_envs: int, every_n: int = 100, **kwargs):
    recorder = VideoRecorder(
        base_dir="/unused",
        enabled=True,
        every_n_episodes=every_n,
        max_steps=600,
        fps=10,
        keep_frames=False,
        total_episodes=episodes,
        stride=num_envs,
        **kwargs,
    )
    return [ep for ep in episode_schedule(episodes, num_envs) if recorder.should_record(ep)]


@pytest.mark.parametrize("num_envs", [1, 2, 4, 5])
def test_records_once_per_interval_for_any_num_envs(num_envs):
    """One video per `every_n` episodes, whatever the stride.

    This is the regression: at num_envs>1 this used to collapse to a single
    video for the whole run.
    """
    episodes = 10000
    hits = recorded(episodes, num_envs)

    # 100 interval boundaries in 10000 episodes. The last boundary (9900) and
    # the always-recorded final episode may or may not coincide, so allow the
    # count to be one over.
    assert 100 <= len(hits) <= 101, f"num_envs={num_envs} produced {len(hits)} videos"


@pytest.mark.parametrize("num_envs", [1, 2, 4, 5])
def test_each_interval_is_covered_exactly_once(num_envs):
    """Every 100-episode window gets exactly one capture -- no gaps, no bursts."""
    episodes = 10000
    hits = [ep for ep in recorded(episodes, num_envs) if ep != episodes - 1]

    windows = sorted(ep // 100 for ep in hits)
    assert windows == sorted(set(windows)), "two captures landed in one window"
    assert windows == list(range(1, 100)) or windows == list(range(1, 101))


def test_stride_one_preserves_exact_legacy_cadence():
    """The serial path must be unchanged, not merely equivalent."""
    hits = recorded(10000, num_envs=1)
    assert [ep for ep in hits if ep != 9999][:5] == [100, 200, 300, 400, 500]


@pytest.mark.parametrize("num_envs", [1, 2, 4, 5])
def test_opening_iterations_are_not_recorded(num_envs):
    """The first window is skipped: no video before the first real boundary."""
    hits = recorded(10000, num_envs)
    assert min(hits) >= 100, f"recorded episode {min(hits)} before the first boundary"


@pytest.mark.parametrize("num_envs", [1, 2, 4, 5])
def test_final_episode_always_recorded(num_envs):
    """The run's last episode is captured regardless of where boundaries fell."""
    episodes = 10000
    assert episodes - 1 in recorded(episodes, num_envs)


def test_disabled_records_nothing():
    recorder = VideoRecorder(
        base_dir="/unused",
        enabled=False,
        every_n_episodes=100,
        max_steps=600,
        fps=10,
        keep_frames=False,
        total_episodes=10000,
        stride=4,
    )
    assert not any(recorder.should_record(ep) for ep in episode_schedule(10000, 4))


def test_should_record_is_pure():
    """Repeated calls for the same episode agree.

    `start()` calls this once per iteration today, but a stateful cursor would
    make the predicate answer differently on a second call and silently break
    any caller that checks before starting.
    """
    recorder = VideoRecorder(
        base_dir="/unused",
        enabled=True,
        every_n_episodes=100,
        max_steps=600,
        fps=10,
        keep_frames=False,
        total_episodes=10000,
        stride=4,
    )
    for ep in (99, 103, 203, 204):
        assert recorder.should_record(ep) == recorder.should_record(ep)
