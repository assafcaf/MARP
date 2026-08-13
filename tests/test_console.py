"""Training progress output has to stay readable in both places it lands.

Interactively it is a live progress bar; under `nohup`, a Hydra multirun or a
redirect it is a periodic status line, because a bar redrawing itself into a
file writes thousands of unreadable lines. The console resolves which of the
two it is from the stream it was handed, so both branches are testable without
a terminal.
"""

import io
from types import SimpleNamespace

from commons_game_marp.train.console import TrainingConsole


class _TtyStream(io.StringIO):
    """A stream that claims to be a terminal, as `sys.stdout` would."""

    def isatty(self) -> bool:
        return True


def test_auto_mode_uses_a_progress_bar_on_a_terminal():
    console = TrainingConsole(mode="auto", stream=_TtyStream())

    assert console.uses_bar


def test_auto_mode_falls_back_to_status_lines_when_output_is_redirected():
    """The case that matters for logged runs: a bar in a file is noise."""
    console = TrainingConsole(mode="auto", stream=io.StringIO())

    assert not console.uses_bar


def test_bar_mode_overrides_the_terminal_check():
    console = TrainingConsole(mode="bar", stream=io.StringIO())

    assert console.uses_bar


def test_plain_mode_overrides_the_terminal_check():
    console = TrainingConsole(mode="plain", stream=_TtyStream())

    assert not console.uses_bar


def test_sections_and_messages_reach_the_stream():
    stream = io.StringIO()
    console = TrainingConsole(mode="plain", stream=stream)

    console.section("Reward model")
    console.info("warmup for 50 episodes")

    output = stream.getvalue()
    assert "Reward model" in output
    assert "warmup for 50 episodes" in output


def test_status_lines_follow_the_configured_cadence():
    stream = io.StringIO()
    console = TrainingConsole(mode="plain", status_every=5, stream=stream)
    console.start_episodes(10)

    for episode in range(10):
        console.episode_end(episode, {"reward": 1.0})

    status_lines = [line for line in stream.getvalue().splitlines() if "ep " in line]
    assert len(status_lines) == 2
    assert "ep 5/10" in status_lines[0]
    assert "ep 10/10" in status_lines[1]


def test_the_last_episode_always_reports_even_off_cadence():
    """Otherwise a run whose length is not a multiple of the cadence ends
    silently, with the final numbers never printed."""
    stream = io.StringIO()
    console = TrainingConsole(mode="plain", status_every=100, stream=stream)
    console.start_episodes(3)

    for episode in range(3):
        console.episode_end(episode, {"reward": 1.0})

    assert "ep 3/3" in stream.getvalue()


def test_status_lines_report_the_metrics_they_were_given():
    stream = io.StringIO()
    console = TrainingConsole(mode="plain", status_every=1, stream=stream)
    console.start_episodes(1)

    console.episode_end(0, {"reward": 12.345, "eff": 0.5})

    output = stream.getvalue()
    assert "reward=12.35" in output
    assert "eff=0.50" in output


def test_the_progress_bar_carries_the_metrics_instead_of_status_lines():
    stream = io.StringIO()
    console = TrainingConsole(mode="bar", status_every=1, stream=stream)
    console.start_episodes(2)

    console.episode_end(0, {"reward": 12.345})
    console.episode_end(1, {"reward": 12.345})
    console.close()

    output = stream.getvalue()
    assert "reward=12.35" in output
    assert "ep 1/2" not in output


def test_info_once_says_a_thing_only_the_first_time():
    """Phase transitions are detected inside the episode loop, so the loop asks
    every episode and the console is what makes the announcement single."""
    stream = io.StringIO()
    console = TrainingConsole(mode="plain", stream=stream)

    for _ in range(5):
        console.info_once("rm-active", "reward model training active")

    assert stream.getvalue().count("reward model training active") == 1


def test_from_config_reads_the_logging_section():
    config = SimpleNamespace(console="plain", status_every=7)

    console = TrainingConsole.from_config(config, stream=io.StringIO())

    assert not console.uses_bar
    assert console.status_every == 7


def test_quiet_mode_writes_nothing():
    stream = io.StringIO()
    console = TrainingConsole(mode="quiet", stream=stream)

    console.section("Reward model")
    console.info("warmup for 50 episodes")
    console.warn("something looks off")
    console.start_episodes(10)
    console.episode_end(9, {"reward": 1.0})
    console.close()

    assert stream.getvalue() == ""
