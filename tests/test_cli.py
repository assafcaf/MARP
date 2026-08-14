"""The `commons-game` dispatcher must route correctly and stay lazy.

These tests cover CLI wiring only -- routing, argv passthrough, flag parsing,
help, and import laziness. The analysis modules' plotting behavior is not
exercised here (and has no coverage elsewhere); `run` is monkeypatched out
wherever a subcommand is dispatched, so no plot is ever produced.
"""

import subprocess
import sys

import pytest

from commons_game_marp import cli


@pytest.mark.parametrize(
    "argv, expected_module",
    [
        (["plot", "run", "logs/x"], "run_metrics"),
        (["plot", "runs", "logs/a", "logs/b"], "multiple_runs"),
        (["plot", "phi"], "phi_comparisons"),
        (["compare-modes", "--narrow-view", "a", "--input-aggregation", "b"], "reward_modes"),
        (["sessions"], "sessions"),
        (["tensorboard"], "tensorboard"),
    ],
)
def test_routes_to_the_right_module(argv, expected_module, monkeypatch):
    """Each registry entry reaches its module's run() with parsed args."""
    called = {}
    # Resolved before patching: cli.importlib IS the global importlib module,
    # so patching its attribute would break this lookup too.
    real_parser = _real_parser(expected_module)

    def fake_import(name, package):
        assert name == f".analysis.{expected_module}"

        class Module:
            @staticmethod
            def build_parser():
                return real_parser

            @staticmethod
            def run(args):
                called["args"] = args
                return 0

        return Module

    monkeypatch.setattr(cli.importlib, "import_module", fake_import)
    assert cli.main(argv) == 0
    assert "args" in called, "run() was never reached"


def _real_parser(module_name):
    """The module's genuine parser, so flag parsing is really exercised."""
    import importlib

    module = importlib.import_module(f"commons_game_marp.analysis.{module_name}")
    return module.build_parser()


def test_train_passthrough_strips_only_the_subcommand(monkeypatch):
    """Hydra reads sys.argv itself, so everything after `train` must survive
    verbatim and the `train` token itself must be gone. This is the contract
    --multirun and --cfg depend on."""
    seen = {}

    class FakeTrainCli:
        @staticmethod
        def main():
            seen["argv"] = list(sys.argv)

    monkeypatch.setattr(cli.importlib, "import_module", lambda *a, **k: FakeTrainCli)
    monkeypatch.setattr(sys, "argv", ["commons-game", "train", "algorithm=ippo", "-m"])

    # _dispatch_train imports via `from . import train_cli`, so patch that too.
    import commons_game_marp

    monkeypatch.setattr(commons_game_marp, "train_cli", FakeTrainCli, raising=False)
    monkeypatch.setitem(sys.modules, "commons_game_marp.train_cli", FakeTrainCli)

    cli.main(["train", "algorithm=ippo", "-m"])

    assert seen["argv"] == ["commons-game", "algorithm=ippo", "-m"]


@pytest.mark.parametrize(
    "module_name, argv, checks",
    [
        ("run_metrics", ["logs/x", "--smooth", "5", "--no-title"],
         {"run_dir": "logs/x", "smooth": 5, "no_title": True}),
        ("multiple_runs", ["logs/a", "logs/b", "-o", "out", "--normalize"],
         {"run_dirs": ["logs/a", "logs/b"], "output_dir": "out", "normalize": True}),
        ("phi_comparisons", ["--algorithm", "mappo", "--smooth", "3"],
         {"algorithm": "mappo", "smooth": 3}),
        ("reward_modes", ["--narrow-view", "a", "b", "--input-aggregation", "c"],
         {"narrow_view": ["a", "b"], "input_aggregation": ["c"]}),
        ("sessions", ["--algorithm", "ippo", "--comparisons-only"],
         {"algorithm": "ippo", "comparisons_only": True}),
        ("tensorboard", ["--port", "7000"], {"port": 7000, "logdir": "logs"}),
    ],
)
def test_documented_flags_still_parse(module_name, argv, checks):
    """Guards the main() -> build_parser()/run() split: the flags each script
    accepted before the move must parse to the same values."""
    args = _real_parser(module_name).parse_args(argv)
    for key, expected in checks.items():
        assert getattr(args, key) == expected, key


def test_sessions_rejects_mutually_exclusive_modes():
    with pytest.raises(SystemExit) as excinfo:
        _real_parser("sessions").parse_args(["--comparisons-only", "--skip-comparisons"])
    assert excinfo.value.code == 2


def test_top_level_help_exits_zero(capsys):
    assert cli.main([]) == 0
    assert cli.main(["--help"]) == 0
    assert "commands:" in capsys.readouterr().out


def test_group_help_lists_only_its_own_commands(capsys):
    assert cli.main(["plot"]) == 0
    out = capsys.readouterr().out
    assert "run" in out and "runs" in out and "phi" in out
    assert "sessions" not in out


def test_unknown_command_exits_two(capsys):
    assert cli.main(["nonsense"]) == 2
    assert "unknown command 'nonsense'" in capsys.readouterr().err


def test_help_after_subcommand_is_not_intercepted(monkeypatch):
    """`--help` following a subcommand belongs to that subcommand's parser,
    not the dispatcher."""
    monkeypatch.setattr(cli.importlib, "import_module", lambda *a, **k: _FakeModule)
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["sessions", "--help"])
    assert excinfo.value.code == 0


class _FakeModule:
    @staticmethod
    def build_parser():
        import argparse

        return argparse.ArgumentParser(prog="fake")

    @staticmethod
    def run(args):
        return 0


def test_help_does_not_import_matplotlib():
    """The whole point of the lazy registry. Run in a subprocess so this
    session's already-imported modules don't mask a regression."""
    code = (
        "import sys; from commons_game_marp import cli; cli.main(['--help']);"
        "sys.exit(1 if 'matplotlib' in sys.modules else 0)"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True)
    assert result.returncode == 0, "matplotlib was imported during --help"
