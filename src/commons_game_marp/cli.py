"""Unified `commons-game` command.

Dispatches to training (Hydra) or to one of the analysis subcommands.

The registry below holds module *names*, not imported modules. Analysis modules
import matplotlib and numpy at import time, so importing them eagerly would add
that cost to every invocation -- including `commons-game train` and `--help`.
A module is imported only once its subcommand is actually selected.
"""

import argparse
import importlib
import sys
from typing import Dict, List, Optional, Sequence, Tuple

# subcommand path -> (module name in .analysis, one-line description)
COMMANDS: Dict[Tuple[str, ...], Tuple[str, str]] = {
    ("plot", "run"): ("run_metrics", "Plot reward and social metrics from one run folder"),
    ("plot", "runs"): ("multiple_runs", "Plot averaged metrics with std dev across runs"),
    ("plot", "phi"): ("phi_comparisons", "Plot phi comparisons and NV vs IA galleries"),
    ("compare-modes",): ("reward_modes", "Compare narrow-view against input-aggregation runs"),
    ("sessions",): ("sessions", "Process sessions and generate cross-session plots"),
    ("tensorboard",): ("tensorboard", "Launch TensorBoard on a log directory"),
}

TRAIN_DESCRIPTION = "Train agents (Hydra: use group=value overrides, -m to sweep)"

_HELP_FLAGS = ("-h", "--help")


def _groups() -> Dict[str, List[Tuple[str, str]]]:
    """Subcommand groups (e.g. "plot") -> [(leaf name, description), ...]."""
    grouped: Dict[str, List[Tuple[str, str]]] = {}
    for path, (_, description) in COMMANDS.items():
        if len(path) > 1:
            grouped.setdefault(path[0], []).append((path[1], description))
    return grouped


def _format_top_level_help() -> str:
    lines = [
        "usage: commons-game <command> [options]",
        "",
        "Multi-agent RL on the Harvest commons environment.",
        "",
        "commands:",
        f"  {'train':<16}{TRAIN_DESCRIPTION}",
    ]
    grouped = _groups()
    for group, leaves in grouped.items():
        lines.append(f"  {group:<16}{group.capitalize()} commands: " + ", ".join(n for n, _ in leaves))
    for path, (_, description) in COMMANDS.items():
        if len(path) == 1:
            lines.append(f"  {path[0]:<16}{description}")
    lines += [
        "",
        "Run 'commons-game <command> --help' for command-specific options.",
    ]
    return "\n".join(lines)


def _format_group_help(group: str) -> str:
    lines = [
        f"usage: commons-game {group} <command> [options]",
        "",
        "commands:",
    ]
    for name, description in _groups()[group]:
        lines.append(f"  {name:<10}{description}")
    lines += [
        "",
        f"Run 'commons-game {group} <command> --help' for command-specific options.",
    ]
    return "\n".join(lines)


def _match(argv: Sequence[str]) -> Optional[Tuple[Tuple[str, ...], List[str]]]:
    """Match the longest registry key against the leading argv tokens.

    Longest-first matters: ("plot", "run") must be tried before any shorter key
    sharing its first token.
    """
    for length in (2, 1):
        candidate = tuple(argv[:length])
        if candidate in COMMANDS:
            return candidate, list(argv[length:])
    return None


def _dispatch_train(argv: List[str]) -> int:
    """Hand everything after `train` to the Hydra entry point.

    Hydra reads sys.argv itself, so the `train` token is removed to leave argv
    exactly as it would be for a bare `commons-game-train` invocation. This is
    what keeps --multirun, --cfg, --help and run-dir handling working.
    """
    from . import train_cli

    sys.argv = [sys.argv[0]] + argv
    train_cli.main()
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)

    if not args or args[0] in _HELP_FLAGS:
        print(_format_top_level_help())
        return 0

    # Checked before anything else so `commons-game train --help` reaches
    # Hydra's help rather than being intercepted here.
    if args[0] == "train":
        return _dispatch_train(args[1:])

    grouped = _groups()
    if args[0] in grouped and (len(args) == 1 or args[1] in _HELP_FLAGS):
        print(_format_group_help(args[0]))
        return 0

    matched = _match(args)
    if matched is None:
        print(f"commons-game: unknown command '{args[0]}'\n", file=sys.stderr)
        print(_format_top_level_help(), file=sys.stderr)
        return 2

    path, rest = matched
    module_name, _ = COMMANDS[path]
    module = importlib.import_module(f".analysis.{module_name}", __package__)
    parser: argparse.ArgumentParser = module.build_parser()
    return module.run(parser.parse_args(rest))


if __name__ == "__main__":
    raise SystemExit(main())
