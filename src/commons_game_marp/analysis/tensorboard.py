"""Launch TensorBoard against a log directory.

Cross-platform replacement for the old scripts/tensorboard.ps1, which only ran
on Windows.
"""

import argparse
import shutil
import subprocess
import sys


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="commons-game tensorboard",
        description="Launch TensorBoard on a log directory.",
    )
    parser.add_argument(
        "--logdir",
        default="logs",
        help="Directory to serve (default: logs).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=6006,
        help="Port to listen on (default: 6006).",
    )
    return parser


def run(args: argparse.Namespace) -> int:
    executable = shutil.which("tensorboard")
    if executable is None:
        print(
            "tensorboard executable not found. It ships with the `tensorboard` "
            "dependency -- try `uv sync`, or run `uv run commons-game tensorboard` "
            "so the project environment is on PATH.",
            file=sys.stderr,
        )
        return 1

    print(f"TensorBoard: http://127.0.0.1:{args.port}")
    try:
        return subprocess.call(
            [executable, "--logdir", args.logdir, "--port", str(args.port)]
        )
    except KeyboardInterrupt:
        # Ctrl-C reaches the child too; exit quietly rather than dumping a
        # traceback over the user's terminal.
        return 130


if __name__ == "__main__":
    raise SystemExit(run(build_parser().parse_args()))
