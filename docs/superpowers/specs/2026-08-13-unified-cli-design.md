# Unified `commons-game` CLI — Design

**Date:** 2026-08-13
**Status:** Approved

## Goal

Replace the current split surface — one installed command (`commons-game-train`)
plus five uninstalled scripts invoked as `uv run python scripts/*.py` — with a
single installed `commons-game` command carrying every operation as a
subcommand.

## Current state

| What | Where | How it runs today |
|---|---|---|
| Training | `src/commons_game_marp/cli.py` | `commons-game-train` (Hydra) |
| Plot one run | `scripts/plot_run_metrics.py` (848 lines) | `uv run python scripts/...` |
| Plot averaged runs | `scripts/plot_multiple_runs.py` (873) | `uv run python scripts/...` |
| Phi comparisons | `scripts/plot_phi_comparisons.py` (679) | `uv run python scripts/...` |
| Compare reward modes | `scripts/compare_reward_modes.py` (323) | `uv run python scripts/...` |
| Process all sessions | `scripts/process_all_sessions.py` (2012) | `uv run python scripts/...` |
| TensorBoard | `scripts/tensorboard.ps1` (17) | PowerShell, Windows only |

The scripts are not installed, so they are unavailable to anyone who installs
the wheel. They also import each other through `sys.path` manipulation:
`plot_phi_comparisons` imports from `process_all_sessions`, which imports from
`plot_multiple_runs`. Both hacks disappear when they become a package.

## Command surface

```
commons-game train algorithm=ippo seed=3
commons-game train -m algorithm=ippo,mappo seed=0,1,2
commons-game train --cfg job

commons-game plot run   logs/<run>/
commons-game plot runs  --runs logs/a logs/b --label ippo
commons-game plot phi   --algo ippo

commons-game compare-modes --narrow-view <dirs> --input-aggregation <dirs>
commons-game sessions --all
commons-game tensorboard --logdir logs --port 6006
```

`commons-game-train` and `uv run python main.py` are kept as working aliases for
the training path. Nothing that works today stops working.

## Architecture

### File structure

```
src/commons_game_marp/
  cli.py                    # dispatcher + subcommand registry (no heavy imports)
  train_cli.py              # the @hydra.main entry, moved out of cli.py
  analysis/
    __init__.py
    run_metrics.py          # was scripts/plot_run_metrics.py
    multiple_runs.py        # was scripts/plot_multiple_runs.py
    phi_comparisons.py      # was scripts/plot_phi_comparisons.py
    reward_modes.py         # was scripts/compare_reward_modes.py
    sessions.py             # was scripts/process_all_sessions.py
    tensorboard.py          # new, replaces scripts/tensorboard.ps1
```

`scripts/` is deleted.

### Module contract

Every analysis module exposes exactly two public functions:

```python
def build_parser() -> argparse.ArgumentParser: ...
def run(args: argparse.Namespace) -> int: ...
```

Each module's existing `main()` splits along the seam already present in it:
everything constructing the parser moves to `build_parser`, everything after
`parse_args()` moves to `run`. The plotting internals — roughly 4,700 lines —
are not restructured, renamed, or otherwise touched.

Modules keep a `if __name__ == "__main__": sys.exit(run(build_parser().parse_args()))`
block so they remain directly executable during debugging.

### Dispatcher

`cli.py` holds a static registry mapping subcommand path to module and one-line
description:

```python
COMMANDS = {
    ("plot", "run"):    ("run_metrics",      "Plot reward and social metrics from one run folder"),
    ("plot", "runs"):   ("multiple_runs",    "Plot averaged metrics with std dev across runs"),
    ("plot", "phi"):    ("phi_comparisons",  "Plot phi comparisons and NV vs IA galleries"),
    ("compare-modes",): ("reward_modes",     "Compare narrow-view against input-aggregation runs"),
    ("sessions",):      ("sessions",         "Process sessions and generate cross-session plots"),
    ("tensorboard",):   ("tensorboard",      "Launch TensorBoard on a log directory"),
}
```

The registry holds *strings*, not imported modules. This is what makes lazy
import work: `commons-game --help` and `commons-game train ...` render or run
without importing matplotlib or numpy at all.

Dispatch:

1. No arguments at all, or `argv[1]` is exactly `-h`/`--help` → print the
   top-level help built from the registry. Exit 0. This tests `argv[1]` only,
   so `--help` appearing *after* a subcommand is never intercepted here and
   always reaches that subcommand's own parser.
2. `argv[1] == "train"` → delete that one token from `sys.argv` and call
   `train_cli.main()`. Hydra then reads exactly the argv it reads today, so
   `--multirun`, `--cfg`, `--help`, and run-directory handling are unchanged.
   Checked before any other matching, so `commons-game train --help` prints
   Hydra's help, not the dispatcher's.
3. `argv[1]` is a group prefix of some registry key with nothing after it, or
   followed only by `-h`/`--help` (today: `commons-game plot`) → print the
   group's help, listing just that group's subcommands and their descriptions.
   Exit 0.
4. Otherwise match the longest registry key against the leading argv tokens,
   import that module, build its parser, parse the *remaining* argv, and return
   `run(args)`. Longest-first matching matters: `("plot", "run")` must be tried
   before any shorter key that shares its first token.
5. No match → print the top-level help to stderr with an "unknown command"
   line naming the offending token. Exit 2.

`main()` returns the subcommand's exit code; the console-script wrapper passes
it to `sys.exit`.

Because each module's own parser handles its flags, `commons-game plot run --help`
delegates to that parser and prints exactly the help text the script prints
today. Every existing flag keeps working with identical semantics.

### Why not argparse subparsers

Registering all six as real subparsers would require importing all six modules
up front to obtain their parsers, defeating lazy import. The hand-rolled
dispatcher is a modest amount of code and keeps `train` fast.

### Cross-module imports

`sys.path.insert(0, script_dir)` in `phi_comparisons.py` and `sessions.py` is
deleted, and the imports it enabled become relative:

- `from process_all_sessions import (...)` → `from .sessions import (...)`
- `from plot_multiple_runs import plot_multiple_runs, PUBLICATION_COLORS, _format_label` → `from .multiple_runs import (...)`

### TensorBoard subcommand

`tensorboard.py` replaces the PowerShell script with a cross-platform
equivalent: resolve the `tensorboard` executable, launch it against `--logdir`
(default `logs`) on `--port` (default 6006), print the URL, and forward
Ctrl-C to the child. If the executable is missing, print an actionable error and
return exit code 1 rather than raising. `scripts/tensorboard.ps1` is deleted.

### Entry points

```toml
[project.scripts]
commons-game = "commons_game_marp.cli:main"
commons-game-train = "commons_game_marp.train_cli:main"
```

`main.py` at the repo root changes its import to `train_cli` and otherwise stays
a two-line shim.

## Error handling

- Every `run()` returns an int exit code; the dispatcher propagates it.
- Unknown subcommand: top-level help to stderr, exit 2 (argparse's convention).
- Missing `tensorboard` executable: actionable message, exit 1.
- Argument errors inside a subcommand are argparse's existing behavior,
  unchanged — the module's own parser raises them.

## Testing

The five analysis modules have **zero test coverage today**, and this work does
not change that. Adding meaningful coverage for 4,700 lines of plotting logic is
a separate project. Tests here cover the CLI wiring only:

1. **Routing** — each registry entry dispatches to the expected module's `run`,
   verified with `run` monkeypatched. No plotting executes.
2. **Train passthrough** — `commons-game train algorithm=ippo -m` leaves
   `sys.argv` as `[prog, "algorithm=ippo", "-m"]` when `train_cli.main` is
   called. This is the contract Hydra depends on and the highest-risk piece.
3. **Flag fidelity** — for each module, `build_parser().parse_args([...])` on
   the flags the README documents produces the expected Namespace. This is what
   catches a botched `main()` split.
4. **Help** — `commons-game --help`, `commons-game plot --help`, and each
   subcommand's `--help` exit 0.
5. **Unknown command** — exits 2 and names the bad token.
6. **Laziness** — after `commons-game --help`, `matplotlib` is absent from
   `sys.modules`. Guards the property that makes the dispatcher worth its
   complexity.

The existing 66-test suite must stay green throughout.

## Out of scope

- Restructuring or splitting the large analysis modules.
- Testing plotting behavior.
- The `base_*` schema entries appearing in `commons-game train --help`'s config
  group listing. That is a ConfigStore concern, not a CLI one, and is left for a
  separate change.
