# Packaging + Hydra Config Migration — Design

**Date:** 2026-08-12
**Status:** Approved

## Goal

Two changes, executed in order:

1. Make the repo an installable Python package (`pyproject.toml`) and install it with `uv`.
2. Replace the hand-rolled JSON config system with Hydra.

Task 2 depends on task 1 (Hydra arrives as a declared dependency, and the config
module moves as part of the package rename), so they run sequentially.

## Naming

| Thing | Value |
|---|---|
| Repo directory | `CommonsGameMARP` (user renames later; work happens in place under `DanfoaTest`) |
| Distribution name | `commons-game-marp` |
| Import package | `commons_game_marp` |

Nothing in `pyproject.toml` references the repo directory name, so renaming the
folder afterwards is safe and requires no code change.

---

## Part 1 — Packaging

### Layout

`git mv` into a src-layout so tests import the installed package, not the
source tree:

```
src/
  commons_game_marp/
    __init__.py          # was src/__init__.py
    env/
    train/
    reward_model/
```

### Import rewrite

Every `src.` import becomes `commons_game_marp.`:

- `main.py`
- `scripts/` — `run_env.py`, `plot_run_metrics.py`, `plot_multiple_runs.py`,
  `plot_phi_comparisons.py`, `compare_reward_modes.py`, `process_all_sessions.py`
- `tests/` — `test_imports.py`, `test_algorithm_init.py`, `test_trainer_obs.py`,
  `test_env_penalty.py`, `test_metrics.py`
- `experiment.ipynb`

The `sys.path.insert(0, ...)` hack at the top of `scripts/run_env.py` is deleted —
an installed package does not need it. (`run_env.py` itself is deleted in Part 2;
it is still listed here because Part 1 must leave the tree working on its own.)

### pyproject.toml

- Build backend: `hatchling`, with `packages = ["src/commons_game_marp"]`.
- `requires-python = ">=3.11"` (environment is CPython 3.11.11).
- Runtime dependencies, from `requirements.txt`: `torch==2.5.1+cu124`, `numpy`,
  `gymnasium==1.0.0`, `pettingzoo==1.24.3`, `opencv-python`, `matplotlib`,
  `tqdm`, `tensorboard`.
- **Torch CUDA index.** The `--extra-index-url` line in `requirements.txt` has no
  `pyproject.toml` equivalent. It is expressed as a uv explicit index plus a
  source pin, so only torch is resolved from the PyTorch index:

  ```toml
  [[tool.uv.index]]
  name = "pytorch-cu124"
  url = "https://download.pytorch.org/whl/cu124"
  explicit = true

  [tool.uv.sources]
  torch = { index = "pytorch-cu124" }
  ```

- Dev tooling moves out of runtime deps into a dependency group:

  ```toml
  [dependency-groups]
  dev = ["pytest", "mypy"]
  ```

- Pytest configuration (`testpaths = ["tests"]`, strict markers) moves into
  `[tool.pytest.ini_options]`.

### Install

`uv sync` creates `.venv` and writes `uv.lock`. `uv.lock` is committed;
`.venv/` is added to `.gitignore`. `requirements.txt` is deleted — superseded by
`pyproject.toml` + lockfile.

### README

The installation section is rewritten around uv: install uv, `uv sync`,
`uv run pytest`, `uv run python main.py`. Any surviving pip/requirements
instructions elsewhere in the README are updated to match.

### Done when

`uv sync` succeeds and `uv run pytest` passes with the same result as before the
move. The existing suite is the proof that the import rewrite is complete.

---

## Part 2 — Hydra

### Config tree

Configs live **inside the package**, at `src/commons_game_marp/configs/`, not at
the repo root. Reason: the console script below is an installed entry point, so
its `config_path` must resolve relative to a module that ships in the wheel — a
repo-root `configs/` is not importable once installed. `uv sync` installs the
project editable, so these files are still edited directly in the repo; only the
path is longer. Ad-hoc config trees outside the package remain usable via
Hydra's `--config-dir`.

```
src/commons_game_marp/configs/
  config.yaml                    # defaults list + top-level episodes/seed
  env/small.yaml
  env/medium.yaml
  algorithm/dqn.yaml
  algorithm/ippo.yaml
  algorithm/mappo.yaml
  algorithm/random.yaml
  reward_model/off.yaml
  reward_model/narrow_view.yaml
  reward_model/input_aggregation.yaml
  logging/default.yaml
  experiment/dqn.yaml
  experiment/ippo.yaml
  experiment/mappo.yaml
  experiment/sequence_narrow_vs_input_agg.yaml
```

The `experiment/` presets reproduce today's shipped JSON configs, so existing
runs remain reproducible by name.

### Schema

The dataclasses in `src/commons_game_marp/train/config.py` are registered in
Hydra's `ConfigStore` and act as the typed schema for the YAML groups. YAML is
validated at composition time, so an unknown or mistyped key fails at startup
instead of silently falling back to a default.

`Trainer` continues to receive a real `TrainerConfig`:

```python
cs = ConfigStore.instance()
cs.store(name="base_config", node=TrainerConfig)

@hydra.main(version_base="1.3", config_path="configs", config_name="config")  # in cli.py
def main(cfg: DictConfig) -> None:
    config: TrainerConfig = OmegaConf.to_object(cfg)
    Trainer(config).train()
```

No change to `Trainer` internals or to any algorithm class.

### AlgorithmConfig simplification

`AlgorithmConfig` currently holds all three algorithm sub-configs simultaneously
(`dqn`, `ippo`, `mappo`) with a `name: str` selecting the active one. Hydra's
config groups make that redundant: `algorithm` becomes the single selected node,
carrying its own `name` field for downstream identification.

Consequence: the `.name`-keyed active-section lookup in
`Trainer._format_reward_obs` no longer has inactive sections to fall through to.
The behavior it guards is preserved; the mechanism is simplified. Its test
(`test_shipped_configs_populate_algorithm_name`) is rewritten to assert against
the Hydra-composed config for each algorithm group.

### Deletions

- The 4 JSON files in `configs/`.
- `load_config()` and `save_config()` from `config.py`, and their re-exports in
  `train/__init__.py`.
- `scripts/run_env.py`. Its cross-product logic over `--algo`/`--seed`/`--map`/
  `--agents`, and the sequence-JSON runner, are exactly Hydra `--multirun`:

  ```
  # was: python scripts/run_env.py --algo ippo mappo --seed 0 1 2
  uv run commons-game-train -m algorithm=ippo,mappo seed=0,1,2

  # was: sequence_narrow_vs_input_agg.json (5x narrow_view + 5x input_aggregation)
  uv run commons-game-train -m +experiment=sequence_narrow_vs_input_agg \
      reward_model=narrow_view,input_aggregation seed=0,1,2,3,4
  ```

  The README documents these equivalents.

### Trainer config snapshot

`Trainer.__init__` writes a config snapshot via `save_config` at
`trainer.py:71`. That call becomes `OmegaConf.save`, writing YAML into the run
directory. The snapshot must round-trip: writing it and re-composing it yields
an equivalent config.

### Output directories

- `hydra.job.chdir=False` so `logging.log_dir: logs` stays relative to the repo
  root and existing log-reading scripts keep working.
- `hydra.run.dir` / `hydra.sweep.dir` point into the existing logs tree rather
  than creating a stray `outputs/` directory at the repo root.

### Entry point

The Hydra entry lives in the package at `src/commons_game_marp/cli.py`, holding
the `@hydra.main`-decorated `main()` shown above. `pyproject.toml` declares:

```toml
[project.scripts]
commons-game-train = "commons_game_marp.cli:main"
```

Hatchling is configured to include the `configs/**/*.yaml` tree in the wheel.

The repo-root `main.py` is reduced to a two-line shim that calls the same
`main()`, so `uv run python main.py` and `uv run commons-game-train` are
equivalent. Its current commented-out "template runs" block is deleted —
selecting a run is now a CLI override, not an edit to a source file.

`hydra-core` is added to the runtime dependencies.

### Done when

- `uv run pytest` passes.
- `uv run commons-game-train --cfg job` composes cleanly for every algorithm,
  env, and reward_model group value.
- Each `experiment/` preset composes to a config matching what the corresponding
  deleted JSON produced (verified by comparing composed values against the JSON
  contents captured before deletion).

---

## Execution

Two dedicated agents, run sequentially — Part 2 edits files Part 1 moves, and
depends on `hydra-core` being a declared dependency.

Each agent must leave the tree green (`uv run pytest`) and commit its own work.
