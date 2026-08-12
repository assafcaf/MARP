# Packaging + Hydra Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make this repo a uv-installable package named `commons_game_marp`, then replace its hand-rolled JSON config system with Hydra config groups.

**Architecture:** Part A moves `src/{env,train,reward_model}` into a src-layout package and declares it in `pyproject.toml`, installed with `uv sync`. Part B splits the four monolithic JSON configs into Hydra groups (`env/`, `algorithm/`, `reward_model/`, `logging/`, `experiment/`) backed by the existing dataclasses registered in Hydra's `ConfigStore`, and replaces the hand-rolled sweep runner `scripts/run_env.py` with Hydra `--multirun`.

**Tech Stack:** Python 3.11, uv, hatchling, hydra-core 1.3, OmegaConf, PyTorch 2.5.1+cu124, PettingZoo, pytest.

**Spec:** `docs/superpowers/specs/2026-08-12-packaging-and-hydra-design.md`

## Global Constraints

- Distribution name: `commons-game-marp`. Import package: `commons_game_marp`. Repo directory keeps its current name during this work.
- `requires-python = ">=3.11"`. The environment is CPython 3.11.11.
- Never run bare `pytest` or `python`. Always `uv run pytest` / `uv run python` after Task 1, so the project venv is used.
- `torch==2.5.1+cu124` must resolve from `https://download.pytorch.org/whl/cu124` and nothing else may resolve from that index.
- `uv.lock` is committed. `.venv/` is git-ignored.
- Every task ends with `uv run pytest` green before its commit. No task may leave the tree red.
- Do not modify anything under `docs/knowledge-base/` — it is reference documentation, not part of this work.
- Existing behavior is preserved throughout. The only intentional behavior-adjacent change is the `AlgorithmConfig` restructure in Task 3, whose semantics are pinned by tests in that task.

## File Structure

**Part A**
- Move: `src/{env,train,reward_model}/` → `src/commons_game_marp/{env,train,reward_model}/`
- Move: `src/__init__.py` → `src/commons_game_marp/__init__.py`
- Create: `pyproject.toml` — build config, dependencies, dev group, pytest config
- Delete: `requirements.txt`
- Modify: `main.py`, `scripts/*.py` (6 files), `tests/*.py` (5 files), `experiment.ipynb`, `.gitignore`, `README.md`

**Part B**
- Modify: `src/commons_game_marp/train/config.py` — per-algorithm dataclasses gain `name`; add `RandomConfig`; add `register_configs()`; delete `load_config`/`save_config`
- Modify: `src/commons_game_marp/train/registry.py` — dispatch on the single selected algorithm node
- Modify: `src/commons_game_marp/train/trainer.py` — simplified `_format_reward_obs`, `OmegaConf.save` snapshot
- Create: `src/commons_game_marp/configs/` — `config.yaml` + 5 group directories
- Create: `src/commons_game_marp/cli.py` — the `@hydra.main` entry point
- Modify: `main.py` — two-line shim
- Delete: `configs/*.json` (4 files), `scripts/run_env.py`
- Modify: `tests/test_trainer_obs.py`, `tests/test_algorithm_init.py`; create `tests/test_hydra_configs.py`

---

# PART A — Packaging

### Task 1: Package skeleton, pyproject.toml, and uv install

The move and `pyproject.toml` must land together: once `src/env/` becomes `src/commons_game_marp/env/`, nothing imports until the package is installed. This is one task with one deliverable — a tree that installs and tests green.

**Files:**
- Create: `pyproject.toml`
- Move: `src/__init__.py`, `src/env/`, `src/train/`, `src/reward_model/` into `src/commons_game_marp/`
- Modify: `main.py`, `scripts/run_env.py`, `scripts/plot_run_metrics.py`, `scripts/plot_multiple_runs.py`, `scripts/plot_phi_comparisons.py`, `scripts/compare_reward_modes.py`, `scripts/process_all_sessions.py`, `tests/test_imports.py`, `tests/test_algorithm_init.py`, `tests/test_trainer_obs.py`, `tests/test_env_penalty.py`, `tests/test_metrics.py`, `experiment.ipynb`, `.gitignore`
- Delete: `requirements.txt`

**Interfaces:**
- Consumes: nothing (first task).
- Produces: importable package `commons_game_marp` with subpackages `env`, `train`, `reward_model`. All later tasks import from `commons_game_marp.*`. `uv run pytest` is the test command from here on.

- [ ] **Step 1: Move the package into src-layout**

`git mv` preserves history. Run from the repo root:

```bash
mkdir -p src/commons_game_marp
git mv src/__init__.py src/commons_game_marp/__init__.py
git mv src/env src/commons_game_marp/env
git mv src/train src/commons_game_marp/train
git mv src/reward_model src/commons_game_marp/reward_model
```

Intra-package imports are all relative (`from ..env.commons_env import ...`), so nothing inside the package needs editing.

- [ ] **Step 2: Rewrite external imports**

Only files *outside* the package reference `src.`. Rewrite them:

```bash
grep -rln '\bsrc\.\(env\|train\|reward_model\)\|from src import\|import src\b' \
  main.py scripts tests experiment.ipynb \
  | xargs sed -i 's/\bsrc\.\(env\|train\|reward_model\)\b/commons_game_marp.\1/g; s/\bfrom src import\b/from commons_game_marp import/g'
```

Then verify no stragglers remain:

```bash
grep -rn '\bsrc\.\(env\|train\|reward_model\)\|from src import' main.py scripts tests experiment.ipynb
```

Expected: no output.

- [ ] **Step 3: Delete the sys.path hack in scripts/run_env.py**

An installed package does not need it. Remove these lines from the top of `scripts/run_env.py`:

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
```

Keep `from pathlib import Path` if `Path` is still used elsewhere in that file — check with `grep -n 'Path' scripts/run_env.py` before deleting the import. (`run_env.py` is deleted in Task 6, but Part A must leave the tree working on its own.)

- [ ] **Step 4: Fix the module-discovery root in tests/test_imports.py**

This test walks the filesystem to find modules. Its root must now be the package directory, so that `relative_to(SRC.parent)` still yields importable dotted names (`commons_game_marp.train.trainer`).

Change:

```python
SRC = pathlib.Path(__file__).resolve().parent.parent / "src"
```

to:

```python
SRC = pathlib.Path(__file__).resolve().parent.parent / "src" / "commons_game_marp"
```

Also update the module docstring from `"""Every module in src/ must import cleanly."""` to `"""Every module in the commons_game_marp package must import cleanly."""`.

- [ ] **Step 5: Write pyproject.toml**

```toml
[project]
name = "commons-game-marp"
version = "0.1.0"
description = "Multi-agent reinforcement learning on the Harvest commons environment with preference-based reward modeling"
readme = "README.md"
requires-python = ">=3.11"
dependencies = [
    "torch==2.5.1+cu124",
    "numpy",
    "gymnasium==1.0.0",
    "pettingzoo==1.24.3",
    "opencv-python",
    "matplotlib",
    "tqdm",
    "tensorboard",
]

[dependency-groups]
dev = [
    "pytest",
    "mypy",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/commons_game_marp"]

# torch's CUDA build is not on PyPI. `explicit = true` means this index is used
# only for packages explicitly pinned to it below -- everything else still
# resolves from PyPI.
[[tool.uv.index]]
name = "pytorch-cu124"
url = "https://download.pytorch.org/whl/cu124"
explicit = true

[tool.uv.sources]
torch = { index = "pytorch-cu124" }

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-ra --strict-markers"
```

- [ ] **Step 6: Ignore the venv**

Add `.venv/` to `.gitignore` (the file already ignores `__pycache__/`, `*.egg-info/`, etc.).

- [ ] **Step 7: Install**

```bash
uv sync
```

Expected: resolves and installs, creating `.venv/` and `uv.lock`. The project itself installs in editable mode.

If torch resolution fails, do NOT fall back to a CPU wheel or drop the version pin — stop and report. The CUDA build is a hard requirement.

- [ ] **Step 8: Verify the whole suite passes**

```bash
uv run pytest
```

Expected: PASS, with the same number of passing tests as before the move. `test_imports.py` is parametrized over every module found on disk, so a missed import rewrite or a bad move shows up here as a collection or import error.

Additionally confirm the console-facing entry still works:

```bash
uv run python -c "from commons_game_marp.train import Trainer, TrainerConfig; print('ok')"
```

Expected: `ok`

- [ ] **Step 9: Delete requirements.txt**

```bash
git rm requirements.txt
```

- [ ] **Step 10: Commit**

```bash
git add -A
git commit -m "feat: package as commons-game-marp, installable with uv

Move src/{env,train,reward_model} into a src-layout package and declare
it in pyproject.toml. Replaces requirements.txt; torch's cu124 build now
resolves through an explicit uv index instead of --extra-index-url."
```

---

### Task 2: README installation section

**Files:**
- Modify: `README.md`

**Interfaces:**
- Consumes: the `uv sync` / `uv run` workflow from Task 1.
- Produces: nothing consumed by later tasks. (Task 6 rewrites the *usage* sections; this task covers *installation* only.)

- [ ] **Step 1: Read the current README top matter**

Read `README.md` lines 1-40. It currently opens with `# MARP Environment`, then `## Training (configurable trainer)` at line 17 showing `python main.py` and an inline `load_config('configs/train_dqn.json')` snippet. There is no installation section at all.

- [ ] **Step 2: Insert an Installation section**

Add this immediately after the intro paragraph and before `## Training (configurable trainer)`:

````markdown
## Installation

This project is a Python package managed with [uv](https://docs.astral.sh/uv/).

```bash
# Install uv (once, if you don't have it)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create the environment and install the project with all dependencies
uv sync
```

`uv sync` reads `pyproject.toml` and `uv.lock`, creates `.venv/`, and installs
`commons-game-marp` in editable mode — edits to `src/commons_game_marp/` take
effect without reinstalling.

Prefix commands with `uv run` to use the project environment:

```bash
uv run pytest              # run the test suite
uv run python main.py      # run training
```

### PyTorch and CUDA

The pinned build is `torch==2.5.1+cu124`, resolved from PyTorch's CUDA 12.4
index (configured in `pyproject.toml`). To run on CPU or a different CUDA
version, change the `torch` pin and the `[[tool.uv.index]]` URL, then re-run
`uv sync`.
````

- [ ] **Step 3: Update the inline-run snippet**

The snippet around line 36 uses the old import path. Change:

```python
config = load_config('configs/train_dqn.json')
```

and its surrounding `from src.train import ...` to use `commons_game_marp`. Leave the `load_config` call itself for now — Task 6 replaces this section wholesale when Hydra lands. The goal here is only that no `src.` import survives in the README.

Verify:

```bash
grep -n 'from src\.\|import src\b\|pip install -r' README.md
```

Expected: no output.

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "docs: document uv-based installation"
```

---

# PART B — Hydra

### Task 3: Restructure config dataclasses into Hydra groups

This is the behavior-adjacent task. `AlgorithmConfig` currently holds all three algorithm sections at once with `name` selecting the active one; Hydra's group mechanism makes `algorithm` a single selected node.

**Files:**
- Modify: `src/commons_game_marp/train/config.py`
- Modify: `src/commons_game_marp/train/registry.py`
- Modify: `src/commons_game_marp/train/trainer.py:88-110` (`_format_reward_obs`)
- Modify: `src/commons_game_marp/train/__init__.py`
- Test: `tests/test_trainer_obs.py`, `tests/test_algorithm_init.py`

**Interfaces:**
- Consumes: the `commons_game_marp` package from Task 1.
- Produces:
  - `DQNConfig`, `IPPOConfig`, `MAPPOConfig` each gain a field `name: str` defaulting to their own algorithm name, and all gain `normalize_obs: bool`.
  - New `RandomConfig(name: str = "random", normalize_obs: bool = True, device: str = "auto")`.
  - `TrainerConfig.algorithm` is typed `Any` and holds exactly one of the four above.
  - `AlgorithmConfig` (the three-section container) and `*.from_dict` are deleted.
  - `register_configs() -> None` registers every node in Hydra's `ConfigStore`; Task 4 and Task 5 call it.
  - `build_algorithm(config)` in `registry.py` now takes the selected algorithm node directly, not a container.

- [ ] **Step 1: Add hydra-core to the dependencies**

In `pyproject.toml`, add to `[project].dependencies`:

```toml
    "hydra-core>=1.3,<2.0",
```

Then:

```bash
uv sync
uv run python -c "import hydra; print(hydra.__version__)"
```

Expected: a version `1.3.x`.

- [ ] **Step 2: Write the failing test for the new config structure**

Create `tests/test_hydra_configs.py`:

```python
"""The Hydra config groups must compose into the dataclasses Trainer expects."""

import pytest
from hydra import compose, initialize_config_module
from omegaconf import OmegaConf

from commons_game_marp.train.config import TrainerConfig, register_configs

register_configs()


def _compose(*overrides):
    with initialize_config_module(version_base="1.3", config_module="commons_game_marp.configs"):
        cfg = compose(config_name="config", overrides=list(overrides))
    return OmegaConf.to_object(cfg)


@pytest.mark.parametrize("algo", ["dqn", "ippo", "mappo", "random"])
def test_algorithm_group_selects_one_node(algo):
    """The selected algorithm node carries its own name and nothing else's."""
    config = _compose(f"algorithm={algo}")

    assert isinstance(config, TrainerConfig)
    assert config.algorithm.name == algo
    # The old container held all three sections simultaneously. It must not.
    for other in ("dqn", "ippo", "mappo"):
        if other != algo:
            assert not hasattr(config.algorithm, other)


def test_every_algorithm_defines_normalize_obs():
    """`_format_reward_obs` reads this off the selected node with no fallback,
    so every algorithm -- including `random`, which used to borrow the dqn
    section -- must define it explicitly."""
    for algo in ("dqn", "ippo", "mappo", "random"):
        config = _compose(f"algorithm={algo}")
        assert isinstance(config.algorithm.normalize_obs, bool)


def test_random_normalizes_observations():
    """Regression guard: `random` has no config section of its own and used to
    silently resolve to the dqn section's scale. It now sets normalize_obs
    itself, and the value must stay True to match what the reward model was
    trained on."""
    config = _compose("algorithm=random")
    assert config.algorithm.normalize_obs is True


def test_unknown_key_is_rejected():
    """Structured configs mean a typo is a startup error, not a silent default."""
    with pytest.raises(Exception):
        _compose("algorithm.lerning_rate=0.1")
```

- [ ] **Step 3: Run it to confirm it fails**

```bash
uv run pytest tests/test_hydra_configs.py -v
```

Expected: FAIL — `ImportError: cannot import name 'register_configs'`. (The YAML tree does not exist yet either; that is Task 4. This test goes green at the end of Task 4, not this task. Leave it failing and move on — Step 9 below runs the rest of the suite.)

- [ ] **Step 4: Rewrite the algorithm dataclasses in config.py**

Give each algorithm its own `name`, add `normalize_obs` to `RandomConfig`, and delete the `AlgorithmConfig` container and every `from_dict`:

```python
from dataclasses import dataclass, field
from typing import Any, Optional

from omegaconf import MISSING


@dataclass
class EnvConfig:
    map_type: str = "small"
    num_agents: int = 1
    agent_view_range: int = 5
    ep_length: int = 600
    render: bool = False
    spawn_speed: str = "slow"
    metric: str = "Efficiency"
    penalty: bool = False


@dataclass
class DQNConfig:
    name: str = "dqn"
    learning_rate: float = 1e-3
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.1
    epsilon_decay: float = 0.995
    batch_size: int = 32
    replay_buffer_size: int = 5000
    train_after: int = 100
    train_every: int = 1
    target_update_freq: int = 200
    max_grad_norm: float = 10.0
    normalize_obs: bool = True
    device: str = "auto"


@dataclass
class IPPOConfig:
    name: str = "ippo"
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.1
    ent_coef_end: float = 0.01
    vf_coef: float = 0.5
    vf_clip: Optional[float] = 10.0
    n_steps: int = 512
    batch_size: int = 128
    update_epochs: int = 2
    hidden_size: int = 256
    max_grad_norm: float = 0.5
    normalize_obs: bool = True
    flatten_obs: bool = False
    device: str = "auto"


@dataclass
class MAPPOConfig:
    name: str = "mappo"
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    n_steps: int = 1024
    batch_size: int = 256
    update_epochs: int = 4
    hidden_size: int = 256
    max_grad_norm: float = 0.5
    normalize_obs: bool = False
    flatten_obs: bool = False
    device: str = "auto"


@dataclass
class RandomConfig:
    """The random policy has no hyperparameters, but `normalize_obs` must be
    declared: `_format_reward_obs` reads it off the selected algorithm node with
    no fallback. True preserves the previous behavior, where `random` borrowed
    the dqn section's scale."""

    name: str = "random"
    normalize_obs: bool = True
    device: str = "auto"
```

Keep `LoggingConfig` and `RewardModelConfig` exactly as they are, minus their `from_dict` static methods.

Then `TrainerConfig`:

```python
@dataclass
class TrainerConfig:
    episodes: int = 100
    seed: Optional[int] = 0
    env: EnvConfig = field(default_factory=EnvConfig)
    # `Any` rather than a union type: Hydra selects one of the four algorithm
    # nodes into this slot via the `algorithm` config group, and each node has a
    # different shape. The selected node is validated against its own schema.
    algorithm: Any = MISSING
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    reward_model: RewardModelConfig = field(default_factory=RewardModelConfig)
```

Delete `load_config` and `save_config` entirely.

- [ ] **Step 5: Add ConfigStore registration to config.py**

Append to `config.py`:

```python
def register_configs() -> None:
    """Register the schema for every config group with Hydra.

    Registering the dataclasses makes the YAML type-checked at composition
    time: an unknown or mistyped key fails at startup instead of silently
    falling back to a default. Safe to call more than once.
    """
    from hydra.core.config_store import ConfigStore

    cs = ConfigStore.instance()
    cs.store(name="base_config", node=TrainerConfig)
    cs.store(group="env", name="base_env", node=EnvConfig)
    cs.store(group="algorithm", name="base_dqn", node=DQNConfig)
    cs.store(group="algorithm", name="base_ippo", node=IPPOConfig)
    cs.store(group="algorithm", name="base_mappo", node=MAPPOConfig)
    cs.store(group="algorithm", name="base_random", node=RandomConfig)
    cs.store(group="logging", name="base_logging", node=LoggingConfig)
    cs.store(group="reward_model", name="base_reward_model", node=RewardModelConfig)
```

- [ ] **Step 6: Update registry.py**

`build_algorithm` now receives the selected node, so it no longer indexes into a container:

```python
def build_algorithm(config):
    """Build the algorithm named by the selected config node.

    `config` is one of DQNConfig / IPPOConfig / MAPPOConfig / RandomConfig --
    the single node Hydra selected into `TrainerConfig.algorithm`.
    """
    name = config.name
    if name == "dqn":
        from .algorithms.dqn import DQNAlgorithm

        return DQNAlgorithm(config)
    if name == "ippo":
        from .algorithms.ippo import IPPOAlgorithm

        return IPPOAlgorithm(config)
    if name == "mappo":
        from .algorithms.mappo import MAPPOAlgorithm

        return MAPPOAlgorithm(config)
    if name == "random":
        from .algorithms.random_policy import RandomAlgorithm

        return RandomAlgorithm(config)
    raise ValueError(f"Unknown algorithm '{name}'. Available: ['dqn', 'ippo', 'mappo', 'random']")
```

Remove the now-wrong `from .config import AlgorithmConfig` import at the top.

`RandomAlgorithm.__init__` in `src/commons_game_marp/train/algorithms/random_policy.py` is already typed `config: Any` and only passes the config to `super().__init__`; it never reaches for `.dqn`/`.ippo`/`.mappo`. It needs no change — receiving a `RandomConfig` instead of the container is transparent to it.

`Trainer.__init__` calls `build_algorithm(config.algorithm)` (`trainer.py:25`) and `_build_logger` reads `self.config.algorithm.name` (`trainer.py:59`). Both keep working unchanged, since `config.algorithm` is now the selected node and every node carries `name`.

- [ ] **Step 7: Simplify `_format_reward_obs` in trainer.py**

The fallback chain existed only because inactive sections were reachable. With one selected node, the lookup is direct. Replace the method body (`trainer.py:88-110`):

```python
    def _format_reward_obs(self, obs: dict, agent_id: str) -> np.ndarray:
        """Format an observation for the reward model.

        Normalization follows the selected algorithm's setting so the reward
        model sees inputs on the same scale as the policy does. Every algorithm
        node declares `normalize_obs`, including `random`.
        """
        img = obs[agent_id]["curr_obs"]
        algorithm_cfg = getattr(self.config, "algorithm", None)
        normalize = getattr(algorithm_cfg, "normalize_obs", False)
        if normalize:
            return (img / 255.0).astype(np.float32)
        return img.astype(np.float32)
```

- [ ] **Step 8: Update the exports in train/__init__.py**

```python
from .config import TrainerConfig, register_configs
from .trainer import Trainer

__all__ = ["Trainer", "TrainerConfig", "register_configs"]
```

- [ ] **Step 9: Rewrite tests/test_trainer_obs.py**

Two of its tests encode the old container semantics and are now structurally impossible (there is no inactive section to leak from, and no fallback to reach). The behavior they protected — the reward model seeing the same scale the policy does — is now covered here plus in `test_hydra_configs.py`. Replace the file:

```python
"""The reward model must normalize observations the same way the policy does."""

import numpy as np
import pytest

from commons_game_marp.train.config import DQNConfig, IPPOConfig, MAPPOConfig, RandomConfig
from commons_game_marp.train.trainer import Trainer


class _ConfigStub:
    def __init__(self, algorithm):
        self.algorithm = algorithm


def _format(algorithm) -> np.ndarray:
    stub = object.__new__(Trainer)
    stub.config = _ConfigStub(algorithm)
    obs = {"agent-0": {"curr_obs": np.full((3, 3, 3), 255, dtype=np.uint8)}}
    return Trainer._format_reward_obs(stub, obs, "agent-0")


@pytest.mark.parametrize("cls", [DQNConfig, IPPOConfig, MAPPOConfig, RandomConfig])
def test_normalization_follows_the_selected_algorithm(cls):
    algorithm = cls()

    algorithm.normalize_obs = True
    assert _format(algorithm).max() == pytest.approx(1.0)

    algorithm.normalize_obs = False
    assert _format(algorithm).max() == pytest.approx(255.0)


def test_random_policy_normalizes_by_default():
    """Regression guard: `random` has no hyperparameters of its own and used to
    fall through to the dqn section. Its default must stay True so reward-model
    inputs match the data the model was trained on."""
    assert _format(RandomConfig()).max() == pytest.approx(1.0)
```

- [ ] **Step 10: Fix tests/test_algorithm_init.py**

It imports `AlgorithmConfig` at line 19 (`from commons_game_marp.train.config import IPPOConfig, MAPPOConfig`) and again inside a function at line 69. Read the file, then remove any `AlgorithmConfig` import and any construction of it. Where a test built `AlgorithmConfig(name="x")` to get at a sub-config, construct the sub-config class directly (`IPPOConfig()`, `DQNConfig()`, ...).

- [ ] **Step 11: Run the tests that can pass now**

```bash
uv run pytest --ignore=tests/test_hydra_configs.py
```

Expected: PASS. `test_hydra_configs.py` still fails (no YAML tree yet) and goes green in Task 4.

- [ ] **Step 12: Commit**

```bash
git add -A
git commit -m "refactor: one algorithm config node per Hydra group

Each algorithm dataclass now carries its own name and normalize_obs, and
TrainerConfig.algorithm holds exactly one of them. Removes the container
that held all three sections at once, along with the fallback chain in
_format_reward_obs that only existed because inactive sections were
reachable. RandomConfig declares normalize_obs=True, preserving the scale
it previously borrowed from the dqn section."
```

---

### Task 4: The YAML config tree

**Files:**
- Create: `src/commons_game_marp/configs/config.yaml`
- Create: `src/commons_game_marp/configs/env/{small,medium}.yaml`
- Create: `src/commons_game_marp/configs/algorithm/{dqn,ippo,mappo,random}.yaml`
- Create: `src/commons_game_marp/configs/reward_model/{off,narrow_view,input_aggregation}.yaml`
- Create: `src/commons_game_marp/configs/logging/default.yaml`
- Create: `src/commons_game_marp/configs/experiment/{dqn,ippo,mappo,sequence_narrow_vs_input_agg}.yaml`
- Modify: `pyproject.toml` (ship the YAML in the wheel)
- Test: `tests/test_hydra_configs.py` (written in Task 3, goes green here)

**Interfaces:**
- Consumes: `register_configs()` and the dataclasses from Task 3.
- Produces: config module path `commons_game_marp.configs`, config name `config`. Group names: `env`, `algorithm`, `reward_model`, `logging`, `experiment`. Task 5's `@hydra.main` uses `config_path="configs"`, `config_name="config"`.

- [ ] **Step 1: Capture the current JSON values before deleting anything**

Task 5 deletes the JSONs; the experiment presets must reproduce them. Snapshot them first so the comparison in Step 7 has a reference:

```bash
mkdir -p /tmp/config-snapshot && cp configs/*.json /tmp/config-snapshot/
```

- [ ] **Step 2: Write the root config.yaml**

```yaml
# Composition root. Override any group from the CLI:
#   uv run commons-game-train algorithm=ippo env=medium reward_model=off
defaults:
  - base_config
  - env: medium
  - algorithm: mappo
  - reward_model: "off"
  - logging: default
  - _self_

episodes: 100
seed: 0

hydra:
  # Keep the process CWD at the repo root so `logging.log_dir: logs` and the
  # plotting scripts resolve the same paths they always have.
  job:
    chdir: false
  run:
    dir: ${logging.log_dir}/hydra/${now:%Y-%m-%d_%H-%M-%S}
  sweep:
    dir: ${logging.log_dir}/hydra/multirun/${now:%Y-%m-%d_%H-%M-%S}
    subdir: ${hydra.job.num}
```

Note `"off"` is quoted: unquoted `off` is a YAML boolean.

- [ ] **Step 3: Write the env group**

`env/small.yaml`:

```yaml
defaults:
  - base_env

map_type: small
num_agents: 1
agent_view_range: 5
ep_length: 600
render: false
spawn_speed: slow
metric: Efficiency
penalty: false
```

`env/medium.yaml` — same, with the values the shipped JSONs used:

```yaml
defaults:
  - base_env

map_type: medium
num_agents: 5
agent_view_range: 5
ep_length: 600
render: false
spawn_speed: slow
metric: Efficiency
penalty: false
```

- [ ] **Step 4: Write the algorithm group**

Each file selects its schema via `defaults` and sets `name`. Copy the hyperparameter values from the corresponding section of the snapshot JSONs (`/tmp/config-snapshot/train_*.json`), falling back to the dataclass defaults from Task 3 for anything a JSON did not set.

`algorithm/mappo.yaml` (values from `train_mappo.json` — note `n_steps: 2048`, which differs from the dataclass default of 1024):

```yaml
defaults:
  - base_mappo

name: mappo
learning_rate: 0.0003
gamma: 0.99
gae_lambda: 0.95
clip_range: 0.2
ent_coef: 0.01
vf_coef: 0.5
n_steps: 2048
batch_size: 256
update_epochs: 4
hidden_size: 256
max_grad_norm: 0.5
normalize_obs: true
flatten_obs: false
device: auto
```

`algorithm/dqn.yaml` (values from `train_dqn.json`; `max_grad_norm` was not set there, so it takes the dataclass default of 10.0):

```yaml
defaults:
  - base_dqn

name: dqn
learning_rate: 0.001
gamma: 0.99
epsilon_start: 1.0
epsilon_end: 0.1
epsilon_decay: 0.995
batch_size: 32
replay_buffer_size: 5000
train_after: 100
train_every: 1
target_update_freq: 200
max_grad_norm: 10.0
normalize_obs: true
device: auto
```

`algorithm/ippo.yaml` (values from `train_ippo.json`):

```yaml
defaults:
  - base_ippo

name: ippo
learning_rate: 0.0003
gamma: 0.99
gae_lambda: 0.95
clip_range: 0.2
ent_coef: 0.1
ent_coef_end: 0.01
vf_coef: 0.5
vf_clip: 10.0
n_steps: 512
batch_size: 128
update_epochs: 2
hidden_size: 256
max_grad_norm: 0.5
normalize_obs: true
flatten_obs: false
device: auto
```

`algorithm/random.yaml`:

```yaml
defaults:
  - base_random

name: random
normalize_obs: true
device: auto
```

- [ ] **Step 5: Write the reward_model and logging groups**

`reward_model/off.yaml`:

```yaml
defaults:
  - base_reward_model

enabled: false
```

`reward_model/narrow_view.yaml` (values from `train_mappo.json`'s `reward_model` block):

```yaml
defaults:
  - base_reward_model

enabled: true
mode: narrow_view
phi: efficiency_x_peace
lr: 0.0001
batch_pairs: 64
train_steps_per_update: 50
update_every_env_steps: 1000
warmup_episodes: 50
max_episodes_in_buffer: 5000
device: auto
save_every_episodes: 200
```

`reward_model/input_aggregation.yaml`: identical but `mode: input_aggregation`.

`logging/default.yaml`:

```yaml
defaults:
  - base_logging

log_dir: logs
run_name: null
log_interval: 1
video_enabled: true
video_every_n_episodes: 100
video_max_steps: 600
video_fps: 10
video_keep_frames: false
log_agent_episode_details: true
```

- [ ] **Step 6: Write the experiment presets**

These reproduce the deleted JSONs by name. They are used with a `+` prefix because `experiment` is not in the root defaults list.

The three JSONs are not uniform — each set its own `episodes` and its own
`logging.video_every_n_episodes`, and `train_dqn.json` used the *small* map with
**5** agents (not the 1 agent that `env/small.yaml` defaults to). Those
differences are carried explicitly by the presets below. Do not "simplify" them
away; they are what makes the presets reproduce the old runs.

`experiment/mappo.yaml`:

```yaml
# @package _global_
defaults:
  - override /env: medium
  - override /algorithm: mappo
  - override /reward_model: narrow_view

episodes: 250
seed: 0
logging:
  video_every_n_episodes: 100
```

`experiment/dqn.yaml`:

```yaml
# @package _global_
defaults:
  - override /env: small
  - override /algorithm: dqn
  - override /reward_model: narrow_view

episodes: 250
seed: 0
env:
  # train_dqn.json ran 5 agents on the small map; env/small.yaml defaults to 1.
  num_agents: 5
logging:
  video_every_n_episodes: 80
```

`experiment/ippo.yaml`:

```yaml
# @package _global_
defaults:
  - override /env: medium
  - override /algorithm: ippo
  - override /reward_model: narrow_view

episodes: 200
seed: 0
logging:
  video_every_n_episodes: 500
```

`experiment/sequence_narrow_vs_input_agg.yaml` — the old sequence file was 5 identical `narrow_view` runs then 5 identical `input_aggregation` runs, all IPPO/medium/5 agents/300 episodes with random seeds. The preset captures the shared part; the sweep axes come from the CLI:

```yaml
# @package _global_
# The reward_model and seed axes are swept from the CLI:
#   uv run commons-game-train -m +experiment=sequence_narrow_vs_input_agg \
#       reward_model=narrow_view,input_aggregation seed=0,1,2,3,4
defaults:
  - override /env: medium
  - override /algorithm: ippo
  - override /reward_model: narrow_view

episodes: 300
seed: 0
reward_model:
  phi: efficiency_x_sustainability
```

- [ ] **Step 7: Ship the YAML in the wheel**

In `pyproject.toml`, under `[tool.hatch.build.targets.wheel]`, add:

```toml
[tool.hatch.build.targets.wheel.force-include]
"src/commons_game_marp/configs" = "commons_game_marp/configs"
```

Hatchling includes package data by default for the declared package path, so verify rather than assume:

```bash
uv sync && uv run python -c "
import importlib.resources as r
print(sorted(p.name for p in r.files('commons_game_marp.configs').iterdir()))"
```

Expected: lists `config.yaml`, `algorithm`, `env`, `experiment`, `logging`, `reward_model`.

If `commons_game_marp.configs` is not importable, add an empty `src/commons_game_marp/configs/__init__.py` — `initialize_config_module` requires the config directory to be an importable module.

- [ ] **Step 8: Run the Hydra config tests**

```bash
uv run pytest tests/test_hydra_configs.py -v
```

Expected: PASS — all four parametrized cases plus the three standalone tests.

- [ ] **Step 9: Verify every group value composes**

```bash
for a in dqn ippo mappo random; do
  for e in small medium; do
    for r in "off" narrow_view input_aggregation; do
      uv run python -c "
from hydra import compose, initialize_config_module
from commons_game_marp.train.config import register_configs
register_configs()
with initialize_config_module(version_base='1.3', config_module='commons_game_marp.configs'):
    compose(config_name='config', overrides=['algorithm=$a','env=$e','reward_model=$r'])
" || echo "FAILED: $a $e $r"
    done
  done
done
```

Expected: no `FAILED` lines.

- [ ] **Step 10: Verify the experiment presets match the deleted JSONs**

For each of `dqn`, `ippo`, `mappo`, compose the preset and diff the values that the corresponding JSON set:

```bash
uv run python - <<'PY'
import json
from hydra import compose, initialize_config_module
from omegaconf import OmegaConf
from commons_game_marp.train.config import register_configs

register_configs()
for name in ("dqn", "ippo", "mappo"):
    with open(f"/tmp/config-snapshot/train_{name}.json") as f:
        old = json.load(f)
    with initialize_config_module(version_base="1.3", config_module="commons_game_marp.configs"):
        cfg = OmegaConf.to_container(compose(config_name="config", overrides=[f"+experiment={name}"]))
    for section in ("env", "logging", "reward_model"):
        for key, want in old.get(section, {}).items():
            got = cfg[section][key]
            assert got == want, f"{name}.{section}.{key}: {got!r} != {want!r}"
    for key, want in old["algorithm"][name].items():
        got = cfg["algorithm"][key]
        assert got == want, f"{name}.algorithm.{key}: {got!r} != {want!r}"
    assert cfg["episodes"] == old["episodes"], f"{name}.episodes"
    print(f"{name}: matches")
PY
```

Expected: `dqn: matches`, `ippo: matches`, `mappo: matches`. If a value differs, fix the YAML to match the JSON — the JSON is the source of truth for reproducing existing runs.

- [ ] **Step 11: Run the full suite**

```bash
uv run pytest
```

Expected: PASS.

- [ ] **Step 12: Commit**

```bash
git add -A
git commit -m "feat: Hydra config groups for env, algorithm, reward_model, logging

Splits the four monolithic JSON configs into composable groups, with
experiment/ presets reproducing each shipped JSON by name. Verified
field-by-field against the JSONs they replace."
```

---

### Task 5: Hydra entry point, and removal of the JSON config system

**Files:**
- Create: `src/commons_game_marp/cli.py`
- Modify: `main.py`, `pyproject.toml`, `src/commons_game_marp/train/trainer.py:71`
- Delete: `configs/train_dqn.json`, `configs/train_ippo.json`, `configs/train_mappo.json`, `configs/sequence_narrow_vs_input_agg.json`, and the now-empty `configs/` directory

**Interfaces:**
- Consumes: the config tree from Task 4, `register_configs()` from Task 3.
- Produces: `commons_game_marp.cli:main` — the console-script entry `commons-game-train`.

- [ ] **Step 1: Write cli.py**

```python
"""Hydra entry point for training.

Run with `commons-game-train` (installed console script) or
`python main.py` from the repo root. Any config value can be overridden on the
command line; see README for examples.
"""

import hydra
from omegaconf import DictConfig, OmegaConf

from .train.config import TrainerConfig, register_configs
from .train.trainer import Trainer

register_configs()


@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    # to_object resolves interpolations and instantiates the registered
    # dataclasses, so Trainer receives a real TrainerConfig rather than a
    # DictConfig -- no changes needed inside Trainer or the algorithms.
    config: TrainerConfig = OmegaConf.to_object(cfg)
    Trainer(config).train()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Reduce main.py to a shim**

Its current body is a `run_training` helper plus a block of commented-out template runs. Selecting a run is now a CLI override, so the whole block goes:

```python
from commons_game_marp.cli import main

if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Declare the console script**

Add to `pyproject.toml`:

```toml
[project.scripts]
commons-game-train = "commons_game_marp.cli:main"
```

Then `uv sync` to install the entry point.

- [ ] **Step 4: Replace the config snapshot in trainer.py**

At `trainer.py:71` the trainer writes `config.json` into the run directory via `save_config`, which no longer exists. Replace with an OmegaConf-written YAML:

```python
        config_path = os.path.join(logger.run_dir, "config.yaml")
        OmegaConf.save(OmegaConf.structured(self.config), config_path)
```

Add `from omegaconf import OmegaConf` to the imports at the top of `trainer.py`, and change the import line `from .config import TrainerConfig, save_config` to `from .config import TrainerConfig`.

- [ ] **Step 5: Write the round-trip test for the snapshot**

Append to `tests/test_hydra_configs.py`:

```python
def test_config_snapshot_round_trips(tmp_path):
    """The config.yaml written into each run directory must load back into an
    equivalent config -- it is the record of what a run actually used."""
    from omegaconf import OmegaConf

    config = _compose("algorithm=ippo", "env=medium")
    path = tmp_path / "config.yaml"
    OmegaConf.save(OmegaConf.structured(config), path)

    reloaded = OmegaConf.load(path)
    assert OmegaConf.to_container(reloaded) == OmegaConf.to_container(
        OmegaConf.structured(config)
    )
```

- [ ] **Step 6: Run it**

```bash
uv run pytest tests/test_hydra_configs.py -v
```

Expected: PASS.

- [ ] **Step 7: Delete the JSON config system**

```bash
git rm configs/train_dqn.json configs/train_ippo.json configs/train_mappo.json configs/sequence_narrow_vs_input_agg.json
```

Confirm nothing still references them:

```bash
grep -rn 'load_config\|save_config\|configs/train_\|sequence_narrow_vs_input_agg\.json' \
  --include='*.py' --include='*.ipynb' . | grep -v '\.git/'
```

Expected: no output. (README hits are handled in Task 6; if this command surfaces README lines, ignore them — restrict the grep to `.py`/`.ipynb` as written.)

- [ ] **Step 8: Verify the entry point end to end**

Composition only, no training run:

```bash
uv run commons-game-train --cfg job
uv run commons-game-train --cfg job algorithm=ippo env=small reward_model=off
uv run commons-game-train +experiment=mappo --cfg job
uv run commons-game-train --help
```

Expected: each prints the composed config (or help) and exits 0.

Then a real but tiny training run, to prove the whole path works including the snapshot write:

```bash
uv run commons-game-train algorithm=random env=small episodes=1 \
    env.ep_length=20 logging.video_enabled=false reward_model=off
```

Expected: exits 0, and a run directory appears under `logs/` containing `config.yaml`.

- [ ] **Step 9: Run the full suite**

```bash
uv run pytest
```

Expected: PASS.

- [ ] **Step 10: Commit**

```bash
git add -A
git commit -m "feat: Hydra entry point, remove the JSON config system

main.py becomes a shim over commons_game_marp.cli, installed as the
commons-game-train console script. Trainer's per-run config snapshot is
now OmegaConf-written YAML. Deletes load_config/save_config and the four
JSON configs they served."
```

---

### Task 6: Delete run_env.py and rewrite the README usage sections

**Files:**
- Delete: `scripts/run_env.py`
- Modify: `README.md` (the `## Training (configurable trainer)`, `## Run the environment script`, and `### Running sequences of games` sections — roughly lines 17-173)

**Interfaces:**
- Consumes: the `commons-game-train` entry point from Task 5.
- Produces: nothing. Final task.

- [ ] **Step 1: Confirm run_env.py has no importers**

```bash
grep -rn 'run_env' --include='*.py' --include='*.ipynb' . | grep -v '\.git/'
```

Expected: only `scripts/run_env.py` itself. If another script imports it, stop and report rather than deleting.

- [ ] **Step 2: Delete it**

```bash
git rm scripts/run_env.py
```

Its cross-product logic over `--algo` / `--seed` / `--map` / `--agents` and its sequence-file runner are both `--multirun`.

- [ ] **Step 3: Rewrite the README usage sections**

Read `README.md` lines 17-173 first — that span covers `## Training (configurable trainer)`, `## Run the environment script`, and `### Running sequences of games`, and is written entirely around `run_env.py` flags and JSON config paths. Replace that span with:

````markdown
## Training

Training is configured with [Hydra](https://hydra.cc/). Run with defaults:

```bash
uv run commons-game-train
```

Override any value from the command line:

```bash
uv run commons-game-train algorithm=ippo env=medium episodes=300 seed=7
uv run commons-game-train reward_model=off env.penalty=true
uv run commons-game-train algorithm=mappo algorithm.learning_rate=1e-4
```

Print the composed config without training:

```bash
uv run commons-game-train --cfg job
```

### Config groups

Configs live in `src/commons_game_marp/configs/`, split into groups:

| Group | Values | Selects |
|---|---|---|
| `env` | `small`, `medium` | Map size and agent count |
| `algorithm` | `dqn`, `ippo`, `mappo`, `random` | Learner and its hyperparameters |
| `reward_model` | `off`, `narrow_view`, `input_aggregation` | Preference-based reward modeling |
| `logging` | `default` | Log directory, video capture |

Top-level `episodes` and `seed` are set in `config.yaml` and overridable
directly.

### Experiment presets

Named combinations live in `configs/experiment/` and are selected with a `+`:

```bash
uv run commons-game-train +experiment=mappo
uv run commons-game-train +experiment=ippo episodes=500
```

### Sweeps

`--multirun` (`-m`) runs the cross-product of comma-separated values
sequentially:

```bash
# Three algorithms x three seeds = 9 runs
uv run commons-game-train -m algorithm=dqn,ippo,mappo seed=0,1,2

# Sweep a hyperparameter
uv run commons-game-train -m algorithm=ippo algorithm.learning_rate=1e-3,3e-4,1e-4

# Compare reward-model modes across five seeds (the former
# sequence_narrow_vs_input_agg.json, as one command)
uv run commons-game-train -m +experiment=sequence_narrow_vs_input_agg \
    reward_model=narrow_view,input_aggregation seed=0,1,2,3,4
```

Multirun output is grouped under `logs/hydra/multirun/<timestamp>/`.
````

- [ ] **Step 4: Sweep the rest of the README for stale references**

The later sections (plotting, reward modeling, "Running IPPO", "Running MAPPO" around lines 174-460) also contain `run_env.py` invocations and JSON config paths. Find them:

```bash
grep -n 'run_env\|configs/train_\|\.json\|load_config\|python main\.py' README.md
```

Rewrite each hit as its `commons-game-train` equivalent. Leave references to *output* JSON files (metrics written by training runs, consumed by the plotting scripts) alone — those are unrelated to config. Re-run the grep afterwards and confirm every remaining hit is an output file, not a config.

- [ ] **Step 5: Verify each documented command composes**

Every command shown in the rewritten sections must actually work. For each one, run it with `--cfg job` appended (which composes and exits without training):

```bash
uv run commons-game-train --cfg job algorithm=ippo env=medium episodes=300 seed=7
uv run commons-game-train --cfg job reward_model=off env.penalty=true
uv run commons-game-train --cfg job algorithm=mappo algorithm.learning_rate=1e-4
uv run commons-game-train --cfg job +experiment=mappo
uv run commons-game-train --cfg job +experiment=ippo episodes=500
```

Expected: all exit 0. A `--multirun` command cannot be checked with `--cfg job`; verify one sweep launches and then interrupt it:

```bash
timeout 20 uv run commons-game-train -m algorithm=random env=small episodes=1 \
    env.ep_length=10 logging.video_enabled=false reward_model=off seed=0,1
```

Expected: two runs complete, exit 0.

- [ ] **Step 6: Run the full suite**

```bash
uv run pytest
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "docs: replace run_env.py with Hydra multirun

Deletes the hand-rolled sweep runner -- its cross-product over algo/seed/
map/agents and its sequence-file mode are both --multirun. README usage
sections rewritten around commons-game-train, with every documented
command verified to compose."
```

---

## Verification Summary

The work is done when, from a clean checkout:

1. `uv sync` succeeds and installs `torch==2.5.1+cu124` from the PyTorch index.
2. `uv run pytest` passes.
3. `uv run commons-game-train --cfg job` composes for every combination of the `algorithm`, `env`, and `reward_model` group values.
4. Each `experiment/` preset composes to the same values as the JSON it replaced (Task 4, Step 10).
5. `grep -rn 'load_config\|save_config\|configs/train_\|run_env' --include='*.py' --include='*.ipynb' .` returns nothing.
6. No `src.env` / `src.train` / `src.reward_model` import remains anywhere.
