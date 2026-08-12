# MARP Refactor — Plan 1: Foundation

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up a fresh repository seeded from the `almogze/marp` fork, remove dead and broken code, and fix the known correctness defects — producing a clean codebase that imports, tests, and behaves correctly before any semantic work begins.

**Architecture:** Three sequential phases on one codebase, each its own branch merged `--no-ff`. Phase 0 seeds the repo. Phase 1 is hygiene with no behavior change. Phase 2 fixes defects, each with a regression test. Nothing here changes environment semantics or algorithm structure — that is Plan 2 and Plan 3.

**Tech Stack:** Python 3.11.11 (conda env `danfoa`), PyTorch 2.5.1+cu124, Gymnasium, PettingZoo, pytest.

**Design spec:** [`docs/superpowers/specs/2026-08-11-marp-refactor-design.md`](../specs/2026-08-11-marp-refactor-design.md) — read §1 (Provenance) before starting. This plan covers spec phases 0, 1, and 2.

## Global Constraints

- **`/home/assaf_caftory/CommonsGame/DanfoaTestSOT` is READ-ONLY.** Never write there. It is the fallback copy of the pre-refactor repo.
- **No training runs.** Verification is tests only. Anything needing GPU time belongs in `docs/experiment-schedule.md` (Plan 4).
- Python env is `danfoa`: `/home/assaf_caftory/miniconda3/envs/danfoa/bin/python`. It is the default `python3` on PATH.
- All tests run from the repo root: `cd /home/assaf_caftory/CommonsGame/DanfoaTest && python3 -m pytest`.
- Imports use the `src.` prefix (e.g. `from src.train.config import EnvConfig`), matching `tests/test_metrics.py`.
- Upstream source of truth for the seed: `https://github.com/almogze/marp`, master, commit `7cff81c`.
- Do not "fix" the PRM reward routing or add terminal-observation handling — both were audited and are already correct. See spec §4.4.
- Commit messages end with a `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>` trailer.

---

## File Structure

**Phase 0 — created:**
- `.git/` — fresh history
- `src/`, `configs/`, `scripts/`, `tests/`, `main.py`, `requirements.txt` — from fork `7cff81c`
- `docs/knowledge-base/`, `docs/superpowers/` — carried over, already present and untracked
- `.gitignore` — new, with `__pycache__/` and `*.pyc` rules

**Phase 1 — deleted:**
- `src/env/utils/train_utils.py` — `ImportError` on import, zero call sites
- `src/train/algorithms/base.py:37-38` — dead `train()` with a signature mismatch
- `src/train/trainer.py` — the unreachable `uses_external_loop()` false branch

**Phase 1 — modified:**
- `requirements.txt` — drop `stable-baselines3`, `tqdm`; pin `gymnasium`, `pettingzoo`; add `numpy`, `pytest`
- `main.py` — remove dead `sys.path.append`, fix the nonexistent config reference
- `src/__init__.py`, `src/env/utils/__init__.py` — new package markers
- `src/train/video_utils.py:4` — import style

**Phase 2 — modified:**
- `src/env/commons_env.py:73-83` — penalty must not corrupt social metrics
- `src/train/algorithms/ippo.py:13-19,247-248,250-253` — layer-wise orthogonal gains, Adam `eps`
- `src/train/algorithms/mappo.py:191-201` — add orthogonal init, Adam `eps`
- `src/train/algorithms/dqn.py:112-114` — gradient clipping
- `src/train/trainer.py:89-96` — resolve `normalize_obs` from the active algorithm

**Phase 2 — created:**
- `tests/test_env_penalty.py`, `tests/test_algorithm_init.py`, `tests/test_trainer_obs.py`

---

# PHASE 0 — Seed the repository

Branch: none (commits go directly to `main`).

### Task 1: Initialize the fresh repository

**Files:**
- Delete: `/home/assaf_caftory/CommonsGame/DanfoaTest/{.git,src,results,old_results,README.md,requirements.txt,.vscode,.mypy_cache}`
- Create: everything from fork `7cff81c`, plus `.gitignore`
- Preserve: `docs/` (currently untracked — contains the knowledge base, the paper PDF, this plan, and the spec)

**Interfaces:**
- Produces: a git repo at `/home/assaf_caftory/CommonsGame/DanfoaTest` with exactly one commit, containing the fork's `src/`, `configs/`, `scripts/`, `tests/`, `main.py`, `requirements.txt`, and the preserved `docs/`.

> **DESTRUCTIVE TASK.** Steps 1–2 verify the safety net before anything is deleted. Do not skip them.
> Do not proceed if either check fails — stop and report instead.

- [ ] **Step 1: Verify the SOT backup is intact**

```bash
ls /home/assaf_caftory/CommonsGame/DanfoaTestSOT/src/env/commons_env.py \
   /home/assaf_caftory/CommonsGame/DanfoaTestSOT/results \
   /home/assaf_caftory/CommonsGame/DanfoaTestSOT/.git
cd /home/assaf_caftory/CommonsGame/DanfoaTestSOT && git log --oneline -1
```

Expected: all paths exist; log prints `c1e37bc added README`.

- [ ] **Step 2: Verify the old history also survives on the remote**

```bash
cd /home/assaf_caftory/CommonsGame/DanfoaTest && git log --oneline -1 && \
  timeout 30 git ls-remote origin main
```

Expected: local HEAD is `c1e37bc`, and `ls-remote` prints a matching SHA for `refs/heads/main`.
This confirms the history being discarded exists in two other places.

- [ ] **Step 3: Clone the fork into a staging directory**

```bash
rm -rf /tmp/marp-seed && git clone https://github.com/almogze/marp /tmp/marp-seed
cd /tmp/marp-seed && git checkout 7cff81c && git rev-parse HEAD
```

Expected: `7cff81cd02e0cb080dc8bff93df02f9dc0e52324`.

- [ ] **Step 4: Remove the old working tree, preserving `docs/`**

```bash
cd /home/assaf_caftory/CommonsGame/DanfoaTest
rm -rf .git src results old_results README.md requirements.txt .vscode .mypy_cache .gitignore
ls -A
```

Expected: only `docs` remains.

- [ ] **Step 5: Copy the fork's contents in**

```bash
cd /tmp/marp-seed && rm -rf .git
cp -r /tmp/marp-seed/. /home/assaf_caftory/CommonsGame/DanfoaTest/
cd /home/assaf_caftory/CommonsGame/DanfoaTest && ls -A
```

Expected: `configs docs main.py requirements.txt scripts src tests` (plus any dotfiles the fork ships).

- [ ] **Step 6: Write `.gitignore`**

```
__pycache__/
*.pyc
*.pyo
.mypy_cache/
.pytest_cache/
logs/
*.egg-info/
.vscode/
```

- [ ] **Step 7: Verify the code imports before committing**

```bash
cd /home/assaf_caftory/CommonsGame/DanfoaTest && python3 -c "
from src.train.config import TrainerConfig
from src.train.trainer import Trainer
from src.train.registry import build_algorithm
from src.reward_model.oracle import compute_phi
print('imports OK')
"
```

Expected: `imports OK`.

- [ ] **Step 8: Initialize git and make commit zero**

```bash
cd /home/assaf_caftory/CommonsGame/DanfoaTest
git init -b main
git add -A
git commit -m "$(cat <<'EOF'
Seed repository from almogze/marp @7cff81c

Starting point for the MARP refactor. Vendored from the fork at
https://github.com/almogze/marp, master, commit 7cff81c.

That fork produced the paper's main results (Figures 5-11); see
docs/superpowers/specs/2026-08-11-marp-refactor-design.md section 1 for the
provenance evidence. The pre-refactor DQN-era codebase is preserved read-only
at ../DanfoaTestSOT and on the danfoatest origin remote.

Includes docs/knowledge-base/ (subsystem audits) and docs/superpowers/
(refactor spec and plans), carried over from the previous working tree.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
git log --oneline && git status --short
```

Expected: one commit; `git status` clean.

---

# PHASE 1 — Hygiene

Branch: `chore/hygiene`. No behavior changes.

### Task 2: Install pytest and confirm the test harness runs

**Files:**
- Modify: `requirements.txt`

**Interfaces:**
- Produces: a working `python3 -m pytest` invocation from the repo root. Every later task depends on this.

- [ ] **Step 1: Create the branch**

```bash
cd /home/assaf_caftory/CommonsGame/DanfoaTest && git checkout -b chore/hygiene
```

- [ ] **Step 2: Confirm pytest is currently missing**

```bash
python3 -c "import pytest" 2>&1 | tail -1
```

Expected: `ModuleNotFoundError: No module named 'pytest'`. This is why the fork's one test file has never been run here.

- [ ] **Step 3: Install pytest**

```bash
python3 -m pip install pytest
python3 -c "import pytest; print(pytest.__version__)"
```

Expected: a version number.

- [ ] **Step 4: Run the fork's existing test suite**

```bash
cd /home/assaf_caftory/CommonsGame/DanfoaTest && python3 -m pytest tests/ -v
```

Expected: `tests/test_metrics.py` collects and passes. If any test fails, **stop and report** — that is a pre-existing defect worth knowing about before we change anything, not something to fix silently here.

- [ ] **Step 5: Commit**

Add `pytest` to `requirements.txt` (full dependency cleanup is Task 4; this line is what makes the suite runnable).

```bash
git add requirements.txt
git commit -m "chore: add pytest so the test suite is runnable

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 3: Delete broken and unreachable code

**Files:**
- Delete: `src/env/utils/train_utils.py`
- Modify: `src/train/algorithms/base.py:37-38`, `src/train/trainer.py` (the `uses_external_loop()` false branch), `main.py`
- Test: `tests/test_imports.py` (create)

Three separate dead things, one task because they share a single verification: every module still imports.

1. `src/env/utils/train_utils.py` does `from DDQN import DDQNAgent, DeepQNet` (line 17) — no `DDQN` module exists — and `from env.utils import utility_funcs` (line 15), a stale absolute import. It is the only module in the package that fails to import, and nothing references it.
2. `Algorithm.train()` (`base.py:37-38`) declares `(self, env, logger, config)` but `trainer.py` calls it with four arguments. It is unreachable because every algorithm inherits `uses_external_loop() -> True`, so the branch guarding the call never executes. Delete the method and the branch rather than fixing a signature nothing calls.
3. `main.py` appends to `sys.path` *after* its top-level import (dead), and references `configs/train_ppo.json`, which does not exist on master.

**Interfaces:**
- Produces: `tests/test_imports.py::test_all_modules_import`, a guard later tasks rely on.

- [ ] **Step 1: Write the failing test**

Create `tests/test_imports.py`:

```python
"""Every module in src/ must import cleanly."""

import importlib
import pathlib

import pytest

SRC = pathlib.Path(__file__).resolve().parent.parent / "src"


def _iter_module_names():
    """Discover modules from the filesystem.

    Deliberately does not use pkgutil.walk_packages, which would require
    importing src first -- that is exactly what this test is checking.
    """
    for path in sorted(SRC.rglob("*.py")):
        if path.name == "__init__.py":
            continue
        relative = path.relative_to(SRC.parent).with_suffix("")
        yield ".".join(relative.parts)


@pytest.mark.parametrize("module_name", _iter_module_names())
def test_module_imports(module_name):
    importlib.import_module(module_name)
```

- [ ] **Step 2: Run it to confirm it fails**

```bash
python3 -m pytest tests/test_imports.py -v
```

Expected: FAIL on `src.env.utils.train_utils` with `ModuleNotFoundError: No module named 'DDQN'`.
Every other module should PASS — that contrast is what identifies the one broken file.

- [ ] **Step 3: Delete the broken module**

```bash
git rm src/env/utils/train_utils.py
```

- [ ] **Step 4: Delete the dead `train()` method**

In `src/train/algorithms/base.py`, remove:

```python
    def train(self, env, logger, config) -> None:
        raise NotImplementedError
```

- [ ] **Step 5: Delete the unreachable branch in the trainer**

In `src/train/trainer.py`, find the `if not self.algorithm.uses_external_loop():` block near the top of `train()` and remove the conditional and its body, keeping the external-loop path as the only path. Verify no other reference to `uses_external_loop` remains in `trainer.py`:

```bash
grep -n "uses_external_loop" src/train/trainer.py
```

Expected: no output.

Leave `uses_external_loop()` itself on `Algorithm` and its overrides — that is public API for a future algorithm that genuinely needs its own loop.

- [ ] **Step 6: Fix `main.py`**

```python
from src.train import Trainer, load_config


def run_training(config_path: str) -> None:
    config = load_config(config_path)
    trainer = Trainer(config)
    trainer.train()


if __name__ == "__main__":
    # Template runs (uncomment one). Toggle reward modeling via the
    # reward_model section in the config.
    # run_training("configs/train_dqn.json")
    # run_training("configs/train_ippo.json")
    run_training("configs/train_mappo.json")
```

- [ ] **Step 7: Run the test to verify it passes**

```bash
python3 -m pytest tests/test_imports.py -v
```

Expected: all parametrized cases PASS.

- [ ] **Step 8: Commit**

```bash
git add -A
git commit -m "chore: remove broken train_utils and unreachable Algorithm.train

- src/env/utils/train_utils.py imported a nonexistent DDQN module and had
  zero call sites; it was the only module in src/ that failed to import.
- Algorithm.train() declared 3 params but trainer.py called it with 4. The
  branch guarding that call is unreachable: every algorithm inherits
  uses_external_loop() -> True. Removed the method and the dead branch.
- main.py: dropped the post-import sys.path append and the reference to
  configs/train_ppo.json, which does not exist on master.

Adds tests/test_imports.py to guard against regressions.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 4: Dependency cleanup

**Files:**
- Modify: `requirements.txt`

`stable-baselines3` has zero imports anywhere in the fork (the codebase dropped SB3 entirely).
`tqdm`'s only consumer outside `utility_funcs.py` was `train_utils.py`, deleted in Task 3 — re-check before removing it. `gymnasium` and `pettingzoo` are unpinned, which is a reproducibility risk given PettingZoo's API drift; pin them to the versions SOT used and that these results were produced against. `numpy` is used throughout but only arrives transitively via torch.

- [ ] **Step 1: Confirm `stable-baselines3` really is unused**

```bash
grep -rn "stable_baselines3\|stable-baselines3\|sb3" --include=*.py --include=*.json --include=*.ipynb . | grep -v "^./docs/"
```

Expected: no output.

- [ ] **Step 2: Check whether `tqdm` is still used**

```bash
grep -rn "tqdm" --include=*.py . | grep -v "^./docs/"
```

If there are no hits, remove `tqdm` in step 3. If `utility_funcs.py` still imports it, keep it and note that in the commit message.

- [ ] **Step 3: Rewrite `requirements.txt`**

```
--extra-index-url https://download.pytorch.org/whl/cu124
torch==2.5.1+cu124
numpy
gymnasium==1.0.0
pettingzoo==1.24.3
opencv-python
matplotlib
tensorboard
pytest
```

(Keep `tqdm` if step 2 found a live consumer.)

- [ ] **Step 4: Verify the pinned versions match what is installed**

```bash
python3 -c "
import gymnasium, pettingzoo, numpy, torch
print('gymnasium', gymnasium.__version__)
print('pettingzoo', pettingzoo.__version__)
print('numpy', numpy.__version__)
print('torch', torch.__version__)
"
```

Expected: `gymnasium 1.0.0`, `pettingzoo 1.24.3`. If the installed versions differ, **pin to what is installed** and note the discrepancy in the commit message — do not upgrade or downgrade packages as part of this task.

- [ ] **Step 5: Confirm nothing broke**

```bash
python3 -m pytest tests/ -v
```

Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add requirements.txt
git commit -m "chore: drop unused deps, pin env-critical ones

- Removed stable-baselines3: zero imports anywhere (the fork dropped SB3).
- Pinned gymnasium and pettingzoo. PettingZoo API drift is a real
  reproducibility risk and these versions are what the results were
  produced against.
- Added numpy explicitly; it was only arriving transitively via torch.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 5: Package markers and import style

**Files:**
- Create: `src/__init__.py`, `src/env/utils/__init__.py`
- Modify: `src/train/video_utils.py:4`

The package currently works only via PEP 420 namespace packages, and only when the working directory is the repo root. It also mixes import styles: `video_utils.py:4` uses absolute `from src.env.utils import utility_funcs` while `agent.py:11` uses relative `from .utils import utility_funcs`. Both break under different rootdir conditions.

Convention adopted: **relative imports within `src/`, absolute `src.`-prefixed imports in `tests/` and `scripts/`.**

- [ ] **Step 1: Add the missing package markers**

```bash
touch src/__init__.py src/env/utils/__init__.py
```

- [ ] **Step 2: Find every absolute intra-package import**

```bash
grep -rn "^from src\.\|^import src\." src/
```

- [ ] **Step 3: Convert each hit to a relative import**

For `src/train/video_utils.py:4`:

```python
from ..env.utils import utility_funcs
```

Apply the equivalent transformation to any other hit from step 2.

- [ ] **Step 4: Verify imports still resolve from the repo root**

```bash
python3 -m pytest tests/test_imports.py -v
```

Expected: all PASS.

- [ ] **Step 5: Verify they also resolve from a different working directory**

```bash
cd /tmp && python3 -c "
import sys; sys.path.insert(0, '/home/assaf_caftory/CommonsGame/DanfoaTest')
import importlib
importlib.import_module('src.train.video_utils')
importlib.import_module('src.env.agent')
print('OK from foreign cwd')
"
```

Expected: `OK from foreign cwd`. This is the regression the package markers exist to prevent.

- [ ] **Step 6: Commit**

```bash
cd /home/assaf_caftory/CommonsGame/DanfoaTest
git add -A
git commit -m "chore: add package markers, unify intra-package imports

src/ and src/env/utils/ had no __init__.py, so the package resolved only via
PEP 420 namespace packages and only from the repo root. Import style was also
mixed (absolute src.* in video_utils.py, relative in agent.py).

Convention: relative imports inside src/, absolute src.* in tests/ and scripts/.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 6: Merge phase 1

- [ ] **Step 1: Full test run**

```bash
cd /home/assaf_caftory/CommonsGame/DanfoaTest && python3 -m pytest tests/ -v
```

Expected: all PASS.

- [ ] **Step 2: Compile check across the tree**

```bash
python3 -m compileall -q src/ tests/ scripts/ main.py && echo "compileall OK"
```

Expected: `compileall OK`.

- [ ] **Step 3: Merge**

```bash
git checkout main && git merge --no-ff chore/hygiene -m "Merge chore/hygiene: dead code removal and dependency cleanup

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
git log --oneline --graph -5
```

---

# PHASE 2 — Correctness fixes

Branch: `fix/correctness`. Each task fixes one audited defect and ships a regression test.

```bash
cd /home/assaf_caftory/CommonsGame/DanfoaTest && git checkout -b fix/correctness
```

### Task 7: The fire penalty corrupts social metrics

**Files:**
- Modify: `src/env/commons_env.py:73-83`
- Test: `tests/test_env_penalty.py` (create)

**The defect.** In `HarvestCommonsEnv.step()`, the `-1` penalty overwrites `rewards[agent_id]` at lines 79-81, and `update_social_metrics(rewards)` is then called at line 82 with the *penalized* values. `update_social_metrics` appends them to `self.rewards_record`, which is what Efficiency and Equality are computed from. So enabling `penalty` silently corrupts the social metrics — including the very metrics the reward model is trained to predict.

SOT does not have this bug: it applies the penalty one layer up, in the PettingZoo wrapper, *after* `commons_env.step()` has already recorded the metrics.

**The fix.** Record social metrics from the true environment rewards; return the penalized rewards to the caller.

**Interfaces:**
- Produces: no signature change. `step()` still returns `(observations, rewards, dones, infos)` with `rewards` penalized.

- [ ] **Step 1: Write the failing test**

Create `tests/test_env_penalty.py`:

```python
"""The FIRE penalty must not leak into the social metrics."""

from src.env.commons_env import HarvestCommonsEnv
from src.env.maps import SMALL_HARVEST_MAP

FIRE_ACTION = 7
STAND_STILL = 4


def _make_env(penalty: bool) -> HarvestCommonsEnv:
    env = HarvestCommonsEnv(
        ascii_map=SMALL_HARVEST_MAP,
        num_agents=2,
        ep_length=50,
        penalty=penalty,
    )
    env.reset()
    return env


def test_penalty_is_returned_to_the_caller():
    env = _make_env(penalty=True)
    _, rewards, _, _ = env.step({"agent-0": FIRE_ACTION, "agent-1": STAND_STILL})
    assert rewards["agent-0"] == -1


def test_penalty_does_not_enter_the_metrics_record():
    env = _make_env(penalty=True)
    env.step({"agent-0": FIRE_ACTION, "agent-1": STAND_STILL})
    # rewards_record feeds efficiency and equality; it must hold the true env
    # reward for the FIRE step, not the -1 penalty.
    assert env.rewards_record["agent-0"] == [0]


def test_metrics_identical_with_and_without_penalty_for_the_same_actions():
    actions = {"agent-0": FIRE_ACTION, "agent-1": STAND_STILL}

    penalised = _make_env(penalty=True)
    plain = _make_env(penalty=False)
    for _ in range(5):
        penalised.step(actions)
        plain.step(actions)

    penalised.compute_social_metrics()
    plain.compute_social_metrics()
    assert penalised.get_social_metrics() == plain.get_social_metrics()
```

- [ ] **Step 2: Run it to confirm it fails**

```bash
python3 -m pytest tests/test_env_penalty.py -v
```

Expected: `test_penalty_is_returned_to_the_caller` PASSES; the other two FAIL — `rewards_record["agent-0"]` is `[-1]`, and the two metric dicts differ. That contrast is the bug.

- [ ] **Step 3: Fix `step()`**

In `src/env/commons_env.py`, replace the body of `step()` (lines 73-83) with:

```python
    def step(self, action):
        observations, rewards, dones, infos = super().step(action)

        # Social metrics must reflect true environment rewards, so record them
        # before the FIRE penalty is applied. Applying the penalty first would
        # corrupt efficiency and equality -- the same quantities the reward
        # model is trained to predict.
        env_rewards = dict(rewards)

        for agent_id, _ in self.agents.items():
            infos[agent_id]['r'] = env_rewards[agent_id]
            infos[agent_id]['fire'] = action[agent_id] == 7
            self.fire_counter += int(action[agent_id] == 7)

        self.update_social_metrics(env_rewards)

        if self.penalty:
            for agent_id, _ in self.agents.items():
                if action[agent_id] == 7:
                    rewards[agent_id] = -1

        return observations, rewards, dones, infos
```

Note `infos[agent_id]['r']` now carries the true env reward. That is deliberate and consistent with `update_social_metrics`; `metrics.py` consumers read `r` for logging, which should report environment reward rather than shaped reward.

- [ ] **Step 4: Run the tests to verify they pass**

```bash
python3 -m pytest tests/test_env_penalty.py -v
```

Expected: all three PASS.

- [ ] **Step 5: Confirm nothing else regressed**

```bash
python3 -m pytest tests/ -v
```

Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add src/env/commons_env.py tests/test_env_penalty.py
git commit -m "fix: FIRE penalty no longer corrupts social metrics

HarvestCommonsEnv.step() overwrote rewards with the -1 FIRE penalty and then
passed the penalized values to update_social_metrics(), which records them in
rewards_record -- the basis for efficiency and equality. Enabling penalty
therefore silently corrupted the social metrics, including the ones the reward
model is trained to predict.

Social metrics are now recorded from true environment rewards; the penalty is
applied afterwards, to the returned rewards only. This matches how the previous
codebase behaved, where the penalty lived in the PettingZoo wrapper and ran
after the env had already recorded its metrics.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 8: IPPO orthogonal initialization uses one gain for every layer

**Files:**
- Modify: `src/train/algorithms/ippo.py:13-19`, `:247-248`
- Test: `tests/test_algorithm_init.py` (create)

**The defect.** `orthogonal_init` (`ippo.py:13-19`) walks every `nn.Linear` and `nn.Conv2d` in the module and applies the *same* gain. It is called as `orthogonal_init(actor, gain=0.01)` at `:247`, so the actor's entire convolutional trunk is initialized with gain 0.01, not just its policy head. The standard scheme (and SB3's) is layer-wise: √2 through the trunk, 0.01 on the policy head, 1.0 on the value head. A trunk at gain 0.01 produces vanishing activations, so the actor starts near-dead.

**The fix.** Give `orthogonal_init` separate trunk and head gains, applying the head gain only to the final layer.

**Interfaces:**
- Produces: `orthogonal_init(module, head_gain=1.0, trunk_gain=np.sqrt(2))` — Task 9 (MAPPO) calls this same function.

- [ ] **Step 1: Write the failing test**

Create `tests/test_algorithm_init.py`:

```python
"""Orthogonal initialization must use layer-wise gains."""

from unittest.mock import patch

import numpy as np
import torch.nn as nn

from src.train.algorithms.ippo import orthogonal_init


def _three_layer_module() -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(3, 8, 3),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(8, 16),
        nn.ReLU(),
        nn.Linear(16, 4),
    )


def test_head_gain_applies_only_to_the_final_layer():
    module = _three_layer_module()
    with patch("torch.nn.init.orthogonal_") as mock_init:
        orthogonal_init(module, head_gain=0.01, trunk_gain=np.sqrt(2))

    gains = [call.kwargs["gain"] for call in mock_init.call_args_list]
    assert len(gains) == 3, "expected one init per Conv2d/Linear layer"
    assert gains[-1] == 0.01, "final layer must get the head gain"
    assert all(g == np.sqrt(2) for g in gains[:-1]), "trunk must get the trunk gain"


def test_trunk_is_not_initialised_with_the_head_gain():
    """Regression guard: the whole actor used to be initialised at gain 0.01."""
    module = _three_layer_module()
    with patch("torch.nn.init.orthogonal_") as mock_init:
        orthogonal_init(module, head_gain=0.01)

    gains = [call.kwargs["gain"] for call in mock_init.call_args_list]
    assert gains.count(0.01) == 1, "only the head may use gain 0.01"


def test_biases_are_zeroed():
    module = _three_layer_module()
    for m in module.modules():
        if isinstance(m, (nn.Linear, nn.Conv2d)) and m.bias is not None:
            nn.init.constant_(m.bias, 5.0)

    orthogonal_init(module, head_gain=0.01)

    for m in module.modules():
        if isinstance(m, (nn.Linear, nn.Conv2d)) and m.bias is not None:
            assert m.bias.abs().sum().item() == 0.0
```

- [ ] **Step 2: Run it to confirm it fails**

```bash
python3 -m pytest tests/test_algorithm_init.py -v
```

Expected: FAIL with `TypeError: orthogonal_init() got an unexpected keyword argument 'head_gain'`.

- [ ] **Step 3: Rewrite `orthogonal_init`**

In `src/train/algorithms/ippo.py`, replace lines 13-19 with:

```python
def orthogonal_init(
    module: nn.Module,
    head_gain: float = 1.0,
    trunk_gain: float = np.sqrt(2),
) -> None:
    """Orthogonally initialize a module with layer-wise gains.

    The final Linear/Conv2d layer is the output head and receives ``head_gain``;
    every preceding layer is trunk and receives ``trunk_gain``. Applying a small
    head gain uniformly -- as this function previously did -- drives the trunk's
    activations toward zero and leaves the network near-dead at initialization.
    """
    layers = [m for m in module.modules() if isinstance(m, (nn.Linear, nn.Conv2d))]
    for index, layer in enumerate(layers):
        is_head = index == len(layers) - 1
        nn.init.orthogonal_(layer.weight, gain=head_gain if is_head else trunk_gain)
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)
```

- [ ] **Step 4: Update the call sites**

At `ippo.py:247-248`, replace:

```python
            orthogonal_init(actor, gain=0.01)  # Small gain for policy head
            orthogonal_init(critic, gain=1.0)
```

with:

```python
            orthogonal_init(actor, head_gain=0.01)
            orthogonal_init(critic, head_gain=1.0)
```

- [ ] **Step 5: Verify no other call sites remain on the old signature**

```bash
grep -rn "orthogonal_init" src/
```

Expected: the definition, the two calls above, and nothing else passing `gain=`.

- [ ] **Step 6: Run the tests to verify they pass**

```bash
python3 -m pytest tests/test_algorithm_init.py -v && python3 -m pytest tests/ -v
```

Expected: all PASS.

- [ ] **Step 7: Commit**

```bash
git add src/train/algorithms/ippo.py tests/test_algorithm_init.py
git commit -m "fix: layer-wise gains for orthogonal initialization

orthogonal_init applied a single gain to every Conv2d and Linear in the module,
and was called as orthogonal_init(actor, gain=0.01) -- so IPPO's entire
convolutional trunk was initialized at gain 0.01 rather than just the policy
head, leaving the actor near-dead at init.

The final layer now receives head_gain and the trunk receives trunk_gain
(sqrt(2) by default), matching the standard scheme.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 9: MAPPO has no orthogonal initialization at all

**Files:**
- Modify: `src/train/algorithms/mappo.py` (imports and the network-construction block around `:191-201`)
- Test: `tests/test_algorithm_init.py` (extend)

MAPPO builds its actor and centralized critic with PyTorch's default initialization while IPPO uses orthogonal — an inconsistency with no stated rationale. Reuse the function fixed in Task 8.

**Context.** `mappo.py:188-196` assigns `self.actor` and `self.critic` in both branches of a
`flatten_obs` if/else, then builds the optimizer at `:198`. The initialization goes between them, so
it covers both branches with one pair of calls.

- [ ] **Step 1: Confirm the construction block is where the plan expects**

```bash
sed -n '188,201p' src/train/algorithms/mappo.py
```

Expected: an `if self.config.flatten_obs:` / `else:` pair assigning `self.actor` and `self.critic`,
followed by `self.optimizer = torch.optim.Adam(`. If the block has moved, adjust the insertion point
in step 4 accordingly.

- [ ] **Step 2: Write the failing test**

Append to `tests/test_algorithm_init.py`:

```python
def test_mappo_imports_orthogonal_init():
    """MAPPO must use the same initialization scheme as IPPO."""
    import src.train.algorithms.mappo as mappo

    assert hasattr(mappo, "orthogonal_init"), (
        "mappo.py should import orthogonal_init from ippo.py"
    )
```

- [ ] **Step 3: Run it to confirm it fails**

```bash
python3 -m pytest tests/test_algorithm_init.py::test_mappo_imports_orthogonal_init -v
```

Expected: FAIL on the `hasattr` assertion.

- [ ] **Step 4: Import and apply it**

Add to `src/train/algorithms/mappo.py`'s imports:

```python
from .ippo import orthogonal_init
```

Then insert the initialization after the `if/else` block closes (after `:196`) and before
`self.optimizer = torch.optim.Adam(` at `:198`, so it applies to both branches:

```python
            self.critic = CNNCritic(global_shape, num_agents).to(self.device)

        orthogonal_init(self.actor, head_gain=0.01)
        orthogonal_init(self.critic, head_gain=1.0)

        self.optimizer = torch.optim.Adam(
```

Match the surrounding indentation — the `if/else` bodies are one level deeper than the
`orthogonal_init` calls, which sit at method-body level alongside `self.optimizer`.

- [ ] **Step 5: Run the tests**

```bash
python3 -m pytest tests/ -v
```

Expected: all PASS. If importing `ippo` from `mappo` creates a circular import, **stop and report** rather than working around it — that would indicate the shared helper belongs in its own module, which is a design change worth surfacing.

- [ ] **Step 6: Commit**

```bash
git add src/train/algorithms/mappo.py tests/test_algorithm_init.py
git commit -m "fix: apply orthogonal initialization in MAPPO

MAPPO built its actor and centralized critic with PyTorch defaults while IPPO
used orthogonal init, an inconsistency with no stated rationale. Reuses the
layer-wise helper from ippo.py.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 10: Adam epsilon for the PPO family

**Files:**
- Modify: `src/train/algorithms/ippo.py:250-253`, `src/train/algorithms/mappo.py:198-201`
- Test: `tests/test_algorithm_init.py` (extend)

SB3's `ActorCriticPolicy` injects `eps=1e-5` when the optimizer is Adam. The fork uses PyTorch's default `1e-8`. This is a well-known PPO implementation detail: the larger epsilon stabilizes updates when advantage-scaled gradients are small.

DQN is deliberately excluded — SB3's DQN does *not* get the epsilon override, so `1e-8` is correct there.

**On the test form below.** These assert against module source text rather than a constructed
optimizer. That is weaker than a behavioral test and is a deliberate compromise: the optimizers are
built inside `on_env_ready()`, which needs a live environment, and standing one up here would make a
one-line change depend on the whole env stack. Plan 4's test-coverage phase introduces a shared env
fixture; these three tests should be rewritten against it then.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_algorithm_init.py`:

```python
import inspect

PPO_ADAM_EPS = 1e-5


def test_ippo_optimizer_uses_ppo_adam_eps():
    source = inspect.getsource(__import__("src.train.algorithms.ippo", fromlist=["x"]))
    assert "eps=" in source, "IPPO's Adam should set eps explicitly"
    assert str(PPO_ADAM_EPS) in source or "1e-5" in source


def test_mappo_optimizer_uses_ppo_adam_eps():
    source = inspect.getsource(__import__("src.train.algorithms.mappo", fromlist=["x"]))
    assert "eps=" in source, "MAPPO's Adam should set eps explicitly"
    assert str(PPO_ADAM_EPS) in source or "1e-5" in source


def test_dqn_does_not_use_ppo_adam_eps():
    """DQN keeps PyTorch's default 1e-8; SB3 does not override it for DQN."""
    source = inspect.getsource(__import__("src.train.algorithms.dqn", fromlist=["x"]))
    assert "1e-5" not in source
```

- [ ] **Step 2: Run to confirm the first two fail**

```bash
python3 -m pytest tests/test_algorithm_init.py -k adam -v
```

Expected: the IPPO and MAPPO cases FAIL; the DQN case PASSES.

- [ ] **Step 3: Add the epsilon in IPPO**

At `ippo.py:250-253`:

```python
            optimizer = torch.optim.Adam(
                list(actor.parameters()) + list(critic.parameters()),
                lr=self.config.learning_rate,
                eps=1e-5,  # PPO convention; PyTorch's 1e-8 default is less stable here
            )
```

- [ ] **Step 4: Add the epsilon in MAPPO**

At `mappo.py:198-201`, add `eps=1e-5` to the `torch.optim.Adam(...)` call in the same way.

- [ ] **Step 5: Run the tests**

```bash
python3 -m pytest tests/ -v
```

Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add src/train/algorithms/ippo.py src/train/algorithms/mappo.py tests/test_algorithm_init.py
git commit -m "fix: set Adam eps=1e-5 for the PPO family

SB3's ActorCriticPolicy injects eps=1e-5 for Adam; the fork used PyTorch's
1e-8 default. Standard PPO implementation detail -- the larger epsilon
stabilizes updates when advantage-scaled gradients are small.

DQN deliberately keeps 1e-8: SB3 does not override epsilon for DQN.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 11: DQN has no gradient clipping

**Files:**
- Modify: `src/train/algorithms/dqn.py:112-114`, `src/train/config.py` (`DQNConfig`)
- Test: `tests/test_algorithm_init.py` (extend)

SB3's DQN silently applies `max_grad_norm=10`. The fork's `train_step` does `zero_grad()` / `backward()` / `step()` with no clipping. IPPO and MAPPO both clip; DQN is the outlier.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_algorithm_init.py`:

```python
def test_dqn_config_exposes_max_grad_norm():
    from src.train.config import DQNConfig

    assert hasattr(DQNConfig(), "max_grad_norm")
    assert DQNConfig().max_grad_norm == 10.0, "SB3's DQN default is 10"


def test_dqn_train_step_clips_gradients():
    source = inspect.getsource(__import__("src.train.algorithms.dqn", fromlist=["x"]))
    assert "clip_grad_norm_" in source, "DQN's train_step must clip gradients"
```

- [ ] **Step 2: Run to confirm it fails**

```bash
python3 -m pytest tests/test_algorithm_init.py -k "grad_norm or clips" -v
```

Expected: both FAIL.

- [ ] **Step 3: Add the config field**

In `src/train/config.py`, add to `DQNConfig` (after `target_update_freq`):

```python
    max_grad_norm: float = 10.0
```

- [ ] **Step 4: Apply clipping in `train_step`**

In `src/train/algorithms/dqn.py`, replace lines 112-114:

```python
        self.model.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
        self.model.optimizer.step()
```

`dqn.py:8` already has `from torch import nn`, so `nn.utils.clip_grad_norm_` resolves without a new import.

- [ ] **Step 5: Run the tests**

```bash
python3 -m pytest tests/ -v
```

Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add src/train/algorithms/dqn.py src/train/config.py tests/test_algorithm_init.py
git commit -m "fix: clip DQN gradients

SB3's DQN silently applies max_grad_norm=10; the fork's train_step had no
clipping at all, making DQN the only algorithm here without it. Exposed as
DQNConfig.max_grad_norm, defaulting to SB3's 10.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 12: The reward model reads DQN's normalize_obs regardless of algorithm

**Files:**
- Modify: `src/train/trainer.py:89-96`
- Test: `tests/test_trainer_obs.py` (create)

**The defect.** `Trainer._format_reward_obs` resolves observation normalization by reading `self.config.algorithm.dqn.normalize_obs` — hardcoded to the `dqn` section no matter which algorithm is running. It is currently benign only because `DQNConfig.normalize_obs` defaults to `True` and the IPPO/MAPPO JSON configs omit a `dqn` block. Set `dqn.normalize_obs: false` while running IPPO and the reward model sees 0–255 inputs while the policy sees 0–1 — a silent, hard-to-diagnose scale mismatch.

Note `MAPPOConfig.normalize_obs` defaults to `False` while `DQNConfig`'s is `True`, so this fix **changes behavior for MAPPO runs**: the reward model will stop normalizing. That is the correct behavior — the reward model should see what the policy sees — and it is worth calling out in the commit message.

**The fix.** Resolve the active algorithm's config section via `AlgorithmConfig.name`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_trainer_obs.py`:

```python
"""The reward model must normalize observations the same way the policy does."""

import numpy as np
import pytest

from src.train.config import AlgorithmConfig
from src.train.trainer import Trainer


class _ConfigStub:
    def __init__(self, algorithm: AlgorithmConfig):
        self.algorithm = algorithm


def _format(trainer_self, algo_name: str, normalize: bool) -> np.ndarray:
    algorithm = AlgorithmConfig(name=algo_name)
    getattr(algorithm, algo_name).normalize_obs = normalize
    trainer_self.config = _ConfigStub(algorithm)
    obs = {"agent-0": {"curr_obs": np.full((3, 3, 3), 255, dtype=np.uint8)}}
    return Trainer._format_reward_obs(trainer_self, obs, "agent-0")


@pytest.mark.parametrize("algo_name", ["dqn", "ippo", "mappo"])
def test_normalization_follows_the_active_algorithm(algo_name):
    stub = object.__new__(Trainer)

    normalized = _format(stub, algo_name, normalize=True)
    assert normalized.max() == pytest.approx(1.0)

    raw = _format(stub, algo_name, normalize=False)
    assert raw.max() == pytest.approx(255.0)


def test_dqn_setting_does_not_leak_into_ippo():
    """Regression guard: normalization used to read the dqn section always."""
    stub = object.__new__(Trainer)
    algorithm = AlgorithmConfig(name="ippo")
    algorithm.dqn.normalize_obs = False   # must be ignored
    algorithm.ippo.normalize_obs = True   # must be honoured
    stub.config = _ConfigStub(algorithm)

    obs = {"agent-0": {"curr_obs": np.full((3, 3, 3), 255, dtype=np.uint8)}}
    result = Trainer._format_reward_obs(stub, obs, "agent-0")
    assert result.max() == pytest.approx(1.0)
```

- [ ] **Step 2: Run to confirm it fails**

```bash
python3 -m pytest tests/test_trainer_obs.py -v
```

Expected: the `ippo` and `mappo` parametrized cases FAIL, and `test_dqn_setting_does_not_leak_into_ippo` FAILS with `255.0 != 1.0`.

- [ ] **Step 3: Fix `_format_reward_obs`**

In `src/train/trainer.py`, replace lines 89-96:

```python
    def _format_reward_obs(self, obs: dict, agent_id: str) -> np.ndarray:
        """Format an observation for the reward model.

        Normalization must follow the *active* algorithm's setting so the reward
        model sees inputs on the same scale as the policy does.
        """
        img = obs[agent_id]["curr_obs"]
        algorithm_cfg = getattr(self.config, "algorithm", None)
        normalize = False
        if algorithm_cfg is not None:
            active = getattr(algorithm_cfg, algorithm_cfg.name, None)
            normalize = getattr(active, "normalize_obs", False)
        if normalize:
            return (img / 255.0).astype(np.float32)
        return img.astype(np.float32)
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
python3 -m pytest tests/test_trainer_obs.py -v
```

Expected: all PASS.

- [ ] **Step 5: Full suite**

```bash
python3 -m pytest tests/ -v
```

Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add src/train/trainer.py tests/test_trainer_obs.py
git commit -m "fix: reward model follows the active algorithm's normalize_obs

_format_reward_obs read config.algorithm.dqn.normalize_obs regardless of which
algorithm was running. Benign only by accident: DQNConfig.normalize_obs
defaults to True and the IPPO/MAPPO configs omit a dqn block. Setting
dqn.normalize_obs=false under IPPO would have fed the reward model 0-255
inputs while the policy saw 0-1.

Behavior change for MAPPO: MAPPOConfig.normalize_obs defaults to False, so the
reward model no longer normalizes on MAPPO runs. That is correct -- the reward
model should see what the policy sees.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 13: Merge phase 2

- [ ] **Step 1: Full test run**

```bash
cd /home/assaf_caftory/CommonsGame/DanfoaTest && python3 -m pytest tests/ -v
```

Expected: all PASS.

- [ ] **Step 2: Compile check**

```bash
python3 -m compileall -q src/ tests/ scripts/ main.py && echo "compileall OK"
```

- [ ] **Step 3: Review the full phase diff before merging**

```bash
git diff main...fix/correctness --stat
```

Expected: changes confined to `src/env/commons_env.py`, `src/train/algorithms/{ippo,mappo,dqn}.py`, `src/train/{trainer,config}.py`, and three new test files.

- [ ] **Step 4: Merge**

```bash
git checkout main && git merge --no-ff fix/correctness -m "Merge fix/correctness: audited defect fixes

- FIRE penalty no longer corrupts social metrics
- Layer-wise orthogonal init (IPPO trunk was initialized at gain 0.01)
- Orthogonal init added to MAPPO
- Adam eps=1e-5 for the PPO family
- Gradient clipping for DQN
- Reward model follows the active algorithm's normalize_obs

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
git log --oneline --graph -12
```

---

## Deferred to later plans

Recorded here so nothing is silently dropped:

- **Module-global `ACTIONS['FIRE']` mutation** — spec §4.3 lists this under Tier 1, but in the fork `commons_env.py:10` is a plain module-level constant (`ACTIONS['FIRE'] = 7`), not a per-instance mutation. The mutation is SOT's pattern and only becomes relevant when beam length becomes spec-dependent. **Moved to Plan 2, Task: env spec registry.**
- **Duplicate penalty sites** — the second site is in `pettingzoo_env.py`, which is dead code deleted wholesale in Plan 2's metric-wiring phase. Fixing it here would mean editing a file about to be removed. **Deferred to Plan 2.**
- Env spec registry, metric/oracle unification → **Plan 2**
- Parameter sharing, vectorized envs, frame stacking, reward-model ensemble → **Plan 3**
- Config hardening, broad test coverage, README, `docs/experiment-schedule.md` → **Plan 4**
