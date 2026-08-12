# Dead-Code / Repo-Hygiene Audit

Scope: factual inventory of what's live vs. suspect in this repo, relative to the confirmed live entrypoint chain (`src/teach_prm.py` / `src/teach_nrp.py` / `src/teach_crm.py` → `PRMRunner`/`NRPRunner`/`CRMRunner` in `src/experiment_runner/runners.py`). No deletions performed. Table format per item: **Item | Evidence | Verdict | Recommended action**.

Already-confirmed findings from prior work (not re-investigated here, included for completeness):
- `src/env2/` — near-duplicate of `src/env/`, referenced only by `src/stam.ipynb`. **Verdict: dead.**
- CRM path (`teach_crm.py`, `CRMRunner`, `crm.yaml`) — never produced a results dir; owner confirmed "could not make it work." **Verdict: dead / non-functional experimental path.**

---

## 1. Root-level scratch/loose files in `src/`

| Item | Evidence | Verdict | Recommended action |
|---|---|---|---|
| `src/test.py` (999 B) | Reads env, wraps with SuperSuit, steps it once, instantiates `CRMShardReplayBuffer()` and does nothing else with it. No `assert`s, no `if __name__`. Not imported anywhere (`grep` for `import test\b` / `from test import` → 0 hits). Last touched `22b6797` (2025-02-19, "added batchnorm to predictor nn"). | Dead — scratch/debug script | Safe to delete. |
| `src/test2.py` (1992 B) | Defines `EpisodeData` and `PRMEpisodeData` `NamedTuple`s with an `__add__`/concat method. Not imported anywhere. Content is an **earlier draft** of `src/buffers/episode_data.py::PRMEpisodeData` (near-identical docstring/logic; `episode_data.py`'s version additionally has `CRMEpisodeData` and drops the buggy generic `EpisodeData` class — note test2.py's `EpisodeData.__add__` has a bug, it `return`s inside the `for` loop after the first field, so it only ever concatenates one field). Last touched same commit as test.py, `22b6797` (2025-02-19). | Dead — superseded draft, now duplicated/fixed in `buffers/episode_data.py` (which is live, see §5) | Safe to delete. |
| `src/stam.ipynb` (115 KB, 8 cells) | Imports `from env2 import parallel_env` (confirms env2 dependency already flagged dead), builds a vec env, samples random actions, and its last cell calls `save_video(file_name, frames)` with `file_name = "test.mp4"` — this is exactly what produces `src/test.mp4`. Last touched `8652c81` (2025-03-23, "stabe prm and baseline"), same commit as `env2`. Not referenced by any `.py` file. | Dead — scratch notebook exploring env2, orphaned once env2 was abandoned | Safe to delete (or archive if owner wants the demo-video-generation snippet). |
| `src/test.mp4` (800 KB / 817,040 B) | Direct output artifact of `stam.ipynb` cell 7 (`save_video("test.mp4", frames)`). Same commit (`8652c81`) as the notebook that generates it. Not referenced by any code. | Dead — orphaned generated artifact, and a large binary sitting in git history | Safe to delete; also consider `git filter-repo`/BFG if repo-size cleanup is ever wanted (817 KB binary blob committed directly, not gitignored). |

---

## 2. `old_results/` vs `results/`

`ls old_results/` (13 entries, mtimes 2025-01-24→2025-02-16) vs `ls results/` (18 entries, mtimes 2025-03-20→2025-05-14). Both directories are gitignored (`/results`, `/old_results` in `.gitignore`) and untracked, so neither is in git history — only present on disk.

Naming pattern differs structurally, not just cosmetically:
- `old_results/`: `nrp-Efficiency-slow`, `nrp_independent-Efficiency-fast`, `nrp_penalty-Efficiency-slow-5_agents`, `prm-Efficiency-fast`, plus a bare `dqn/` dir — no agent-algorithm prefix, no reward-metric suffix.
- `results/`: `nrp-dqn-Efficiency-fast-3_agents`, `prm-dqn-episodial-Efficiency*Peace-fast-4_agents`, `prm-single-ppo-Efficiency-fast-1_agents` — consistently includes the SB3 algorithm (`dqn`/`ppo`), an `episodial`/`single`/`independent`/`penalty` variant tag, and an explicit `N_agents` suffix; also uses composite reward-metric names with `*` (e.g. `Efficiency*Peace`) that don't appear in `old_results` at all.

The `results/` naming pattern (algo + variant + composite metric + agent-count) matches the directory-naming logic currently built into `runners.py`'s config/experiment-name construction, while `old_results/`'s flatter scheme does not (it predates the DQN/PPO algorithm split and composite-metric feature). Chronology confirms this: `old_results` mtimes stop 2025-02-16, right before `edc1522`/`22b6797`/`8652c81` commits that add batchnorm and stabilize PRM; `results` mtimes only start 2025-03-20, right after those.

| Item | Evidence | Verdict | Recommended action |
|---|---|---|---|
| `old_results/` | Naming scheme incompatible with current runner output naming; date range entirely predates the "stable prm and baseline" commit (8652c81, 2025-03-23) | Superseded output from an earlier code version — not reproducible with current code/configs as-is (e.g. no composite-metric or per-algorithm dirs) | Not code, so no refactor action needed directly; safe to archive/delete as stale experiment output whenever the owner is done referencing old numbers. Not part of the code dead-code question. |
| `results/` | Naming matches current runner conventions, most recent runs (up to 2025-05-14) | Live/current | No action. |

---

## 3. `.mypy_cache/`, `__pycache__/`, `.vscode/`

| Item | Evidence | Verdict | Recommended action |
|---|---|---|---|
| `.mypy_cache/` | Present on disk at repo root. Not in `git ls-files` (0 tracked files under it). Not explicitly listed in the repo's `.gitignore` — but mypy self-generates `.mypy_cache/.gitignore` (containing `*`) which git honors as a nested ignore rule, so it's effectively ignored without the root `.gitignore` needing an entry. | Confirmed pure tooling artifact, not tracked | No action required, but adding an explicit `.mypy_cache/` line to the root `.gitignore` would make the ignore rule visible/robust instead of relying on mypy's self-written nested file. |
| `__pycache__/` (all packages) | **97 `.pyc` files are currently tracked in git** (`git ls-files | grep -c __pycache__` → 97), e.g. `src/buffers/__pycache__/buffers.cpython-311.pyc`. Root `.gitignore` contains only `/results`, `/old_results`, `/vscode` — no `__pycache__/` or `*.pyc` entry at all. | **Hygiene bug** — bytecode caches are committed to version control, not just an untracked local artifact | Flag for owner: add `__pycache__/` and `*.py[co]` to `.gitignore`, then `git rm -r --cached '**/__pycache__'` to untrack the 97 already-committed files. |
| `.vscode/` | Contains one tracked file, `.vscode/launch.json` (506 B, debug config). Root `.gitignore` has an entry `/vscode` (no leading dot) — this does **not** match the actual `.vscode/` directory, so it's a no-op / typo. | `.vscode/launch.json` is intentionally tracked (debug config), but the `/vscode` gitignore line is dead/mismatched | Minor: either delete the useless `/vscode` gitignore line, or if the intent was to exclude `.vscode/`, fix the typo — but since `launch.json` is currently tracked and looks intentional (personal IDE config committed for convenience), recommend leaving `.vscode/launch.json` tracked and just removing the stray `/vscode` gitignore line as cleanup. |

---

## 4. Package `__init__.py` exports — used vs. unused

Method: grepped each exported name across the whole repo excluding the exporting package's own folder.

| Package | Exported | Used outside own package? | Verdict |
|---|---|---|---|
| `env/__init__.py` | `parallel_env` | Yes — `experiment_runner/runners.py` | Live |
| `rl_agents/__init__.py` | `DQN`, `IndependentDQN`, `DQNPRM`, `DQNCRM`, `IndependentDQNRP`, `PPO`, `IndependentPPO`, `PPOPRM`, `CnnFeatureExtractor`, `CustomCNN` | Yes, all — `experiment_runner/runners.py` (and `CustomCNN` also used inside `reward_predictor/prm/nn2.py`) | Live |
| `reward_predictor/__init__.py` | `RPMRewardPredictor`, `CRMRewardPredictor`, `AgentLoggerSb3` | Yes — `experiment_runner/runners.py` | Live |
| `reward_predictor/__init__.py` | `parallel_collect_segments` (from `segment_sampling.py`) | **No** — grep outside the defining file itself finds zero call sites anywhere in the repo, including within `reward_predictor/` itself (only referenced at its own `def` and its own `__init__.py` import line) | **Dead export** — function is defined and re-exported but never called |
| `reward_predictor/__init__.py` | `LabelAnnealer` (from `label_schedules.py`) | **No** — same pattern, zero call sites anywhere, including inside the package | **Dead export** |
| `reward_predictor/__init__.py` | `function_wrapper` (from `utils.py`) | **No** — zero call sites anywhere | **Dead export** |
| `buffers/__init__.py` | `PRMShardReplayBuffer`, `CRMShardReplayBuffer`, `PRMShardReplayBufferEpisodial`, `PRMShardRolloutBuffer` | Yes, all four — `experiment_runner/runners.py` | Live |
| `learners/__init__.py` | `CollectiveRLRPLearner`, `IndependentRLRPLearner` | Yes — `experiment_runner/runners.py` | Live |
| `callbacks/__init__.py` | `SingleAgentCallback` | Yes — `experiment_runner/runners.py` | Live |
| `configs/__init__.py` | `Config` | Yes — `experiment_runner/runners.py` | Live |

Recommended action for the three dead exports (`parallel_collect_segments`, `LabelAnnealer`, `function_wrapper`): remove from `reward_predictor/__init__.py`; their source files (`segment_sampling.py`, `label_schedules.py`, `utils.py`) also contain other things (`utils.py::corrcoef` is live-used directly by `prm/reward_model.py` and `crm/reward_model.py` without going through `__init__.py`; `label_schedules.py::ConstantLabelSchedule` is defined but has zero call sites anywhere — same dead status as `LabelAnnealer`). Since another agent covers `reward_predictor` internals in depth, flagging here as **uncertain — needs owner input** on whether these were meant for a human-feedback-collection workflow that's simply not wired into any current runner (see §6 — `clip_manager.py`/`comparison_collectors.py` similarly reference an uninstalled `human_feedback_api` package and are also unreferenced by anything), rather than confidently dead.

---

## 5. Files present in a package folder but NOT re-exported by `__init__.py`

| File | Evidence | Verdict | Recommended action |
|---|---|---|---|
| `src/learners/independent_rlrp_learner.py` (219 lines) | Defines standalone `class IndependentRLRPLearner` (no base class). `learners/__init__.py` exports `IndependentRLRPLearner` from `.learners` instead (a different, `BaseLearner`-derived implementation, 307-line file). `grep -rn "independent_rlrp_learner"` across all `.py` files (excluding `__pycache__`) → **0 import references anywhere**, not even by direct/full-path import. Last git-touched at `843b881` ("first commit", 2025-11-28 per repo history — same commit as `learners.py`). | Dead — earlier/parallel standalone duplicate, same class name, superseded by the `BaseLearner`-derived version in `learners.py` which is what's actually exported and used | Safe to delete, pending confirmation from whoever is doing the deep learners.md analysis that `learners.py`'s version is behaviorally equivalent-or-better. |
| `src/learners/collective_rlrp_learner.py` (208 lines) | Same pattern: standalone `class CollectiveRLRPLearner` (no base class), duplicate name of the exported `learners.py::CollectiveRLRPLearner(BaseLearner)`. Zero import references anywhere in the repo. Same last-touch commit `843b881`. | Dead — same as above | Safe to delete, same caveat. |
| `src/buffers/buffers.py` (219 lines) | Defines `class PRMShardReplayBuffer(ReplayBuffer)` and `class CRMShardReplayBuffer(PRMShardReplayBuffer)` — duplicate names of what `buffers/__init__.py` actually exports from `replay_buffers.py` (which additionally has `PRMShardReplayBufferEpisodial`, a class `buffers.py` lacks entirely — evidence `replay_buffers.py` is the more-evolved successor). `grep` for any import of `.buffers` / `buffers.buffers` → **0 hits** anywhere in the repo. Both files were last touched at the *same* commit, `8652c81` (2025-03-23), suggesting `buffers.py` was left behind stale when `replay_buffers.py` picked up the Episodial variant. | Dead — earlier duplicate, superseded by `replay_buffers.py` | Safe to delete, pending confirmation from whoever is doing buffers.md deep-dive that no behavioral divergence matters. |
| `src/buffers/episode_data.py` (61 lines) | Defines `PRMEpisodeData` and `CRMEpisodeData` `NamedTuple`s. **Not** listed in `buffers/__init__.py`'s exports directly, but `grep` shows `src/buffers/replay_buffers.py:6: from .episode_data import PRMEpisodeData, CRMEpisodeData` — i.e. it's imported internally by `replay_buffers.py`, which *is* exported and live. Last touched at `843b881` ("first commit", 2025-11-28). | **Live** — used indirectly via `replay_buffers.py`, just not re-exported at package level (correctly so, since callers only need the buffer classes, not the raw tuple type) | No action; this is the expected "internal helper module, not exported" pattern, not dead code. |

Note: `src/test2.py` (§1) independently defines an `EpisodeData`/`PRMEpisodeData` pair that is an earlier, buggier draft of `buffers/episode_data.py`'s `PRMEpisodeData` — corroborating that `episode_data.py` is the evolved, kept version and `test2.py` is scratch.

---

## 6. `requirements.txt`

`cat requirements.txt` is a `pip freeze`-style pinned list (68 packages). Cross-checked every top-level `import`/`from` package name found across `src/**/*.py` against it.

| Check | Result |
|---|---|
| `supersuit` (imported as `import supersuit as ss`) | Present as `SuperSuit==3.9.3` (case differs, pip is case-insensitive — not an issue) |
| `stable_baselines3` | Present, `2.4.1` |
| `pettingzoo` | Present, `1.24.3` |
| `torch` | Present, `2.5.1` |
| `gym` / `gymnasium` | Both present (`gym==0.26.2`, `gymnasium==1.0.0`) — both are actually imported somewhere in the repo, so not redundant, though having both is itself a sign of an incomplete `gym`→`gymnasium` migration (worth flagging for the refactor, not a requirements.txt bug) |
| `cv2` (imported as `import cv2`) | Matches `opencv-python==4.11.0.86` |
| `yaml` | Matches `PyYAML==6.0.2` |
| `matplotlib`, `numpy`, `pandas`, `tqdm`, `psutil` | All present with pins |
| `tensorflow` (imported in `src/env/utils/train_utils.py` and `src/env2/utils/train_utils.py`) | **Not in requirements.txt**, and `python3 -c "import tensorflow"` fails locally (not installed) | Verified `train_utils.py` is **not imported by anything** in either `env/` or `env2/` (`env/__init__.py` only imports `.pettingzoo_env`; no file imports `train_utils`) — so this is a missing dependency for dead code, not a live-path gap. Not a requirements.txt bug. |
| `DDQN` (imported in same two `train_utils.py` files, `from DDQN import DDQNAgent, DeepQNet`) | Not in requirements.txt, and no local `DDQN` package/module exists in the repo (`find . -iname "DDQN*"` → nothing) | Same as above — orphaned import inside an unreachable file, not a live gap. |
| `human_feedback_api` (imported inside `src/reward_predictor/clip_manager.py` and `src/reward_predictor/comparison_collectors.py`, as deferred/local imports inside methods) | Not in requirements.txt, and not installed (`ModuleNotFoundError`) | Verified **neither `clip_manager.py` nor `comparison_collectors.py` is imported by anything else in the repo** (0 grep hits) — same as above, dead files reaching for a package that was never packaged/vendored. Ties into the §4 finding about `parallel_collect_segments`/`LabelAnnealer` — this looks like a whole never-wired-up "collect human feedback via a web UI" subsystem. |

| Item | Evidence | Verdict | Recommended action |
|---|---|---|---|
| `requirements.txt` itself | Every package actually imported by the **live** code path is present and pinned; the only "missing" packages (`tensorflow`, `DDQN`, `human_feedback_api`) are all imported exclusively by files with zero inbound references (`train_utils.py` ×2, `clip_manager.py`, `comparison_collectors.py`) | Not stale relative to live code — internally consistent | No action needed on the file itself. The three orphaned imports are a signal (see §4/§5) that there's a larger dead "human-feedback-collection + DDQN-based env utils" subsystem sitting in `env/utils/`, `env2/utils/`, and parts of `reward_predictor/` that never got wired into any runner — worth a follow-up pass by whoever owns `environment.md` since it's outside this audit's specifically-assigned file list. |

---

## Summary of files recommended for deletion (pending owner sign-off)

- `src/test.py`, `src/test2.py`, `src/stam.ipynb`, `src/test.mp4`
- `src/learners/independent_rlrp_learner.py`, `src/learners/collective_rlrp_learner.py`
- `src/buffers/buffers.py`
- Three dead exports in `src/reward_predictor/__init__.py`: `parallel_collect_segments`, `LabelAnnealer`, `function_wrapper` (uncertain — see §4/§6, may be part of an intentionally-shelved human-feedback subsystem)

## Summary of hygiene flags (not code deletions)

- 97 `__pycache__/*.pyc` files are tracked in git; `.gitignore` has no `__pycache__/`/`*.pyc` rule.
- `.gitignore`'s `/vscode` line doesn't match the actual `.vscode/` directory (typo/dead rule); `.vscode/launch.json` is tracked, apparently intentionally.
- `old_results/` (13 dirs, Jan–Feb 2025) uses a naming scheme incompatible with current runner output and predates the March 2025 "stable prm and baseline" commit — stale experiment output, not reproducible by current code as-is.
