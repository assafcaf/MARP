# Foundation Plan — Handoff

Written after Plan 1 ("Foundation",
[`docs/superpowers/plans/2026-08-11-foundation.md`](../superpowers/plans/2026-08-11-foundation.md))
finished and merged to `main`. During execution a controller kept a scratch ledger at
`.superpowers/sdd/2026-08-11-foundation/progress.md` (git-ignored, not part of history) recording
human rulings, deferred findings, and places where the plan document drifted from what shipped. That
ledger is being deleted; this doc is what survives it. Written for whoever writes or executes Plan 2
(env spec registry, metric/oracle wiring), Plan 3 (sample efficiency), or Plan 4 (config hardening,
test coverage, docs) with no memory of the Foundation execution.

## 1. What Foundation actually delivered

The repo was re-seeded from `almogze/marp` master @ `7cff81c` (root commit `6d582bc`, "Seed
repository from almogze/marp @7cff81c"), replacing the prior history entirely. Two phases followed,
each merged `--no-ff` into `main`, plus a final fix wave prompted by a whole-branch review:

| Phase | Branch | Merge commit | Content |
|---|---|---|---|
| Phase 1 | `chore/hygiene` | `d65ebad` | Dead code removal, dependency cleanup, package markers/import style — no behavior change |
| Phase 2 | `fix/correctness` | `8969c80` | Six audited correctness fixes (Tasks 7–12) |
| Final review | `fix/final-review` | `5f71cae` | Four findings from a whole-branch review: one real regression, three doc/test-quality fixes |

Current `main` HEAD: **`5f71cae`**. Suite state: **60 passed, 1 xfailed, 0 warnings**
(`python3 -m pytest -q`, verified against the live tree while writing this doc). All three phase
branches (`chore/hygiene`, `fix/correctness`, `fix/final-review`) are preserved locally; nothing was
pushed to a remote.

Git history from here carries the detail — read the thirteen task commits between `6d582bc` and
`8969c80`, then `4b4b0e7`/`9d5e4a7`/`f772bb3` for the final wave, for exact diffs.

## 2. Human rulings — decisions a future reader could otherwise reverse

Six rulings were made by a human during execution, outside the plan document. Each binds something
that isn't visible from the code alone.

### Ruling 1 — destructive reseed authorized
Explicit consent was given to run Phase 0's repo re-init in place on `main` (deletes `.git`,
re-initializes history). No worktree was possible for that step. Not actionable going forward — it's
a one-time fact about how `6d582bc` came to exist — but explains why `main`'s history starts there
and not earlier.

### Ruling 2 — "the plan governs on test form" — **later overturned, see Ruling 6**
Initially: where the plan prescribed a weak test form (`hasattr` checks, source-text/`inspect.getsource`
assertions, a retained-but-uncalled `uses_external_loop()`), the plan's literal test form was to be
implemented as written, not strengthened, and objections parked rather than re-opened. Rationale at
the time: the plan claimed building a real `IPPOAlgorithm`/`MAPPOAlgorithm` for a behavioral test
needed a live environment, which looked like a legitimate scope boundary for a hygiene/correctness
plan. **This premise turned out to be false — see Ruling 6, which reverses this ruling's practical
effect.** The `uses_external_loop()` retention itself is unaffected: it's dead code kept as
documented public API for a future algorithm, not a weak test.

### Ruling 3 — pre-existing test failure diagnosed and marked `xfail(strict=True)`
`tests/test_metrics.py::TestCheckAteLastAppleInCluster::test_cluster_with_apples_at_spawn_radius`
failed on the freshly seeded fork. Diagnosis: the test is correct; the code has the known
squared-distance bug — `src/train/metrics.py:93` and `src/env/commons_env.py:156` both compare a
squared distance against an unsquared radius (`j**2 + k**2 <= apple_radius` with `APPLE_RADIUS = 2`),
so offset `(0, 2)` gives `4 <= 2` → `False` and a boundary apple is dropped, yielding a 9-cell 3×3
block instead of `B₂`'s 13 cells. This is design spec §4.1 "Known deviations, deviation #1" — a
defect shared with the pre-refactor codebase, not introduced here.

**Ruling:** mark the test `xfail(strict=True)` with a comment citing spec §4.1 deviation #1 and
**"Plan 2 / Phase 3"** as the place it gets fixed under the `legacy` preset. Rationale: keeps the
suite green so later tasks' "all PASS" gates stay meaningful, keeps the defect visible instead of
silently skipped, and `strict=True` means an XPASS becomes a hard failure — so **when Plan 2/Phase 3
lands the real fix, this xfail marker must be removed** (or the suite will fail on the newly-passing
test). Do not "fix" `metrics.py:93`/`commons_env.py:156` as a drive-by anywhere else; that fix belongs
to Plan 2/Phase 3 as a deliberate, spec-referenced change, not an incidental one.

**Consequence still binding:** "all tests pass" for this codebase currently means 60 passed + 1
xfailed. An XPASS on this specific test is a signal Plan 2/Phase 3 has landed, not a bug.

### Ruling 4 — plan-mandated `parametrize` generator fixed despite being "as specified"
The plan's Task 3 Step 1 prescribes `tests/test_imports.py` with `@pytest.mark.parametrize` fed a
generator, which emits `PytestRemovedIn10Warning` (pytest v10 drops non-`Collection` iterable
support, so the import guard later tasks depend on would eventually break outright). Both the
implementer and reviewer flagged it independently.

**Ruling:** fix it — wrap as `list(_iter_module_names())`. Behavior-preserving (same 23 modules,
same walk), but a deliberate deviation from the plan's literal snippet. The plan document was not
updated to match; see §3 below.

### Ruling 5 — broken prescribed code in Task 11, not implemented as written
Task 11 Step 4 prescribes `nn.utils.clip_grad_norm_(self.model.parameters(),
self.config.max_grad_norm)` inside `DQNAgent.train_step`. `DQNAgent` (`dqn.py`) has no `self.config`
— it receives config values as explicit constructor kwargs, which `DQNAlgorithm.on_env_ready` passes
from `self.config` (where `self.config` legitimately exists, one class up). Written literally, the
plan's snippet raises `AttributeError` on the first training step, and the plan's own prescribed test
(a source-text grep for `"clip_grad_norm_"`) would not have caught it.

**Ruling:** thread `max_grad_norm` through `DQNAgent.__init__` like every other config field — ctor
param, `self.max_grad_norm`, passed from `DQNAlgorithm.on_env_ready`, clip with `self.max_grad_norm`.
This is what actually shipped (`config.py:34`, `dqn.py:43,56,116,153`). **See §3 for the precise plan
correction — this is the single most dangerous piece of drift in the plan document**, because a
future agent re-reading Task 11 literally would reintroduce the `AttributeError`.

### Ruling 6 — Ruling 2's premise disproven; source-text tests converted to behavioral ones now
The final whole-branch review directly disproved the "needs a live environment" premise behind Ruling
2: `IPPOAlgorithm.on_env_ready` and `MAPPOAlgorithm.on_env_ready` touch only three duck-typed
attributes off `env` (`env.observation_space["curr_obs"].shape`, `env.action_space.n`,
`env.agents.keys()`). The reviewer built both algorithms against a 5-line `FakeEnv` stub (CPU-only),
read back real optimizer `eps` values and real per-layer weight gains, and drove `DQNAgent.train_step`
with a spy on `clip_grad_norm_` — no environment construction needed anywhere.

**Ruling:** convert the five affected source-text tests to behavioral ones immediately, on this
branch, rather than deferring to Plan 4 as the plan document originally scheduled. Landed in
`f772bb3` via a new `tests/conftest.py::FakeEnv` fixture (shared, importable as
`from tests.conftest import FakeEnv`, also exposed as a `fake_env` pytest fixture). Rationale given:
MAPPO's orthogonal init was guarded only by `hasattr(mappo, "orthogonal_init")`, and **Plan 3
rewrites that exact `on_env_ready` block for parameter sharing** — a dropped or mis-indented call
under the old test form would fail no test and surface only as a silently worse learning curve.
Mutation testing (deleting both init calls; mis-indenting one into a single branch) confirmed the new
tests actually catch both failure modes.

**Consequence for Plan 4:** the "shared env fixture, rewrite these tests behaviorally" line item that
Plan 4 was going to own is **already done**. Don't redo it; `tests/conftest.py::FakeEnv` is the
fixture. Plan 4's test-coverage phase should extend `FakeEnv` usage to new gaps (§4 below), not
reinvent it.

## 3. Corrections to the plan document

`docs/superpowers/plans/2026-08-11-foundation.md` was **not edited** to match what shipped. This
section is the reconciliation list — precise enough to act on without re-deriving it. (This handoff
doc does not modify the plan; that edit is still outstanding.)

| # | Location | What the plan says | What actually shipped | Why it matters |
|---|---|---|---|---|
| 1 | Task 11, Step 4 (`docs/superpowers/plans/2026-08-11-foundation.md:1085-1096`) | `nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)` inside `DQNAgent.train_step`, edited at "`dqn.py:112-114`" | `DQNAgent` has no `self.config`; the value is threaded through the constructor instead. Four real edit sites: `config.py:34` (`DQNConfig.max_grad_norm: float = 10.0`), `dqn.py:43` (ctor param, no default), `dqn.py:56` (`self.max_grad_norm = max_grad_norm`), `dqn.py:153` (`DQNAlgorithm.on_env_ready` passes `max_grad_norm=self.config.max_grad_norm`). Clip call at `dqn.py:116` uses `self.max_grad_norm`. | **Dangerous.** The plan's snippet raises `AttributeError` on the first training step if implemented literally. See Ruling 5. |
| 2 | Task 10, "On the test form below" (`:961-965`) | Justifies source-text/`inspect.getsource` tests by claiming "the optimizers are built inside `on_env_ready()`, which needs a live environment, and standing one up here would make a one-line change depend on the whole env stack" | **Demonstrated false** by the final review (Ruling 6): `on_env_ready` needs only three duck-typed attributes, satisfied by a 5-line stub. All five affected tests were already converted to behavioral ones in `f772bb3`/`tests/test_algorithm_init.py`, using `tests/conftest.py::FakeEnv`. | If Plan 4 (or any later plan) reads this rationale at face value, it will re-justify weak tests instead of using the fixture that already exists. |
| 3 | Task 3, Step 1 (`:281`) vs. "Interfaces" (`:251`) | "Interfaces" names the produced test `test_all_modules_import`; the Step 1 code block itself defines `test_module_imports` | Shipped code is `test_module_imports` (matches the code block, not the prose) | Pre-existing inconsistency inside the plan document itself, not introduced by execution — the implementer correctly followed the code over the prose. Low stakes, but don't "fix" the test name to match the prose; the prose is wrong. |
| 4 | Task 3, Step 1 (`:281`) code block | `@pytest.mark.parametrize("module_name", _iter_module_names())` — a generator passed directly | Wrapped as `list(_iter_module_names())` (Ruling 4) | Literal form emits `PytestRemovedIn10Warning`; pytest v10 drops non-`Collection` iterable support for `parametrize`, which would eventually break this import guard outright. |
| 5 | File Structure section, Task 5 bullet (`:39-40` area) | Names only `video_utils.py:4` as needing an import-style change | Six absolute intra-package imports were actually converted: `trainer.py:6-9`, `video_utils.py:4`, `metrics.py:11` (→ `..env.*` / `..reward_model.*`) | Undercounts scope; Step 3's "any other hit" clause is what actually makes the task correct as executed. Not a defect, just don't be surprised the diff is bigger than the File Structure section implies. |

No other plan-vs-shipped drift was found during this write-up beyond what's tabulated above and what
Ruling 3/5/6 already describe.

## 4. Carry-forward items for Plans 2, 3, 4

Grouped by the plan that most naturally owns each. All verified against the ledger and, where
checkable without running code, against the current `main` tree.

### Plan 2 (env spec registry, metric/oracle wiring)
- **The squared-distance bug itself** (`src/train/metrics.py:93`, `src/env/commons_env.py:156`,
  `j**2 + k**2 <= apple_radius` vs. `APPLE_RADIUS = 2`) — spec §4.1 deviation #1. Fixing it must
  remove the `xfail(strict=True)` marker on
  `tests/test_metrics.py::TestCheckAteLastAppleInCluster::test_cluster_with_apples_at_spawn_radius`
  in the same change, or the suite will fail on an unexpected XPASS (Ruling 3).

### Plan 3 (sample efficiency)
- **`orthogonal_init`'s registration-order assumption.** It infers the output head as "the last
  Linear/Conv2d encountered in `module.modules()` order," which only works because every current
  network class registers submodules in forward order. This is now documented in the function's
  docstring (`ippo.py:13`, added in finding 3 of the final fix wave) but still structurally fragile.
  Plan 3's MAPPO parameter-sharing rewrite touches this exact `on_env_ready` code path — verify the
  assumption still holds after any restructuring, and confirm with `tests/test_algorithm_init.py`'s
  behavioral gain tests (not just import success).
- Consider moving `orthogonal_init` out of `ippo.py` into a shared module (e.g. `src/train/init.py`
  or similar) — it's currently imported into `mappo.py` via `from .ippo import orthogonal_init`
  (`mappo.py:11`), which works (no circular import) but means MAPPO depends on IPPO's module for a
  general-purpose utility that has nothing to do with IPPO specifically.
- **Duplicated network classes.** `MLPActor`, `MLPCritic`, `CNNActor`, `CNNCritic` are defined
  independently in both `src/train/algorithms/ippo.py` and `src/train/algorithms/mappo.py` — same
  class names, separately maintained. A change to one (e.g. an architecture tweak for sample
  efficiency) risks silently not propagating to the other. Worth consolidating into a shared module
  before or during Plan 3's architecture work.

### Plan 4 (config hardening, test coverage, docs)
- **Test-coverage gaps, specific and still open** (verified present in the current tree):
  - `tests/test_trainer_obs.py` bypasses `Trainer.__init__` entirely and never exercises
    `AlgorithmConfig.from_dict`; a future bug where `.name` fails to populate from a real
    `configs/*.json` file would not be caught here (deferred from Task 12).
  - The two PPO-family eps tests (`test_ippo_optimizer_uses_ppo_adam_eps`,
    `test_mappo_optimizer_uses_ppo_adam_eps`) now read `optimizer.param_groups[0]["eps"]` directly —
    this specific gap from Task 10 (independent substring checks that could pass while broken) was
    closed by the Ruling 6 rewrite, but confirm this if re-deriving coverage priorities from the plan
    document, which still describes the old, weaker version.
  - `tests/test_env_penalty.py`'s equivalence test only exercises the reward-is-zero case (FIRE +
    STAND_STILL never moves an agent onto an apple); it never covers a nonzero true reward
    co-occurring with a penalty hitting a different agent in the same step (deferred from Task 7; the
    separate pin test covers -1-vs-0 independently, so this is a real but narrower gap).
- **Unpinned `numpy`** in `requirements.txt` — every other ML dependency is pinned
  (`torch==2.5.1+cu124`, `gymnasium==1.0.0`, `pettingzoo==1.24.3`); `numpy` has no version constraint.
- **`pytest` sits in `requirements.txt`** (runtime requirements), not a separate dev/test
  requirements file — added in Task 2 to make the suite runnable at all, never revisited.
- **`MAPPOConfig.normalize_obs` defaults to `False`** (`config.py:80`) while `DQNConfig` and
  `IPPOConfig` both default to `True` (`config.py:35,58`). All three shipped configs
  (`configs/train_dqn.json`, `train_ippo.json`, `train_mappo.json`) explicitly set
  `normalize_obs: true`, so the mismatched default is currently masked by every shipped config
  overriding it — but the dataclass default itself is inconsistent and a config that omits the key
  would silently get different behavior for MAPPO than for the other two algorithms.
- **Logging inconsistency: `episode_rewards` is penalized, `efficiency` is not.** Confirmed by
  reading `trainer.py:144-235` against Task 7's fix: `episode_rewards` (and therefore
  `reward_sum`/`reward_mean`/`reward_per_agent` in the logged episode payload) accumulates the
  `rewards` dict returned from `env.step()`, which carries the FIRE penalty. `efficiency` (part of
  `env.get_social_metrics()`) is computed from the pre-penalty `env_rewards` snapshot that Task 7
  deliberately introduced so the penalty wouldn't corrupt social metrics. Both behaviors are
  individually correct and intentional (Task 7's whole point was to stop the penalty from leaking into
  metrics) — but the result is that two numbers logged side-by-side in the same episode payload answer
  different questions (post-penalty return vs. true environment productivity) with no label saying so.
  Worth a doc/log-key clarification in Plan 4, not a behavior change.

## 5. Residual risks for the future experiment schedule

No training runs were permitted during Foundation (verification was tests only). Several behavior
changes landed that affect learning dynamics and were never validated end-to-end. Record these in the
future `docs/experiment-schedule.md` as things to account for when scheduling runs — not as things to
go test now.

- **IPPO initialization change invalidates comparison against previously logged IPPO runs.** Before
  Task 8, `orthogonal_init` applied a single `gain=0.01` uniformly to the *entire* actor/critic,
  including the trunk. After, the trunk gets `gain=√2` (standard orthogonal-init convention) and only
  the output head keeps `gain=0.01`. This is a real, intentional initialization-scale change (not a
  cosmetic refactor) — trunk gain went from 0.01 to ~1.41 — and it changes early-training dynamics.
  Any IPPO run logged before `6e24e02` is not a fair baseline for any IPPO run after it. MAPPO never
  had orthogonal init before Task 9, so this doesn't apply retroactively to MAPPO (there is no prior
  MAPPO baseline to invalidate).
- **Reward-model checkpoints predating this work were trained on differently-scaled inputs.** Before
  Task 12's fix (`9376c4f`), `Trainer._format_reward_obs` always read `algorithm.dqn.normalize_obs`
  regardless of which algorithm was actually training. This was benign for DQN and IPPO (both default
  `normalize_obs=True`) but a live scale mismatch for MAPPO: `MAPPOConfig.normalize_obs` defaults
  `False` and `train_mappo.json` ships no `dqn` block, so any MAPPO run with the reward model enabled,
  trained before this fix, had its policy observing raw 0–255 pixel values while the reward model was
  trained on 0–1 normalized ones. Any reward-model checkpoint saved from a pre-fix MAPPO run is scaled
  inconsistently with what the fixed code now feeds it and should not be reused directly.
- Six correctness fixes landed total (Tasks 7–12); all were unit/behaviorally tested in isolation and
  their composition was checked by the final whole-branch review (per-layer gains measured on all six
  network classes, `eps` read back live from all four optimizers, `train_step` driven with a clip spy,
  the FIRE-penalty fix exercised in the nonzero-reward case). None were validated against an actual
  training curve. Any of the six is a candidate confound if a future run's learning curve looks
  different from a pre-Foundation run and the cause isn't otherwise obvious.

## Related documents

- [refactor-plan.md](refactor-plan.md) — the design-session handoff that led to adopting the
  `almogze/marp` fork; written before Foundation was planned or executed.
- [`docs/superpowers/plans/2026-08-11-foundation.md`](../superpowers/plans/2026-08-11-foundation.md)
  — the plan this doc corrects and closes out. Read together with §3 above before trusting any code
  snippet in it verbatim.
- [`docs/superpowers/specs/2026-08-11-marp-refactor-design.md`](../superpowers/specs/2026-08-11-marp-refactor-design.md)
  — §4.1 "Known deviations," source of the squared-distance bug's spec citation (Ruling 3).
