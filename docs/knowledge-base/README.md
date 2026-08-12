# DanfoaTest Knowledge Base

Technical reference for this repo, built for an upcoming refactor. Compiled by reading the full
entrypoint→runner call graph directly, then dispatching one focused review per subsystem. Every
claim in the linked docs traces to a specific file/line; anything not confirmed is marked
"uncertain — needs owner input" rather than guessed. Source paper:
`docs/paper-tmlr-anonymized-2026-07-26.pdf` — **"Why Study Emergent Behavior When You Can Regulate
It? Aligning Multi-Agent Systems with Reward Prediction"** (anon. TMLR submission). The paper's own
name for this framework is **MARP** (Multi-Agent Reward Prediction); "PRM/NRP/CRM" are code-only
abbreviations that never appear in the paper text.

**Refactor in progress**: see [refactor-plan.md](refactor-plan.md) for the decisions made, the
`almogze/marp` fork adopted as the new codebase's starting point, and the next steps.

## Documents

| Doc | Covers |
|---|---|
| [environment.md](environment.md) | `src/env/` game mechanics, obs/action space, reward wiring, map system, `env/` vs `env2/` |
| [learners-and-agents.md](learners-and-agents.md) | `src/rl_agents/`, `src/learners/`, `src/callbacks/` — class hierarchy, independent vs collective training, reward-injection point |
| [reward-predictor.md](reward-predictor.md) | `src/reward_predictor/` — the preference-comparison pipeline, where labels actually come from, why Local-Trajectory *and* Joint-Episode both live inside PRM, and how CRM is a separate third design |
| [buffers.md](buffers.md) | `src/buffers/` — shard buffer class map, why "shard", PRM vs CRM buffer diff |
| [paper-and-configs.md](paper-and-configs.md) | Paper↔code terminology mapping, metric definitions (Efficiency/Equality/Sustainability/Peace), all 7 YAML configs mapped to `results/` naming |
| [dead-code-audit.md](dead-code-audit.md) | File-by-file live/dead inventory, `__init__.py` export audit, repo hygiene |
| [refactor-plan.md](refactor-plan.md) | **Start here for the refactor.** Decisions made, the `almogze/marp` fork adopted as the new base, its architecture, gaps found, and the recommended next steps |

## 1. The live path, in one picture

```
teach_prm.py ──┐
teach_nrp.py ──┼──► experiment_runner/runners.py ──► BaseRunner.{PRMRunner,NRPRunner,CRMRunner}
teach_crm.py ──┘         │
                          ├─ env/            (parallel_env — NOT env2/, which is dead)
                          ├─ configs/*.yaml  (7 files, loaded via configs.Config)
                          ├─ rl_agents/      (DQN/PPO + Independent*/​*PRM/*CRM/*RP variants)
                          ├─ reward_predictor/  (prm/, crm/ — only for PRMRunner, CRMRunner)
                          ├─ buffers/        (PRMShard*/CRMShard* replay/rollout buffers)
                          ├─ learners/       (CollectiveRLRPLearner, IndependentRLRPLearner)
                          └─ callbacks/      (SingleAgentCallback — eval/video/logging)
```

Only three files are true entrypoints. Everything reachable from their import graph is
presumptively live; everything else (`env2/`, `test.py`, `test2.py`, `stam.ipynb`, the two
standalone `*_rlrp_learner.py` files, `buffers/buffers.py`, the `clip_manager.py`/
`comparison_collectors.py`/`segment_sampling.py`/`label_schedules.py` cluster) is dead scaffolding
— see §4.

## 2. Terminology — trust the owner, not the README (and see the correction below)

The three experimental conditions have **three different definitions in circulation** (README vs.
repo owner vs. paper). Full detail in [paper-and-configs.md §1](paper-and-configs.md#1-terminology-conflicts-surface-these-to-the-repo-owner--not-silently-resolved).

**Correction to the first pass of this knowledge base**: the original review mapped PRM ≈
Local-Trajectory and CRM ≈ Joint-Episode. That mapping was wrong, caught on a second pass prompted
by re-reading the paper's Method section (p.6-8) against the buffer code. The paper states both its
variants use **the same single-agent-input reward model** — Joint-Episode differs only by training
on "concatenated inputs that combine all agent trajectories from a given episode into a single
sequence," never a global state. That's exactly what `prm.yaml`'s `RP_PARAMETERS.episodial` flag
does inside `reward_predictor/prm/` (`buffers.md` §3): `episodial: False` = Local-Trajectory,
`episodial: True` = Joint-Episode — both sharing one reward-model class and loss. **CRM is not
either paper variant** — it conditions its model on `env.state_space` (the global full-map state),
an architecture the paper never describes. Full derivation in
[reward-predictor.md §5](reward-predictor.md#5-correction-local-trajectory-vs-joint-episode-are-both-inside-prm--crm-is-a-third-separate-thing).

| Code term | README says | Owner says (authoritative) | Paper's closest concept | Status |
|---|---|---|---|---|
| **PRM** | "Preference-based Reward Model" | "personalized reward model" | Houses **both** paper variants: `episodial: False` = Local-Trajectory Inference ("Narrow View Inference"); `episodial: True` = Joint-Episode ("Input Aggregation") — same reward-model class either way | **Live, primary approach.** DQN path works (has results for both episodial settings, e.g. `results/prm-dqn-episodial-Efficiency*Peace-fast-4_agents`); PPO path (`PPOPRM`) is broken as wired — see §3. |
| **NRP** | "Neural Reward Predictor" (backwards — actively misleading) | "no reward predictor" — baseline trained on ground-truth env reward | Baseline-Original / Baseline-Penalty — plain PPO/DQN on env reward | **Live baseline.** No reward predictor involved anywhere in this path. |
| **CRM** | "Counterfactual Reward Model" (doesn't match anything in the paper) | "collective reward model" — "could not make it work" | **Not a published MARP variant.** A separate centralized/global-state reward-model design (`env.state_space` + joint action vector) the paper doesn't describe | **Non-functional, and outside the paper's validated scope** — not "Joint-Episode but buggy." Zero `results/crm*` dirs ever produced; see §3 for confirmed code-level breaks. |

Also note: the reward predictor class for the **PRM** experiment is internally named
`RPMRewardPredictor` (`reward_predictor/prm/reward_model.py`) — RPM, not PRM, a real naming swap in
the code itself, not a typo in this documentation.

**Previously flagged as the single biggest open question — now resolved**: the paper reports its
Joint-Episode variant working and competitive with Local-Trajectory in every figure, which seemed to
conflict with "CRM never worked." It doesn't conflict once Joint-Episode is correctly identified as
`prm.yaml` with `episodial: True` (which has real, working results) rather than as CRM (which has
none). They were never the same method — CRM is a separate, likely earlier or abandoned experiment
that happens to share the superficial "operates on multiple agents jointly" framing but not the
paper's actual architecture. Worth a quick confirmation from the owner, but the code and paper
evidence now agree.

## 3. Why CRM (and PRM+PPO) don't actually run — consolidated bug list

CRM being outside the paper's validated scope (§2) explains why nobody would have caught these — it
was never the path being tuned to reproduce a paper figure. Four independent reviews each found a
different concrete break in the CRM path, and one separate break in PRM's PPO path. None of these
were cross-referenced against each other by the reviewing agents (each worked in isolation) —
collected here for the first time:

| # | Path | Bug | Where | Source doc |
|---|---|---|---|---|
| 1 | CRM | `env.env_method("state_space")` calls a `Box` **attribute** as if it were a callable method → `TypeError` | `runners.py:137`, only hit when `experiment == "crm"` | [environment.md §5](environment.md#5-pettingzoo-api-surface) |
| 2 | CRM | `CRMRunner.setup_agent` passes `async_rp_training`/`parallel_agents` kwargs to `IndependentRLRPLearner`, whose real (imported) constructor (`BaseLearner.__init__` in `learners.py`) doesn't accept them → `TypeError` | `runners.py` (`CRMRunner.setup_agent`) | [learners-and-agents.md §5](learners-and-agents.md#5-dead-code-determination-independent_rlrp_learnerpy--collective_rlrp_learnerpy) |
| 3 | CRM | Reward-model loss is **unweighted** `CrossEntropyLoss`, unlike PRM's **δ-weighted Bradley-Terry** (which matches the paper's Eq. 8 for both of the paper's real variants) | `reward_predictor/crm/reward_model.py` vs `prm/reward_model.py` | [reward-predictor.md §4](reward-predictor.md#4-reward-model-architecture) |
| 4 | CRM | Loss supervises the **mean** of `num_agent` output heads against one team-summed label — individual heads are under-constrained (many per-head solutions satisfy the same mean) | `crm/reward_model.py` | [reward-predictor.md §5](reward-predictor.md#5-correction-local-trajectory-vs-joint-episode-are-both-inside-prm--crm-is-a-third-separate-thing) |
| 5 | PRM (PPO only) | `runners.py` wires the PPO rollout-buffer slot to `PRMShardReplayBuffer`/`Episodial` (a `ReplayBuffer` subclass), but `PPOPRM.collect_rollouts` calls `.replay_store()`/`.reset()`/`.compute_returns_and_advantage()`, methods that only exist on the unused `PRMShardRolloutBuffer` (`RolloutBuffer` subclass) → `TypeError`/`AttributeError` | `runners.py:66-69`, `rl_agents/ppo/rp_agents.py` | [buffers.md §5](buffers.md#5-dead-code-determination) |

Bugs #1 and #2 alone are each independently sufficient to crash a `teach_crm.py` run before any
training happens, which is consistent with "never got it working" regardless of whether #3/#4 would
also have caused poor training. **PRM's live path is DQN-only** (`DQNPRM`/`IndependentDQNRP`, and
that includes both `episodial` settings — i.e. both Local-Trajectory and Joint-Episode go through
DQN in this codebase); the PPO variant of PRM (`prm_ppo.yaml`, `prm_single_ppo.yaml`) is plausibly
untested per bug #5 — worth confirming with the owner whether these configs were ever actually run
successfully before relying on them.

## 4. Dead code — consolidated deletion candidates

Full evidence and per-item verdicts in [dead-code-audit.md](dead-code-audit.md); environment- and
buffer-specific items cross-checked against [environment.md §6](environment.md#6-env-vs-env2-diff)
and [buffers.md §5](buffers.md#5-dead-code-determination).

**High confidence — safe to delete:**
- `src/env2/` (frozen after one 2025-03-23 commit; only referenced by `stam.ipynb`)
- `src/test.py`, `src/test2.py`, `src/stam.ipynb`, `src/test.mp4` (scratch scripts / notebook / its generated artifact)
- `src/learners/independent_rlrp_learner.py`, `src/learners/collective_rlrp_learner.py` (un-refactored duplicates of the classes actually exported from `learners.py`; the independent one has a live `self.loggers` bug that would `AttributeError` if ever called)
- `src/buffers/buffers.py` (superseded by `replay_buffers.py`; independently broken — references `PRMEpisodeData`/`Any` without importing either)

**Uncertain — needs owner input before deleting** (looks like an abandoned human-in-the-loop
labeling subsystem, never wired into any runner):
- `src/reward_predictor/segment_sampling.py`, `comparison_collectors.py`, `label_schedules.py`, `clip_manager.py` — zero call sites anywhere; `clip_manager.py`/`comparison_collectors.py` import a nonexistent `human_feedback_api` package
- The three dead exports in `reward_predictor/__init__.py`: `parallel_collect_segments`, `LabelAnnealer`, `function_wrapper`
- `src/env/utils/train_utils.py` and `src/env2/utils/train_utils.py` (identical, both reference nonexistent `DEFAULT_COLORMAP`/`DEFAULT_COLOURS` symbols and an uninstalled `DDQN`/`tensorflow` training loop; not imported by anything)

**Repo hygiene (not code):**
- 97 `__pycache__/*.pyc` files are tracked in git; `.gitignore` has no `__pycache__/`/`*.pyc` rule
- `.gitignore`'s `/vscode` line doesn't match the actual `.vscode/` directory (typo)
- `old_results/` uses a naming scheme incompatible with the current runner (predates the March 2025 "stable prm and baseline" commit) — stale, not reproducible as-is, not part of the code question

## 5. Two real, non-cosmetic bugs on the *live* path worth fixing regardless of refactor scope

Everything in §3 is on the already-abandoned CRM/PPO+PRM paths. These two are on paths that
currently produce results and are worth fixing even before a broader refactor:

- **`ascii_map` map selection uses `eval()`** against `commons_env.py`'s import namespace
  (`commons_env.py:35`) — an arbitrary-code-execution surface, and it silently contradicts
  `MapEnv`'s own docstring that says `ascii_map` should already be the parsed list of strings.
  ([environment.md §4](environment.md#4-map-system))
- **PPO's outer training loop never increments `total_steps`** in `learn_ppo`
  (`learners.py`) — the `while total_steps < total_timesteps` loop only exits via
  `collect_rollouts` returning `continue_training=False`; harmless in practice only because that
  eventually happens, but the stated step budget (`n_episodes * ep_length * ...`) is not actually
  enforced for any PPO-trained run. ([learners-and-agents.md §4](learners-and-agents.md#4-collectiverlrplearner--independentrlrplearner-orchestration))

## 6. Metrics quick reference (full equations in paper-and-configs.md §3)

| Metric | Meaning | Config value |
|---|---|---|
| Efficiency | Mean total reward across agents (raw productivity) | `metric: "Efficiency"` (default; dense per-step reward) |
| Equality | Gini-style fairness of reward distribution across agents | part of composite metrics, e.g. `Efficiency*Peace*Equality` |
| Sustainability | Mean timestep of first positive reward — later = more patient harvesting | `Efficiency*Sustainability` |
| Peace | Fraction of agent-timesteps *not* spent tagged/timed-out | `Efficiency*Peace`, `Efficiency*Peace2` |

Any `metric` other than `"Efficiency"` turns the task from dense per-step apple reward into a
**sparse, episode-terminal** reward equal to the chosen composite — this happens in
`env/pettingzoo_env.py`, outside the core env logic in `commons_env.py`/`map_env.py`.
