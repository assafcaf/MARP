# MARP Refactor — Design Spec

**Date:** 2026-08-11
**Status:** Approved, ready for implementation planning
**Supersedes:** `docs/knowledge-base/refactor-plan.md` (several of its findings are corrected below)

---

## 1. Provenance — read this first

The single most important finding of the design session. The prior refactor plan was built on a
wrong assumption about which codebase produced the paper's results, and every downstream decision
inherited it.

| Artifact | Produced by | Evidence |
|---|---|---|
| Paper's **main results** (Figs 5, 6, 7, 9, 10, 11) | **`almogze/marp` fork**, master `7cff81c` | 19/19 hyperparameters in Table 1+2 match `configs/train_ippo.json` + `RewardModelConfig` exactly; Fig 7's four line colors match `PHI_COMPARISON_COLORS` by exact hex, in order; Fig 5's three conditions are literally `no_apple_eaten` / `nearby_apples == 0` / `nearby_apples >= 4` from `scripts/process_all_sessions.py:405-420`; `IPPO_SESSIONS` (`:114`) is a committed registry of 8 conditions × 5 runs, `agents=5`, matching "standard error over five runs" |
| Paper's **Figure 4** (Perolat-reproduction DQN) | **`DanfoaTestSOT`** | All six legend entries map to `results/` directories with 8001 rows = 8000 episodes, `"learner": "dqn"`, matching `prm.yaml`/`nrp.yaml` `n_episodes: 8_000` |

Confirmed by the repo owner. Confidence ~95% on both, with one gap: **no fork run artifacts exist on
this machine** — zero `metrics.jsonl` files anywhere on the filesystem. The 40 runs live on the fork
author's machine. Figure 4's attribution, by contrast, is backed by on-disk artifacts.

**Consequences:**

- The fork is the **primary artifact**, not a cleaner reimplementation to be validated against SOT.
- SOT is a **superseded DQN-era codebase**, relevant as a source of RL engineering practice and as
  Figure 4's origin — *not* as a validation target. Any plan to reproduce `results/` numerically is
  validating the real artifact against an obsolete one.
- `DanfoaTestSOT` is **read-only** for the entire refactor. Nothing writes there.

---

## 2. Goals and non-goals

**Goals**

1. A clean, presentable codebase releasable alongside the paper.
2. Environment semantics that are explicit and selectable rather than accidental — including a
   preset that is faithful to Perolat et al. 2017, which neither current codebase implements.
3. RL engineering that reflects deliberate choices: fix outright defects, adopt what SOT does better,
   and modernize the reward model.
4. A documented, ordered experiment schedule so the work can be validated later.

**Non-goals**

- **No training runs during this refactor.** Verification is tests only. Everything requiring GPU
  time is deferred to `docs/experiment-schedule.md`.
- **The NRP/PRM architecture confound is out of scope** (see §7). Recorded in the knowledge base,
  handled separately by the owner.
- No attempt to reproduce SOT's `results/` numerically.
- CRM remains dropped, per the original plan.

---

## 3. Decisions

| Question | Decision |
|---|---|
| Repo layout | Fresh `git init` in `/home/assaf_caftory/CommonsGame/DanfoaTest`, history starting at commit 0. `DanfoaTestSOT` is the read-only reference. Both currently point at the same remote, so the old history survives in SOT and on `origin`. |
| Commit zero | Fork master `7cff81c` (`src/`, `configs/`, `scripts/`, `tests/`, `main.py`, `requirements.txt`) + `docs/knowledge-base/` + this spec. Attribution to `almogze/marp` in the commit message and README. |
| Fork branches | Adopt `master` as-is. 7 of 8 branches are already merged; the 8th (`feature/new-dqn-ppo-implementation`) is a stale side-line that deletes `ippo.py`, `tests/`, and the metric feature. Do **not** pull it. |
| Env fidelity | Preset registry (§4.1), not a binary flag. `legacy` = Perolat 2017 1:1 and is the default; `marp_paper` = the fork's env as it produced Figs 5–11. |
| Preset scope | Presets pin `ep_length` and `num_agents`, overridable by an explicit config value. |
| Metric wiring | Episode-terminal substitution moves into `Trainer` and routes through `oracle.compute_phi`. The dead `pettingzoo_env.py` wrapper is deleted. One metric implementation in the codebase. |
| RL scope | Tier 1 (defects) + Tier 2 (SOT's sample efficiency) + reward-model ensemble. |
| Validation | Deferred entirely to `docs/experiment-schedule.md`. |
| Governance | Master agent scopes and reviews; one fresh subagent per phase; separate review agent for phases 3, 5, 6; `--no-ff` merges. One branch open at a time. |

---

## 4. Architecture

### 4.1 Environment spec registry

The env's behavior is currently implicit — spread across hardcoded constants, a squared-distance
bug, and a config field that does nothing. It becomes an explicit, named, selectable spec.

```python
# src/env/specs.py
@dataclass(frozen=True)
class EnvSpec:
    respawn_neighborhood: str   # "ball_r2" (13 cells) | "block_3x3" (9 cells)
    respawn_buckets: tuple      # ((1,0.01),(3,0.05),(5,0.1)) | collapsed min(L,3) form
    beam_length: int | str      # 10 | "view_range"
    beam_width: int             # 5 | 3
    obs_window: str             # "forward_biased_20x21" | "square"
    colormap: str               # "same" | "different"
    self_marker: bool
    timeout: int = 25           # paper-correct in both codebases
    ep_length: int = ...        # pinned, overridable
    num_agents: int = ...       # pinned, overridable

ENV_SPECS = {
    "legacy":     EnvSpec(...),  # Perolat 2017, 1:1. Default.
    "marp_paper": EnvSpec(...),  # the fork's env behind Figs 5-11
}
```

`EnvConfig` gains `spec: str`, resolved at env construction. This mirrors the `compute_phi` registry
already in `oracle.py`, giving the codebase one idiom for "named, swappable definition."

Rationale for `legacy` meaning *Perolat*: this project is a continuation of that line, so the
canonical definition is the baseline and the project's deviations are named and opt-in rather than
buried in an `APPLE_RADIUS` squared-distance bug.

**Paper values for `legacy`** (Perolat et al. 2017, arXiv:1707.06600, App. A unless noted):

- Respawn: `L = |{apples ∈ B₂(c)}|`, `p = 0` (L=0), `0.01` (L∈{1,2}), `0.05` (L∈{3,4}), `0.1` (L>4)
- Beam: width 5; length 10 (Fig. 3 caption — the experiment that produced the published figures).
  Appendix A says 20; the paper is self-contradictory. Use 10 and document the conflict.
- Observation: 20 squares ahead × 21 wide, egocentric, orientation-normalized
- Actions: 8 (fwd/back/left/right, rotate L/R, tag, stand still)
- Tagging: one hit; 25-timestep timeout; **no direct reward or penalty** for tagging or being tagged
- Episode 1000 steps; 12 agents
- Reward: dense per-apple (+1 inferred — the paper never states the numeric value)

**Known deviations both codebases share, corrected only under `legacy`:**

1. `j²+k² <= APPLE_RADIUS` with `APPLE_RADIUS=2` treats 2 as a *squared* distance → 3×3 block of 9
   cells, not `B₂`'s 13. Inherited from the open-source `sequential_social_dilemmas` reimplementation.
2. `spawn_speed[min(L,3)]` collapses the paper's 4-way bucketing → regrowth saturates far earlier.
3. Beam width 3, not 5.
4. Square observation window, not forward-biased.

Items 1 and 2 compound: the common-pool resource is materially more forgiving than the one the paper
models, which affects the tragedy/recovery dynamic these experiments are about.

**Metric scale note:** both codebases normalize Sustainability by `ep_length` and Peace by
`N·ep_length`, so their values are monotone-equivalent to the paper's but not directly comparable to
its plotted magnitudes. Document the reporting scale explicitly.

### 4.2 Metric substitution

`pettingzoo_env.py` is **dead code in the fork** — `Trainer._build_env` constructs
`HarvestCommonsEnv` directly and never calls `parallel_env`. `config.env.metric` is stored and never
read, so setting `"metric": "Efficiency*Peace"` currently does nothing.

Fix: delete the wrapper; put episode-terminal substitution in `Trainer.train()` — the single external
loop all four algorithms already share — computing φ through `oracle.compute_phi`. This collapses two
problems (dead metric path, duplicated metric formulas) into one change.

Unification is low-risk: both paths consume the same `get_social_metrics()` dict, both produce a
scalar float, and both fire once per episode after `compute_social_metrics()`. The only deltas are
naming (`Efficiency*Peace` vs `efficiency_x_peace`) and two non-linear variants (`peace**4`,
`sustainability*2`) needing new registry entries.

**Framing:** composite-metric-as-reward has **no basis in Perolat et al.** — §2.3 defines U/E/S/P as
post-hoc analysis instruments only. It is this project's contribution, layered on the Perolat env, and
the README must say so. It is not a deviation to be "fixed."

### 4.3 RL quality

**Tier 1 — defects** (each traced to a specific line, all found by audit):

- IPPO's `orthogonal_init` applies gain 0.01 to *every* `Linear` and `Conv` in the actor including the
  conv trunk, versus SB3's layer-wise √2 (trunk) / 0.01 (policy head) / 1.0 (value head). Near-dead
  actor at init.
- MAPPO has no orthogonal init at all.
- Adam `eps` 1e-8 vs SB3's 1e-5 for the PPO family (`ActorCriticPolicy` injects it).
- No gradient clipping in DQN; SB3 silently applies `max_grad_norm=10`.
- **The `-1` fire penalty is applied before `update_social_metrics`, corrupting U/E/S.**
- Two penalty sites exist (`commons_env.py` and the dead wrapper); de-duplicate.
- `Trainer._format_reward_obs` reads `config.algorithm.dqn.normalize_obs` regardless of which
  algorithm is running — latent, currently benign only because of default values.
- SOT-inherited: per-instance mutation of the module-global `ACTIONS` dict (breaks with concurrent
  envs). Applies to whichever preset ties beam length to view range.

**Tier 2 — sample efficiency, adopted from SOT:**

- Optional parameter sharing + pooled experience buffer. SOT pools 4 envs × 4 agents into one network
  and one 6.4M-transition buffer; the fork gives each agent its own network and a 5,000-transition
  buffer. For homogeneous agents this is the single largest efficiency difference between the two.
- Vectorized environments (SOT runs 4; the fork runs 1, sequential).
- Frame stacking (SOT uses `num_frames: 2`; the fork stacks nothing, so no motion information is
  observable).

All three are **opt-in**, defaulting to the fork's current behavior, so the paper's configuration
remains expressible unchanged.

**Tier 3 — reward model:** ensemble with disagreement-based uncertainty, following current
preference-based RL practice (PEBBLE, SURF, B-Pref). This strengthens the project's actual
contribution rather than the surrounding scaffolding.

### 4.4 Deliberate divergences from SOT

Keep the fork's behavior, and document why:

- **Time-limit handling.** The fork treats the horizon as truncation and bootstraps; SOT treats it as
  a hard terminal. SOT's behavior is an artifact of nothing ever setting `TimeLimit.truncated` in the
  supersuit chain, not a design choice. The fork is textbook-correct.
- **Seeding.** SOT is genuinely unreproducible — no seed ever reaches SB3, and `MapEnv.reset` accepts
  `seed` and ignores it. The fork seeds `random`/`numpy`/`torch`/`cuda`.
- **φ read directly from metrics** rather than smuggled through the reward channel by zeroing all
  per-step rewards.

**Verified as already correct, not to be "fixed":** PRM reward routing is semantically identical in
both codebases (predicted reward from pre-step obs + taken action; env reward retained only for
preference labelling). Neither has the bootstrap-from-reset-observation bug. `/255` normalization,
Huber loss, and vanilla-not-Double DQN all match.

---

## 5. Phases

One branch each, merged `--no-ff`. No training runs in any phase.

| # | Branch | Contents | Verification |
|---|---|---|---|
| 0 | `main` | Seed fork `7cff81c` + `docs/knowledge-base/` + this spec; README attributing almogze/marp | imports resolve |
| 1 | `chore/hygiene` | Delete broken `env/utils/train_utils.py` (ImportError on import, zero call sites) and the dead `Algorithm.train` branch (signature mismatch, unreachable); fix `main.py`; drop `stable-baselines3` and `tqdm`; pin `gymnasium`/`pettingzoo`; add `numpy`/`pytest`; package markers; unify import style | all modules import; `compileall` clean |
| 2 | `fix/correctness` | Tier 1 defects (§4.3) | unit test per fix; explicit test that the penalty no longer perturbs social metrics |
| 3 | `feat/env-spec-registry` | `EnvSpec` + `ENV_SPECS`; `legacy` and `marp_paper`; add `MEDIUM2_HARVEST_MAP` + `ARIGINALHARVEST_MAP_LARGER`; delete dead `timeout_time` plumbing, keep 25 | conformance test asserting every paper-specified value for `legacy` |
| 4 | `feat/metric-oracle-wiring` | Delete `pettingzoo_env.py`; wire terminal substitution into `Trainer` via `compute_phi`; add missing composites to the oracle | golden-value test vs the legacy `if/elif`, all 6 metrics numerically identical |
| 5 | `feat/sample-efficiency` | Tier 2: opt-in parameter sharing + pooled buffer, vectorized envs, frame stacking | shape/semantics unit tests |
| 6 | `feat/reward-model-ensemble` | Tier 3: ensemble + disagreement uncertainty | unit tests on ensemble and uncertainty |
| 7 | `feat/config-hardening` | Clear errors on unknown config keys (`from_dict` is `Cls(**data)`, so a renamed key raises a raw `TypeError`); sequence files distinguishable from `TrainerConfig` | round-trip + malformed-config tests |
| 8 | `test/coverage` | Tests for φ, preference-buffer sampling, δ-weighted BT loss, env core mechanics, config round-trip | tests pass |
| 9 | `docs/deliverables` | README + `docs/experiment-schedule.md` | — |

Phases 3, 5, and 6 each get a separate review agent, so author ≠ reviewer.

### Sequencing rationale

This order inverts the original refactor plan on one point. That plan had "port the missing metric
composites into the env" as an early step and "unify the two metric implementations" as a later one,
which would write those composites twice — once into an `if/elif` chain, then again into the oracle
after deleting the chain. Here, unification (phase 4) comes first and the missing composites are
written **once**, into the oracle registry.

Otherwise: hygiene precedes correctness so fixes land on a codebase that imports cleanly, and the env
spec registry precedes metric wiring because the latter reads env-spec-dependent values.

---

## 6. Risk: phases 5 and 6 ship unvalidated

By construction, since the refactor performs no runs. Both phases change learning behavior, and
tests confirm only that the code does what it claims — not that it learns better, or as well.

Mitigations:

1. Every Tier 2 feature is **opt-in and defaults off**, so the paper's configuration is bit-for-bit
   expressible after the refactor.
2. `docs/experiment-schedule.md` is a first-class deliverable specifying which runs confirm which
   phase, in what order, with what comparison. It is what makes phases 5–6 trustworthy later.
3. Until those runs happen, the README cites the paper and the almogze results as the reference
   numbers, and does not claim the refactored code reproduces them.

---

## 7. Out of scope — recorded, not actioned

**NRP/PRM architecture confound (SOT, Figure 4 only).** `BaseRunner.setup_agent` builds
`policy_kwargs` at `runners.py:151-155` but the argument is commented out at `:158`, and `NRPRunner`
does not override it. `PRMRunner.setup_agent` (`:342-352`) passes it. Confirmed effect:

| Arm | Extractor | q_net params |
|---|---|---|
| NRP baseline | `FlattenExtractor` (no convolutions) | 91,144 |
| PRM | `CnnFeatureExtractor` | 380,240 |

The comment-out dates to commit `8652c81` (2025-03-23); all 4-agent runs are May 2025. Positive
control: a Jan-2025 NRP run in `old_results/` *does* contain `q_net.features_extractor.cnn.0.weight
(8,6,5,5)`. So the baseline used a CNN and silently stopped.

Owner's decision: out of scope for the refactor, recorded in the knowledge base, handled separately.

**Also recorded, not actioned:**

- Several 4-agent PRM runs truncated at 25–37% of their configured budget — all superseded
  `ep_length: 600` configs. Every run of the final 51.2M-step configuration completed, except two
  May-14 runs that died at startup (cause UNCONFIRMED; no stdout or OOM records survive).
- The `psutil` >95% memory guard calls bare `exit()`, but lives in `DQNCRM.learn` and
  `IndependentDQN.learn` — **not** on the PRM path, which has no such guard.

---

## 8. Open questions

1. **`marp_paper` preset fidelity.** Defined from the fork's committed configs and code, since no run
   artifacts exist locally. If the 40 `metrics.jsonl` files are ever obtained from the fork author,
   the preset should be checked against a per-run `config.json`. The fork's `train_ippo.json` says
   `episodes: 200` while the figures show 250 — per-run counts live in those missing files.
2. **Apple reward value.** +1 in both codebases; the paper never states it. Inference, kept.
3. **`B₂` metric.** Euclidean/Manhattan (13 cells) vs Chebyshev (25). The paper does not say.
   Recommend Euclidean/Manhattan as the most natural reading of "ball of radius 2"; document the choice.
4. **Beam length under `legacy`.** 10 (Fig. 3) vs 20 (App. A). Recommend 10; a length-20 beam spans
   most of a 26–53-column map, making exclusion strategies trivially available.
