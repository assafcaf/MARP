# Refactor Plan

Written at the end of a design session covering: what's wrong with the current `src/`
(see the other docs in this folder), what a clean rewrite should look like, and — mid-session —
discovery that a fork, [almogze/marp](https://github.com/almogze/marp), already implements most of
the target architecture. This doc is the handoff: decisions made, what the fork provides, what's
missing, and the recommended next steps. Written so a fresh session with no memory of this
conversation can pick it up and start executing.

## 1. Decisions made this session

| Question | Decision |
|---|---|
| Refactor driver | Both: personal research velocity **and** making the code presentable if released alongside the paper. Roughly equal weight. |
| Execution style | Clean rewrite in a **new top-level directory in this repo** (not a new repo), copying over only what's needed. Current `src/` stays untouched as a working reference/fallback. |
| Variant scope | **Drop CRM entirely.** New codebase supports the NRP baseline + PRM's two real variants (Local-Trajectory, Joint-Episode) — see [reward-predictor.md §5](reward-predictor.md#5-correction-local-trajectory-vs-joint-episode-are-both-inside-prm--crm-is-a-third-separate-thing) for why CRM isn't one of the paper's published methods and never worked. |
| Config system | Started open to adopting Hydra (the common ML-lab standard). **Superseded** — see below, the fork already achieves composability with plain dataclasses + JSON, no Hydra needed. Hydra remains an option to layer on later if multirun/sweep needs grow beyond what `scripts/run_env.py`-style CLI handles. |
| RL backbone | Started as "keep Stable-Baselines3." **Reversed** after finding the fork: SB3's rigid internal training loop is *why* the original repo has the `DQNPRM`/`PPOPRM`/`DQNCRM`/`IndependentDQNRP` class-explosion (each needed a `collect_rollouts` override). The fork proves a plain `act()`/`observe()` external loop avoids this entirely. **Decision: drop SB3**, adopt the fork's custom DQN/IPPO/MAPPO. |
| Base to build from | **Adopt the `almogze/marp` fork as the literal starting point** for the new package, not just as a reference. Audit it for gaps (below), port forward what's missing from the current `src/`, then continue from there — not a from-scratch design. |
| Naming | Recommend naming things after the paper's real terms (`local_trajectory`, `joint_episode`, `baseline`) instead of PRM/NRP/CRM, which caused real confusion (see [paper-and-configs.md](paper-and-configs.md)). The fork already does this independently (`reward_model.mode: "narrow_view" | "input_aggregation"`) — good sign of independent convergence on the same conclusion. |

## 2. The fork: `almogze/marp`

Fetch with `git clone https://github.com/almogze/marp` (default branch `master`; HEAD at review time was
`7cff81c`, "Merge pull request #8 from almogze/feature/real-ippo-implementation"; other branches present:
`feature/adjust-nearby-apples`, `feature/mappo-implementation`, `feature/new-dqn-ppo-implementation`,
`feature/optimize-input-aggregation` — check whether these are already merged into `master` or contain
unmerged work worth pulling in before adopting).

### Structure

```
marp-fork/
├── src/env/            # ported from this repo's src/env/, since diverged (see §3)
├── src/reward_model/    # oracle.py, preference_buffer.py, reward_model.py, reward_trainer.py
├── src/train/           # config.py, trainer.py, registry.py, algorithms/{base,dqn,ippo,mappo,random_policy}.py,
│                         # logging_utils.py, metrics.py, video_utils.py
├── configs/              # JSON configs (train_dqn.json, train_ippo.json, train_mappo.json, ...)
├── scripts/              # run_env.py (CLI sweeps), plotting scripts, tensorboard helper
├── tests/                # test_metrics.py only
└── main.py               # entrypoint
```

### Why this solves the coupling problem you asked about

The core issue in the current `src/`: reward-source (none/PRM/CRM) × multi-agent strategy
(shared/independent) × RL algorithm (DQN/PPO) was expressed as a **combinatorial class hierarchy**
(`DQNPRM`, `IndependentDQNRP`, `DQNCRM`, `PPOPRM`, `CollectiveRLRPLearner`, `IndependentRLRPLearner`, ...),
each overriding SB3's `collect_rollouts`/`_store_transition` slightly differently.

The fork collapses this to two orthogonal axes with **no class explosion**:

- **`Algorithm(ABC)`** (`train/algorithms/base.py`) defines `on_env_ready` / `act` / `observe` /
  `on_episode_end` / `uses_external_loop`. `DQNAlgorithm`, `IPPOAlgorithm`, `MAPPOAlgorithm`,
  `RandomAlgorithm` each implement it once. Multi-agent strategy (independent Q-nets per agent for
  DQN/IPPO, centralized critic for MAPPO) is internal to each algorithm's own implementation, not a
  wrapper class layered on top.
- **`Trainer.train()`** (`train/trainer.py`) is a single external loop: step the env, and if a reward
  model is enabled, compute `pred_rewards` inline via `reward_model.predict(...)` and pass that dict
  to `algorithm.observe()` instead of env rewards — otherwise pass env rewards directly. **No
  subclassing needed for reward substitution at all.**
- **Verified directly** (not just inferred from the README): `IPPOAlgorithm.uses_external_loop()` and
  `MAPPOAlgorithm.uses_external_loop()` both return `True`, same as the base default — meaning **all
  three algorithms (DQN, IPPO, MAPPO) go through the identical `Trainer` loop**, including reward-model
  substitution. This was the one thing that needed direct code confirmation rather than assumption,
  since PPO-style algorithms often want their own rollout loop — here they don't need one.

### Reward model (`src/reward_model/`)

Cleanly split into four single-responsibility files, versus the original's one file doing everything:

| File | Responsibility |
|---|---|
| `oracle.py` | Pure functions: `compute_phi(metrics, phi_key)` (a registry of composable social-metric formulas — `efficiency`, `efficiency_x_peace`, `efficiency_x_peace_x_equality`, etc.) and `preference(phi_i, phi_j)` (Bradley-Terry label + delta magnitude) |
| `preference_buffer.py` | `PreferenceBuffer` — episode-level ring buffer (`EpisodeRecord`), with `aggregate_episode()` (all agents' trajectories concatenated — Joint-Episode) and `sample_agent_trajectory()` (one agent — Local-Trajectory) as buffer methods |
| `reward_model.py` | The network itself |
| `reward_trainer.py` | `RewardModelTrainer.train()` — δ-weighted Bradley-Terry BCE loss (confirmed: same `softmax(z-scored deltas)` weighting math as the original's PRM, just cleanly isolated), with AMP + chunked scoring for memory control |

`mode: "narrow_view" | "input_aggregation"` in `RewardModelConfig` selects which `preference_buffer`
method feeds the comparison — directly maps to Local-Trajectory / Joint-Episode. `phi` selects the
metric formula. No CRM-equivalent (global-state reward model) exists in the fork at all.

### Config (`src/train/config.py`)

Plain `@dataclass`es (`EnvConfig`, `DQNConfig`, `IPPOConfig`, `MAPPOConfig`, `AlgorithmConfig`,
`LoggingConfig`, `RewardModelConfig`, `TrainerConfig`) with `from_dict`/`load_config`/`save_config`
round-tripping through JSON. No Hydra. Composability for sweeps comes from `scripts/run_env.py`'s
CLI (repeatable flags like `--algo dqn ippo --map small large --seed 0 1 random` run all
combinations, or a JSON sequence file) rather than Hydra multirun — functionally similar, less
framework weight.

### Logging & tooling

`ResultLogger` writes `metrics.jsonl`, `config.json`, TensorBoard logs, and (if
`log_agent_episode_details`) per-agent step-level CSVs including `predicted_reward`,
`nearby_apples`, `ate_last_apple_in_cluster`. `VideoRecorder` captures episode videos at a
configurable cadence. Several `scripts/plot_*.py` produce publication-quality (300 DPI, serif fonts,
colorblind-safe palette) comparison plots directly from `metrics.jsonl` — materially more complete
than the original's ad hoc `SingleAgentCallback`/`stam.ipynb` combination.

### Already-fixed issues from the original repo

- **`eval()` map loading is gone.** `Trainer._build_env` resolves `MAP[env_cfg.map_type]` to an actual
  map object before constructing the env; `HarvestCommonsEnv` no longer calls `eval()` on it.
- **CRM is already absent** — independent confirmation that dropping it was the right call.
- **Reward-model terminology already paper-aligned** (`narrow_view`/`input_aggregation`, not PRM/CRM).

## 3. Gaps found — need to close before/during adoption

The fork was forked from an **earlier snapshot** of this repo's `src/env/` and has since diverged in
its own direction; it did not receive later refinements made on this repo's `src/env/` after the fork
point. Confirmed by direct diff (`diff src/env/*.py` between the two repos):

1. **Missing metric composites**: fork's `pettingzoo_env.py` only has `Efficiency`, `Efficiency*Peace`,
   `Efficiency*Peace*Equality` branches. This repo's current `src/env/pettingzoo_env.py` additionally
   has `Efficiency2`, `Efficiency*Peace2`, `Efficiency*Sustainability`. Port these forward if you want
   parity with configs like `prm_single.yaml`'s `Efficiency*Sustainability`.
2. **Missing `MEDIUM2_HARVEST_MAP`** in the fork's `maps.py`/`commons_env.py` `MAP` dict (used by
   `nrp.yaml`, `prm.yaml`).
3. **Missing per-episode-scaled timeout**: this repo's `commons_env.py` passes
   `timeout_time=int(ep_length*0.025)` into `HarvestCommonsAgent`; the fork's agents always use the
   class-level `TIMEOUT_TIME=25` regardless of episode length.
4. **`penalty` handling moved**: the fork applies the FIRE-action `-1` penalty directly inside
   `commons_env.py`'s own `step()` (a constructor `penalty` flag), whereas this repo applies it one
   layer up in `pettingzoo_env.py`'s wrapper. The fork's placement is arguably cleaner (penalty is
   core env behavior, not a wrapper-layer bolt-on) — worth confirming as a deliberate choice rather
   than an oversight, then keeping it consistent.
5. **Metric-substitution logic still duplicated, not unified.** The env's inline `if/elif` chain
   (`pettingzoo_env.py`) and `reward_model/oracle.py`'s `compute_phi` registry are two separate
   implementations of "what does metric X mean." Recommend unifying them — have the env's reward
   substitution call `oracle.compute_phi` too — now that both pieces will live in one package. This is
   the one remaining piece of the "env/runner coupling" complaint that the fork hasn't yet addressed.
6. **Test coverage is thin**: only `tests/test_metrics.py` exists (good unit tests for the
   nearby-apples/cluster-detection helpers, with mocks). No tests yet for `reward_model/`,
   `train/algorithms/`, `env/` core mechanics, or config round-tripping.
7. **Stale dependency**: `requirements.txt` still lists `stable-baselines3` despite zero code imports
   of it anywhere in the fork — drop it when adopting.
8. **No cross-validation against this repo's known-working results yet.** The fork hasn't been
   compared against `results/prm-dqn-*`/`results/nrp-dqn-*` (this repo's proven runs) to confirm the
   port preserves learning behavior, since its env diverged from an earlier snapshot (point 1-3 above).

## 4. Recommended sequence for the next session

1. Clone/vendor `almogze/marp`'s `src/` into this repo under a new top-level dir (e.g. `marp/`),
   preserving its internal structure. Leave current `src/` untouched.
2. Port forward the three env gaps (§3.1-3.3: extra metric composites, `MEDIUM2_HARVEST_MAP`,
   per-episode-scaled timeout) from this repo's current `src/env/` into the vendored copy. Confirm the
   `penalty`-placement choice (§3.4) and document it.
3. Unify metric-substitution (§3.5): route the env's reward-substitution through `oracle.compute_phi`
   instead of keeping two parallel metric-formula implementations.
4. Drop `stable-baselines3` from `requirements.txt`; audit for other stale deps.
5. Add test coverage: `reward_model/` (phi functions, preference-buffer sampling, trainer loss),
   `train/config.py` (load/save round-trip), and at least one end-to-end smoke test (env + one
   `Algorithm` + `Trainer` running a handful of episodes without crashing).
6. **Validation run**: one Local-Trajectory (`narrow_view`) DQN config and one baseline
   (`reward_model.enabled: false`) DQN config, compared qualitatively against this repo's
   `results/prm-dqn-*` and `results/nrp-dqn-*` — not necessarily bit-identical, but plausible/comparable
   learning curves, to build confidence the port didn't silently change behavior.
7. Only after 1-6: make the new package's entrypoint (`train.py` or similar) the repo's primary way to
   launch experiments; decide whether to archive or keep `src/`/`teach_*.py` as historical reference.

## Related documents

- [README.md](README.md) — index of the original `src/` codebase audit (entrypoints, dead code, bugs)
- [reward-predictor.md](reward-predictor.md) §5 — full derivation of why Local-Trajectory/Joint-Episode
  are both PRM variants and CRM is a separate, unpublished design
- [paper-and-configs.md](paper-and-configs.md) — paper↔code terminology, per the source paper
