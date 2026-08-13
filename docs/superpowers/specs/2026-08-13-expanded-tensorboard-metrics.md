# Expanded TensorBoard Metrics — Spec

**Status:** approved for implementation
**Date:** 2026-08-13
**Reference implementation:** `../../../../DanfoaTestSOT` (`CommonsGameSOT`)

## Problem

The TensorBoard output of this repo is a strict subset of what the earlier
`DanfoaTestSOT` runs produced. Reading a run here, you can see *that* reward went
up; you cannot see *how* the agents got there, and you cannot see whether the
learned reward model is separating good behaviour from bad.

Tags the reference run has and this repo does not:

| Reference tag | Meaning |
|---|---|
| `on_action/{move left,move right,move up,move down,stay,turn,fire}` | mean **predicted** reward conditioned on the action taken |
| `outcome_avg/{apple_eaten,no_apple_eaten,delta}` | mean predicted reward on harvest vs non-harvest steps, and their gap |
| `outcome_std/{apple_eaten,no_apple_eaten,delta}` | same, standard deviations |
| `predicter rewards by appeals in porximity/{0,1,2,3+}` | mean predicted reward on harvest steps, bucketed by apples in proximity |
| `predictor/correlations` | step-level Pearson correlation between predicted and true reward |
| `time/fps` | throughput |

Tags this repo already has and keeps: `social/{efficiency,equality,sustainability,peace,fire_attempts,fire_sucsses}`,
`train/{reward_sum,reward_mean,reward_pred_sum,reward_pred_mean,steps,loss}`,
`reward/agent_*`, `reward_pred/agent_*`, `reward_model/*`, `algo/*`.

Two further problems exist independently of the reference:

1. `nearby_apples` and `ate_last_apple_in_cluster` are already computed
   (`train/metrics.py`) but land only in a per-agent CSV, for environment 0
   only, and only when `logging.log_agent_episode_details` is on. They never
   reach TensorBoard.
2. With `env.penalty: true`, `train/reward_mean` is the *penalised* reward
   (-1 per FIRE), not the harvest reward. The true environment reward is
   carried on `infos[agent]['r']` and is currently discarded.
   `ResultLogger._log_tensorboard` already handles `reward_env_*` payload keys —
   nothing ever produces them.

## Goal

Every number a person needs to diagnose a commons-harvest run appears in
TensorBoard, at one scalar per logged iteration, without changing existing tag
names or the shape of `metrics.jsonl` records that already exist.

## Non-goals

- Changing the training algorithms, the reward-model objective, or the env
  dynamics. This is instrumentation only.
- Per-step TensorBoard resolution. Everything is aggregated to one point per
  logged iteration, on the existing `episode` x-axis.
- Reproducing the reference's typos (`fire_sucsses` is kept, because it is an
  existing tag in *this* repo and renaming it would break comparisons; the new
  tags use correct spelling).

## Metric catalog

Aggregation is over every row of the iteration — all `num_envs` environments and
all `num_agents` agents, `ep_length` steps each — unless stated otherwise.
`num_rows = num_envs * num_agents`.

### 1. `social/*` — env-computed, per-episode (extended)

Existing: `efficiency`, `equality`, `sustainability`, `peace`, `fire_attempts`,
`fire_sucsses`. Added, computed inside `HarvestCommonsEnv.compute_social_metrics`
and therefore also stored on `EpisodeRecord.metrics`:

| Key | Definition |
|---|---|
| `fire_hit_rate` | `fire_sucsses / fire_attempts`, 0 when no attempts |
| `apples_eaten` | count of positive true-reward events in the episode |
| `apples_spawned` | apples added by `spawn_apples` over the episode |
| `apple_stock_mean` | mean apples present on the map, over steps |
| `apple_stock_min` | minimum apples present on the map |
| `apple_stock_final` | apples on the map at the last step |
| `depletion_fraction` | fraction of steps with zero apples on the map |
| `timeout_steps` | agent-steps spent in timeout (the numerator `peace` hides) |
| `reward_min_agent` / `reward_max_agent` / `reward_std_agent` | spread of per-agent episode returns |

These are additive keys. `compute_phi` reads named keys only, so preference
labels are unchanged bit-for-bit.

### 2. `action/*` — behaviour distribution

Fraction of steps each action was taken, over all rows:
`move_left`, `move_right`, `move_up`, `move_down`, `stay`, `turn`, `fire`.
`turn` merges actions 5 and 6, matching the reference's `on_action/turn`.
Plus `action/entropy`: Shannon entropy (nats) of the empirical action
distribution — a policy collapse shows here before it shows in reward.

### 3. `harvest/*` — resource-use behaviour

| Tag | Definition |
|---|---|
| `harvest/apples_per_agent` | apples eaten / `num_agents` (equals `social/efficiency` when unpenalised; kept explicit) |
| `harvest/harvest_rate` | apples eaten / agent-steps |
| `harvest/nearby_apples_mean` | mean apples within `nearby_apple_radius` of an agent, over all steps |
| `harvest/nearby_apples_on_harvest` | same, restricted to steps where the agent ate an apple |
| `harvest/last_in_cluster_rate` | fraction of harvests that emptied the cluster — the over-harvesting signal |

`nearby_apples` is the same quantity the reference calls `aip` (apples in
proximity). It is measured **after** the step, at the agent's new position, so
for a harvest step it counts what was left around the apple just taken.

### 4. `reward_env/*` — unpenalised environment reward

`reward_env/sum`, `reward_env/mean`, `reward_env/agent_<id>`, read from
`infos[agent]['r']`. Identical to `train/reward_*` when `env.penalty` is false;
the pair is what makes a penalised run readable.

### 5. `rm_*` — reward-model step diagnostics

Only emitted when `reward_model.enabled`. All are computed from the predicted
rewards the trainer already produces for every row of every step, paired with
the true environment reward for the same step — no extra forward passes.

| Tag family | Definition |
|---|---|
| `rm_on_action/{move_left,…,turn,fire}` | mean predicted reward per action (reference `on_action/*`) |
| `rm_outcome_avg/{apple_eaten,no_apple_eaten,delta}` | mean predicted reward on harvest / non-harvest steps and the gap (reference `outcome_avg/*`) |
| `rm_outcome_std/{apple_eaten,no_apple_eaten,delta}` | same for standard deviation (reference `outcome_std/*`) |
| `rm_outcome/separation` | `delta / pooled_std` — a d-prime; scale-free, so it is comparable across runs where the reference `delta` is not |
| `rm_by_nearby_apples/{0,1-2,3-4,5+}` | mean predicted reward on harvest steps bucketed by nearby apples (reference `predicter rewards by appeals in porximity/*`) |
| `rm_pred/{mean,std,min,max}` | distribution of predicted reward over all steps |
| `rm_pred/step_corr` | Pearson correlation of predicted vs true reward across all steps (reference `predictor/correlations`) |

Buckets are `0`, `1-2`, `3-4`, `5+`. The reference's buckets overlap
(`1` is `aip in {1,2}` and `2` is `aip in {2,3}`, so `aip == 2` is counted
twice); ours partition.

Empty subsets emit no scalar rather than a NaN — a missing point in TensorBoard
reads as "did not happen", a NaN point poisons the axis.

### 6. `time/*` — throughput

`time/fps` (env-agent steps per second for the iteration), `time/iteration_sec`,
`time/elapsed_hours`.

### 7. `algo/*` — per-agent expansion

Existing behaviour logs only scalar entries of the algorithm's metric dict.
Nested `{agent_id: float}` entries (e.g. IPPO's `ent_coef_per_agent`) are
flattened to `algo/<name>/<agent_id>` instead of being dropped.

## Configuration

New keys on `LoggingConfig`:

| Key | Default | Meaning |
|---|---|---|
| `detailed_metrics` | `true` | Master switch for sections 2, 3, 5. Off restores the previous tag set exactly. |
| `nearby_apple_radius` | `2` | Radius for `nearby_apples` / the `rm_by_nearby_apples` buckets. |
| `histogram_every_n_episodes` | `0` | `> 0` also writes `rm_pred/hist` and `reward/agent_hist` histograms at that interval. `0` disables. |

Nothing else changes. Existing configs stay valid and keep their behaviour, with
new tags added.

## Cost

Per step the accumulator does O(`num_rows`) numpy work plus one small window
slice per row for the nearby-apple count. Measured budget: the extra work must
stay under 5% of iteration wall time at `num_agents=5, num_envs=4,
ep_length=600`. The existing `count_nearby_apples` is a Python double loop over
the whole agent view (225 iterations per call); it is replaced by a masked
window slice, which is why running it for every row rather than only env 0 is
affordable.

## Correctness requirements

1. Existing tag names, and the meaning behind them, do not change.
2. `metrics.jsonl` gains keys; no existing key changes type or meaning.
3. `EpisodeRecord.metrics` gains keys; `compute_phi` output is unchanged for
   every supported `phi_key`.
4. With `logging.detailed_metrics: false` the emitted tag set equals the current
   one plus `social/*` additions plus `time/*`.
5. Every new statistic is defined over an explicitly empty-safe subset: no NaN
   is ever written to TensorBoard.
6. The nearby-apple count must equal the existing `count_nearby_apples` result
   for every agent position and view, including agents in timeout (which sit at
   `OUTCAST_POSITION` and must count 0).
