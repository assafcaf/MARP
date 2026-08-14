# TensorBoard metric catalog

Launch with `uv run commons-game tensorboard --logdir logs`.

Everything is one point per logged iteration, on an x-axis of **episodes** (not
iterations), so runs with different `env.num_envs` overlay directly.

Unless noted, statistics are aggregated over every agent of every environment
for the whole iteration — `num_envs * num_agents * ep_length` samples.

**Absent points are meaningful.** A statistic conditioned on an event that did
not happen (a harvest with five apples nearby, in an iteration where no such
harvest occurred) is omitted rather than written as zero or NaN. A gap in a
series means "did not happen", not "was zero".

---

## `social/*` — per-episode outcomes

Computed by `HarvestCommonsEnv.compute_social_metrics`, averaged across the
environments of the iteration. These are also what the reward model's oracle
scores episodes on, via `reward_model.phi`.

| Tag | Meaning | Read it for |
|---|---|---|
| `efficiency` | total harvest / `num_agents` | Did they eat? |
| `equality` | 1 − Gini of per-agent returns | Was it shared? |
| `sustainability` | mean timestep at which reward arrived, normalised | Did the harvest last, or was it front-loaded? |
| `peace` | fraction of agent-steps *not* spent in timeout | How much conflict? |
| `fire_attempts` | FIRE actions taken | |
| `fire_sucsses` | FIRE actions that actually timed an agent out (spelling kept for continuity with existing runs) | |
| `fire_hit_rate` | `fire_sucsses / fire_attempts` | Aimed fire, or spray? |
| `apples_eaten` | harvests in the episode | |
| `apples_spawned` | apples regrown in the episode | |
| `apple_stock_mean` | mean apples standing on the map | The single best sustainability read. Falling toward zero = the commons is being stripped. |
| `apple_stock_min` | fewest apples standing at any step | |
| `apple_stock_final` | apples standing at the last step | Did they leave anything behind? |
| `depletion_fraction` | fraction of steps with an empty map | > 0 means total collapse happened |
| `timeout_steps` | agent-steps spent in timeout | The raw count `peace` normalises |
| `reward_min_agent` / `reward_max_agent` / `reward_std_agent` | spread of per-agent returns | Who went hungry. Moves when one agent monopolises a patch while the Gini still looks respectable. |

## `train/*` and `reward*/*` — returns

| Tag | Meaning |
|---|---|
| `train/reward_sum` / `train/reward_mean` | return the **policy** received, per episode |
| `train/reward_env_sum` / `train/reward_env_mean` | **unpenalised** environment return |
| `train/reward_pred_sum` / `train/reward_pred_mean` | return under the learned reward model |
| `train/steps`, `train/loss` | |
| `reward/agent_<id>` | per-agent policy return |
| `reward_env/agent_<id>` | per-agent unpenalised return |
| `reward_pred/agent_<id>` | per-agent predicted return |

With `env.penalty: false` the `reward` and `reward_env` families are identical.
With it on they diverge, and `reward_env` is the one that answers "how many
apples did they get" — `train/reward_mean` includes the −1 per FIRE.

With `reward_model.enabled: true` the policies optimise `reward_pred`; the
`reward_env` family is then the ground truth the run is actually judged on.

## `action/*` — what the policy does

Fraction of steps spent on each action, summing to 1 across the seven groups.
`turn` merges TURN_CLOCKWISE and TURN_COUNTERCLOCKWISE.

`action/entropy` is the Shannon entropy (nats) of the **raw** action
distribution, so it is directly comparable with `algo/entropy` and with
`algo/target_entropy` when `ent_coef_mode: adaptive`. `ln(8) ≈ 2.08` is uniform;
a value collapsing toward 0 is a policy that has stopped exploring, and it moves
well before reward does.

## `harvest/*` — what the policy does to the commons

| Tag | Meaning |
|---|---|
| `apples_per_agent` | harvests per agent per episode |
| `harvest_rate` | harvests per agent-step |
| `nearby_apples_mean` | mean apples within `logging.nearby_apple_radius` of an agent, over all steps — how much abundance the agents live in |
| `nearby_apples_on_harvest` | the same, restricted to harvest steps — how dense a patch they eat from |
| `last_in_cluster_rate` | fraction of harvests that took the last apple in its cluster |

`last_in_cluster_rate` is the over-harvesting signal. An apple with no surviving
neighbours cannot seed regrowth (`spawn_apples` scales spawn probability with
neighbour count), so a rate near 1 is a policy eating its own future. It usually
rises *before* `social/apple_stock_mean` falls.

## `rm_*` — reward model, step level

Only when `reward_model.enabled`. Computed from the predicted rewards the
trainer already produces for every step; no extra forward passes.

`reward_model/*` (below) is the pair-level view — loss, preference accuracy.
These are the step-level view, and the two can disagree: a model can post a
falling loss while the per-step reward the policy actually optimises stays flat.

| Tag | Meaning |
|---|---|
| `rm_on_action/<action>` | mean predicted reward conditioned on the action taken |
| `rm_outcome_avg/apple_eaten` | mean predicted reward on harvest steps |
| `rm_outcome_avg/no_apple_eaten` | mean predicted reward on every other step |
| `rm_outcome_avg/delta` | the gap between them |
| `rm_outcome_std/*` | the same three, for standard deviation |
| `rm_outcome/separation` | `delta / pooled_std` — a d-prime |
| `rm_by_nearby_apples/{0,1-2,3-4,5+}` | mean predicted reward on harvest steps, bucketed by apples in proximity |
| `rm_pred/{mean,std,min,max}` | distribution of predicted reward |
| `rm_pred/step_corr` | Pearson correlation of predicted vs true reward, across all steps |

**How to read them.** `rm_outcome/separation` is the headline: it says how many
pooled standard deviations separate a harvest from a non-harvest in the model's
eyes. Prefer it to `rm_outcome_avg/delta`, which moves with the model's
arbitrary output scale and so cannot be compared across runs.

`rm_on_action/fire` against the other actions tells you whether the model has
learned to discourage conflict, which is the point of a `*_x_peace` phi.

`rm_by_nearby_apples/*` asks whether the model distinguishes an apple taken
from a dense patch from the last one in a thin patch — the difference between a
reward model that encodes sustainability and one that just counts apples. The
buckets partition (unlike the reference implementation's, where `aip == 2` fell
in two buckets at once).

## `reward_model/*` — reward model, pair level

From `RewardModelTrainer.train`: `loss`, `pref_accuracy`, `effective_pairs`,
`tie_fraction`, `grad_norm`, `grad_overflow_rate`, `score_phi_corr`.

`effective_pairs` (`1 / Σw²`) is the one to watch: it is how many of
`reward_model.batch_pairs` the delta-weighted loss really used. Far below
`batch_pairs` means the weighting has collapsed onto a few outlier pairs — raise
`reward_model.delta_temperature`.

## `algo/*` — algorithm internals

Whatever the selected algorithm's `on_episode_end` returns. Nested per-agent
dicts are flattened to `algo/<name>/<agent_id>`, so IPPO's `ent_coef_per_agent`
and `entropy_per_agent` are visible per agent — a single averaged coefficient
hides one agent's controller running away.

## `time/*` — throughput

`fps` (agent-steps per second), `iteration_sec`, `elapsed_hours`.

## Histograms

Off by default. Set `logging.histogram_every_n_episodes` above 0 for
`reward/agent_hist` (per-agent returns — the distribution behind `equality`) and
`rm_pred/hist` (every predicted reward in the iteration). They cost orders of
magnitude more disk than a scalar.

---

## Configuration

| Key | Default | Effect |
|---|---|---|
| `logging.detailed_metrics` | `true` | Off removes `action/*`, `harvest/*` and `rm_*`. Everything else is unaffected. |
| `logging.nearby_apple_radius` | `2` | Radius for `nearby_apples` and the `rm_by_nearby_apples` buckets. 2 is `APPLE_RADIUS`, the neighbourhood that drives regrowth. |
| `logging.histogram_every_n_episodes` | `0` | Episodes between histograms; 0 disables. |
| `logging.log_agent_episode_details` | `true` | Per-step CSVs under `extended_info/`, environment 0 only. |

Measured cost of `detailed_metrics` at 5 agents × 4 environments × 600 steps:
**0.4%** of iteration wall time.

## Beyond TensorBoard

- `metrics.jsonl` — one JSON record per logged iteration, carrying everything
  above plus a `sections` object holding the `action`/`harvest`/`rm_*` families.
- `extended_info/agent_<id>_episodes.csv` — per-step rows for environment 0:
  action, reward, `env_reward`, `predicted_reward`, `apple_eaten`,
  `nearby_apples`, `ate_last_apple_in_cluster`.
