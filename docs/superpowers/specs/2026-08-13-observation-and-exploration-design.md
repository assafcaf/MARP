# Wider view, frame stacking, and adaptive exploration — Design

**Date:** 2026-08-13
**Status:** Approved

## Goal

Three changes to bring the training setup closer to the reference implementation
(`DanfoaTestSOT`) and to stop policies collapsing into degenerate
single-action behaviour:

1. Widen the agent view range from 5 to 7.
2. Add optional observation frame stacking (`num_frames`), defaulting to 1.
3. Replace the fixed entropy-coefficient schedule in IPPO and MAPPO with an
   adaptive controller that targets a policy entropy rather than a coefficient.

## Motivation

### Observation shape

`DanfoaTestSOT` runs `agent_view_range: 7` and `num_frames: 2`
(`src/configs/prm.yaml`), applying the stack as an environment wrapper —
`ss.frame_stack_v1(env, num_frames)` in `src/experiment_runner/runners.py:120`.
Its reward predictor reads the stack depth straight off the observation space
(`src/reward_predictor/prm/nn.py:24`), so in the reference the policy *and* the
reward model both see stacked frames.

This repo has `agent_view_range: 5` and no frame stacking at all: `grep -rn
"num_frames\|frame_stack" src/ tests/` returns nothing.

### Exploration

Run `20260813-125003-seed=0` (IPPO, medium map, 5 agents, `narrow_view`,
`efficiency_x_peace`) produced a 25× spread in per-agent return — agent-4
collapsed to a 50.8% `STAY` policy earning ~5 per episode while three others
earned ~120.

The first hypothesis was that entropy annealing removed the exploration budget.
The logged coefficient rules that out:

| episode | `ent_coef` | entropy (nats) |
|---|---|---|
| 0 | 0.0999 | 2.078 |
| 300 | 0.0729 | 0.997 |
| 350 | 0.0684 | 0.639 |
| 500 | 0.0549 | 0.527 |
| 597 | 0.0462 | 0.607 |

Entropy had already collapsed to 0.64 while the coefficient was still 0.068 —
the run never approached the 0.01 floor, ending at 0.046. In the loss,
`ent_coef · entropy ≈ 0.044` against `policy_loss ≈ -0.0014`; the entropy term
was ~30× larger and the policy still went deterministic.

The conclusion is that the equilibrium entropy under this reward is low
regardless of a modest coefficient, so **the coefficient is the wrong variable
to control**. Raising `ent_coef_end` from 0.01 to 0.03 would have been a no-op
for this run. Targeting the entropy directly is the intervention that bites.

## Non-goals

Parameter sharing across agents. `DanfoaTestSOT` runs PRM with
`independent: False` — a single shared `DQNPRM` policy over a VecEnv of agent
slots — whereas `IPPOAlgorithm` builds a separate actor, critic, and optimizer
per agent (`train/algorithms/ippo.py:252-276`). That difference is the larger
lever on per-agent divergence, and it gets its own decision rather than being
bundled here.

## 1. View range 5 → 7

Observation shape is `(2·view+1, 2·view+1, 3)`
(`env/commons_agent.py:44`), so this takes 11×11×3 to 15×15×3.

| File | Change |
|---|---|
| `configs/env/medium.yaml` | `agent_view_range: 5` → `7` |
| `configs/env/small.yaml` | `agent_view_range: 5` → `7` |
| `train/config.py` `EnvConfig` | default `5` → `7` |

Nothing else changes. Every convolutional network derives its flattened size
from a dummy forward pass and reads channel count from `obs_shape[2]` —
`CNNActor`/`CNNCritic` (`algorithms/ippo.py:134-137`), the MAPPO equivalents,
`DQNNetwork`, and `RewardModel` (`reward_model/reward_model.py:42-45`). The
shape change propagates on its own.

## 2. Frame stacking

### Configuration

`EnvConfig` gains `num_frames: int = 1`, surfaced in both env config groups.
**The default of 1 means no behavioural change until a config opts in.**

### `FrameStackEnv`

New module `src/commons_game_marp/env/frame_stack.py`.

- Holds a per-agent `deque(maxlen=num_frames)` of `curr_obs` frames.
- `reset()` fills each deque by repeating the first frame, so the first step
  already has a full stack.
- Concatenates along the channel axis: `(H, W, 3)` → `(H, W, 3·num_frames)`,
  dtype `uint8`.
- `observation_space` returns the widened `Box`.
- `__getattr__` delegates everything else to the wrapped env, so `env.agents`,
  `get_social_metrics()`, `state`, and the video recorder keep working
  untouched.

### Wiring

The trainer wraps the env **only when `num_frames > 1`**. At the default the
env object is the same instance as today, so the default code path is
unchanged rather than merely equivalent.

The reward model requires no changes: `_format_reward_obs` returns `curr_obs`
verbatim (`train/trainer.py:136`) and `RewardModel` is constructed from
`env.observation_space["curr_obs"].shape` (`train/trainer.py:151`). Both pick
the stack up automatically, matching the reference.

### Memory

`PreferenceBuffer` stores raw frames, so resident size scales linearly with
both view area and stack depth:

| configuration | bytes/frame | buffer at `max_episodes_in_buffer: 5000` (600 steps × 5 agents) |
|---|---|---|
| view 5, 1 frame (today) | 363 | 5.4 GB |
| view 7, 1 frame | 675 | 10.1 GB |
| view 7, 2 frames | 1350 | 20.3 GB |

The `PreferenceBuffer` docstring records a real OOM from exactly this
arithmetic. The trainer will log a warning at startup when the projected
resident size — `max_episodes_in_buffer × ep_length × num_agents ×
bytes_per_frame`, or the `store_max_steps_per_agent` cap in place of
`ep_length` when set — exceeds **8 GB**, naming the two knobs that bound it.
This is a warning only; it does not adjust the configuration.

## 3. Adaptive entropy target

### Configuration

Added to both `IPPOConfig` and `MAPPOConfig`:

```yaml
ent_coef_mode: adaptive     # "fixed" | "anneal" | "adaptive"
ent_coef: 0.1               # initial value (adaptive) / schedule start (anneal)
ent_coef_end: 0.01          # anneal only
target_entropy_frac: 0.6    # adaptive: target = frac * ln(num_actions)
ent_coef_lr: 0.0003
ent_coef_min: 0.001
ent_coef_max: 0.5
```

`adaptive` becomes the default for both algorithms. At 8 actions,
`target_entropy_frac: 0.6` gives a target of `0.6 · ln(8) = 1.25` nats —
comfortably above the 0.5–0.64 the collapsed run settled at, and below the
2.08 uniform maximum. This fraction is the primary tuning knob.

MAPPO currently applies a flat `self.config.ent_coef`
(`train/algorithms/mappo.py:365`) with no schedule and no episode tracking. It
gains the full set of fields plus the supporting machinery IPPO already has:
`set_total_episodes`, a `_current_episode` counter updated in `on_episode_end`,
and `ent_coef` in its returned `algo_metrics`. The trainer already calls
`set_total_episodes` behind a `hasattr` check (`train/trainer.py:140-141`), so
no trainer change is needed.

### Mechanism

SAC-style dual gradient ascent on a learnable `log_ent_coef` — one per agent in
IPPO (matching its per-agent networks), one shared in MAPPO. Each has its own
Adam optimizer at `ent_coef_lr`, separate from the policy optimizer.

Per minibatch:

```python
ent_coef = log_ent_coef.exp().clamp(ent_coef_min, ent_coef_max)
ent_coef_loss = log_ent_coef * (entropy.detach() - target_entropy)
```

When entropy is below target the multiplier is negative, so minimising the loss
drives `log_ent_coef` up and strengthens the bonus. The policy loss uses
`ent_coef.detach()` so the controller and the policy do not backpropagate into
each other.

### Modes

- `fixed` — constant `ent_coef`, no schedule, no controller.
- `anneal` — the current linear `ent_coef` → `ent_coef_end` over training.
  Reproduces today's behaviour exactly, so the analysed run stays reproducible
  by setting a single field.
- `adaptive` — the controller above. Default.

### Logging

`algo_metrics` already carries `ent_coef` and `entropy` for IPPO. Add:

- `target_entropy` — the setpoint, so the controller's response is legible in
  `metrics.jsonl` without re-deriving it from config.
- `ent_coef_per_agent` (IPPO only) — the per-agent coefficients. IPPO's
  controllers are per-agent, so a diverging agent shows up directly as a
  diverging coefficient. Given that per-agent divergence is the failure this
  change exists to catch, that series is worth recording rather than averaging
  away. The scalar `ent_coef` stays as the mean across agents.

- `entropy_per_agent` (IPPO only) — same reasoning. `_update_all` already holds
  the per-agent values before averaging, so this costs nothing. The scalar
  `entropy` stays as the mean.

### Documentation

`configs/experiment/example.yaml` documents the algorithm block field by field.
Its IPPO section (lines 126-128) and its commented MAPPO section need the new
fields, and the comment `ent_coef: 0.01  # Constant -- MAPPO has no
ent_coef_end.` (line 171) becomes false and must be rewritten.

## 4. Split gradient clipping

`train/algorithms/ippo.py:492-495` and `train/algorithms/mappo.py:369-372` clip
actor and critic parameters under a single joint norm:

```python
nn.utils.clip_grad_norm_(
    list(actor.parameters()) + list(critic.parameters()),
    self.config.max_grad_norm,
)
```

When `value_loss` spikes — it went 0.02 → 0.14 around episode 100, where
entropy first dropped — the shared clip scales the actor's gradient down too.
The scaling is uniform across both terms so it does not move the entropy
equilibrium, but it does let critic instability silently throttle actor
learning. Replace with separate `clip_grad_norm_` calls per network, in IPPO
and MAPPO.

## Testing

| Test | Covers |
|---|---|
| `tests/test_frame_stack.py` | Observation shape after `reset`/`step` at `num_frames=2`; `num_frames=1` leaves the env unwrapped; frames evict oldest-first; attribute delegation reaches the wrapped env |
| `tests/test_entropy_controller.py` | Coefficient rises when entropy is below target and falls when above; `ent_coef_min`/`ent_coef_max` clamps hold; `anneal` mode reproduces the current linear schedule numerically; `fixed` mode stays constant |
| `tests/test_hydra_configs.py` (extend) | New fields present and typed; every experiment config still composes, including `example.yaml` |
| Buffer-size warning | Fires above the 8 GB threshold, stays silent below it, and honours `store_max_steps_per_agent` in the projection |

The existing suite must stay green.

## Migration

- Runs started after this change use view 7 and adaptive entropy by default;
  they are not comparable to runs recorded before it.
- Reward-model checkpoints carry `obs_shape` and will not load against the new
  observation shape. Existing `reward_model.pt` files become incompatible, which
  is expected for a fresh sweep.
- To reproduce the analysed run exactly: `env.agent_view_range=5`,
  `env.num_frames=1`, `algorithm.ent_coef_mode=anneal`.
