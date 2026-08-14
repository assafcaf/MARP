# Throughput

All numbers below: medium map, 5 agents, view range 7, random policy, 20-core
machine. Measured, not estimated.

## The two knobs

| Key | Default | What it does |
|---|---|---|
| `env.num_envs` | 1 | How many environment copies run per iteration. A **batching** knob: it decorrelates the samples behind each policy update. On its own it buys **no** wall-clock speed. |
| `env.num_workers` | 0 | How many processes those copies are spread across. A **speed** knob. 0 steps them in-process. |
| `env.include_state_in_info` | false | Whether to render the full map to RGB per agent per step. Leave off. |

`num_envs` stays the total episode budget divided across iterations, so raising
it does not change how much experience a run collects.

## A note on the numbers

Absolute throughput on this machine varies by more than 2x depending on what
else is running, so every figure below comes from **interleaved A/B runs** --
the two variants measured back to back within seconds of each other, from the
per-iteration `time/fps` the trainer logs (which excludes setup and teardown).
Trust the ratios; treat the absolute values as indicative.

## What `num_workers` is worth

`num_envs=8`, medium map, 5 agents, random policy:

| `num_workers` | agent-steps/s | speedup |
|---|---|---|
| 0 (in-process) | 4,541 | 1.00x |
| 8 | 18,388 | **4.05x** |

Environment stepping in isolation scales up to the point where each worker owns
a single environment; past that the per-step pipe round trip costs more than the
work it carries. Measured on pure stepping at `num_envs=8`: 3.50x at 4 workers,
falling back to 2.35x at 8. In a full training run the extra workers still help,
because the parent is doing other work between steps.

The remaining gap to linear is the serial part of the loop -- action selection,
the statistics accumulator, the reward model's forward pass, logging -- none of
which moves to a worker.

### Choosing a value

- `num_workers = num_envs` is a reasonable default; `num_envs / 2` costs little
  and leaves cores free.
- More workers than `num_envs` is clamped to `num_envs`.
- Each worker is a process; keep the total under the core count.
- Short runs will look worse than this table: workers take about a second to
  spawn, and closing the TensorBoard writer costs ~5s at the end of *any* run.
  Neither scales with episode count, so both vanish in a real run.

## Colour conversion

`map_to_colors` converts a character grid to RGB. It used to be a nested Python
loop doing a dict lookup and a three-element assignment per cell -- 225 of each
per agent view, per agent, per step. It is now a single fancy-index into a
lookup table built once per colour dict.

| `num_workers` | nested loop | lookup table | speedup |
|---|---|---|---|
| 0 | 2,737 | 4,541 | **1.66x** |
| 8 | 18,479 | 18,388 | 1.00x |

Worth reading carefully: the win is real in-process and **nil** with workers,
because the conversion happens inside the worker, which is no longer the
bottleneck. It is the right change regardless -- it makes the single-process
path much faster, which is what tests, short runs and debugging all use -- but
it will not speed up a worker-backed production run.

## The other 2.19x

`env.include_state_in_info` defaults to `false`. It used to be unconditional:

| | agent-steps/s |
|---|---|
| `include_state_in_info: true` | 1,398 |
| `include_state_in_info: false` | 3,064 |

It rendered the whole map to RGB once per agent per step — 1938 bytes against
the 675 of the observation an agent actually sees — and nothing in this repo
read it. Turn it on only for an external consumer that needs global state; with
workers it also adds ~15 MB per iteration of pipe traffic.

## Results do not depend on any of this

`num_workers` is a pure performance knob. A run at `num_workers=8` produces
byte-identical `metrics.jsonl` records to the same run at `num_workers=0` —
asserted in `tests/test_trainer_parallel_workers.py` for a random policy and for
a full IPPO + reward-model run, and at the vector-env level in
`tests/test_subproc_vec_env.py` for 1, 2 and 4 workers.

Three fixes were needed to make that true, and each was a real bug:

1. **`spawn_point` shuffled its own list in place**, so `reset(seed=...)` did not
   determine the episode — the same seed gave different agent layouts.
2. **Every source of environment randomness drew from the process-global RNG**,
   including apple spawning on every step. Episodes therefore depended on how
   many copies shared a process. Each environment now owns its generators.
   Relatedly, `reset()` used to reseed the *global* streams, resetting the
   policy's exploration RNG once per episode.
3. **`action_space.sample()` uses gymnasium's per-space RNG**, so
   `algorithm=random` ignored `seed` entirely.

Runs seeded before these fixes will not reproduce against runs seeded after
them.

## Where the remaining time goes

Profiling a worker-backed run puts the main process in `posix.read`, blocked on
workers — which is the intended shape. The ceiling is the serial remainder
listed above. If more is needed, in rough order of value:

1. **Overlap acting with stepping.** The main loop currently sends actions, then
   blocks until every worker replies. Double-buffering would hide most of the
   round trip.
2. **Send observations through shared memory** rather than pickling them down a
   pipe. At `num_envs=8`, 5 agents, a 15x15x3 frame, that is ~13.5 KB per step
   each way.
3. **Cut the remaining per-step allocation.** `get_map_with_agents` copies the
   whole world map every step, and `return_view` pads a fresh array per agent.
