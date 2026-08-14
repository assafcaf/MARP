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

## What `num_workers` is worth

Steady-state training throughput, `num_envs=8`, measured from the per-iteration
`time/fps` the trainer logs (so setup and teardown are excluded):

| `num_workers` | agent-steps/s | speedup |
|---|---|---|
| 0 (in-process) | 1,624 | 1.00x |
| 2 | 2,014 | 1.24x |
| 4 | 2,978 | 1.83x |
| 6 | 3,286 | 2.02x |
| 8 | 4,229 | **2.60x** |

Environment stepping alone parallelises better than the run as a whole:

| implementation | agent-steps/s | speedup |
|---|---|---|
| in-process | 2,657 | 1.00x |
| 2 workers | 4,459 | 1.68x |
| 4 workers | 9,305 | **3.50x** |
| 8 workers | 6,252 | 2.35x |

Two things to read off that second table. Stepping scales well up to the point
where each worker owns only one environment; past that the per-step pipe
round-trip costs more than the work it carries, which is why 8 workers over 8
environments is *slower* than 4. And the gap between 3.50x on stepping and
2.60x end-to-end is the serial remainder — action selection, the statistics
accumulator, the reward model's forward pass, logging — none of which moves to a
worker.

### Choosing a value

- Start at `num_workers = num_envs / 2`, so each worker owns two environments.
- More workers than `num_envs` is clamped to `num_envs`.
- Each worker is a process; keep the total under the core count.
- Short runs will look worse than this table: workers take about a second to
  spawn, and closing the TensorBoard writer costs ~5s at the end of *any* run.
  Neither scales with episode count, so both vanish in a real run.

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
3. **Cut `map_to_colors`.** Even after the `state` fix it is the largest single
   cost inside a step. It is a per-cell Python loop over a colour dict; a
   vectorised lookup table would replace it.
