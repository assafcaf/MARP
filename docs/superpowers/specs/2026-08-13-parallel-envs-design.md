# Parallel environments — Design

**Date:** 2026-08-13
**Status:** Approved

## Goal

Run `num_envs` independent copies of `HarvestCommonsEnv` in lockstep so each
policy update consumes decorrelated rollouts from several environments rather
than one long correlated trajectory.

The motivation is **statistical, not wall-clock**: more independent samples per
update. Environments are stepped serially in one process — there are no worker
processes, and env stepping does not get faster.

## Motivation

Today's trainer holds exactly one environment
(`src/commons_game_marp/train/trainer.py:87-105`) and steps it one action at a
time (`trainer.py:270-345`). Every PPO update therefore draws its whole batch
from a single trajectory, where consecutive transitions are strongly
correlated.

The reference implementation (`DanfoaTestSOT`) gets parallel environments from
SuperSuit: `ss.pettingzoo_env_to_vec_env_v1` followed by
`ss.concat_vec_envs_v1(num_vec_envs=num_envs, num_cpus=...)` in
`src/experiment_runner/runners.py:108-131`, driven by `num_envs: 4` in
`src/configs/prm_ppo.yaml:15`. This repo has neither `supersuit` nor
`stable-baselines3` as a dependency and drives its own training loop, so it
needs its own vec layer — but it should adopt SuperSuit's **data layout**
so the two codebases stay legible against each other.

A secondary benefit falls out for free: `IPPOAlgorithm.act`
(`src/commons_game_marp/train/algorithms/ippo.py:286-311`) currently runs one
batch-of-1 forward pass per agent per step. Under the vec layout each agent's
actor sees a `(num_envs, ...)` batch, so the number of forward calls per step
stays at `num_agents` regardless of `num_envs`.

## Non-goals

- **Multiprocessing.** No `SubprocVecEnv` equivalent, no worker pool, no
  pickling of environments. If wall-clock throughput becomes the priority
  later, the interface defined here is the seam a process backend would slot
  into, but nothing in this design anticipates it.
- **SuperSuit / SB3 as dependencies.** We copy a convention, not a library.
- **Auto-reset semantics.** SuperSuit's `MarkovVectorEnv` resets internally
  when all agents are done and returns the *reset* observation, stashing the
  real final one under `infos[agent]["terminal_observation"]`
  (`markov_vector_wrapper.py:76-101`). We deliberately do not copy this; see
  "Lockstep episodes" below.
- **Per-environment RNG streams.** See "Seeding".
- **Tuning DQN for the vec layout.** DQN and the random policy are adapted so
  they remain correct and runnable, not optimised.

## Data layout

One flat batch of `N = num_envs * num_agents` rows, ordered **env-major,
agent-minor**:

```
row index = env_idx * num_agents + agent_idx
```

This is exactly SuperSuit's ordering. `MarkovVectorEnv` maps the `A` agents of
one PettingZoo env onto `A` vector slots in `possible_agents` order
(`markov_vector_wrapper.py:31-47`), and `ConcatVecEnv` stacks `k` of those
along axis 0 with actions sliced `idx : idx + venv.num_envs`
(`concat_vec_env.py:78-99`), which pins env-major ordering.

SuperSuit uses a flat layout because SB3 downstream trains a **single shared
policy** over every row. This repo's IPPO and MAPPO keep per-agent networks and
buffers keyed by `agent_id`, so the flat array alone is not enough. It does not
have to be: `flat.reshape(num_envs, num_agents, ...)` is a zero-copy view of
the same buffer, and `[:, i]` is agent *i*'s `(num_envs, ...)` batch. The flat
array is canonical; the reshape is how per-agent consumers read it.

| quantity | shape | dtype |
|---|---|---|
| observations | `(N, H, W, C)` | `uint8` (as the env emits) |
| actions | `(N,)` | `int64` |
| rewards | `(N,)` | `float32` |
| dones | `(N,)` | `bool` |
| infos | list of `N` dicts | — |

`infos` as a list of per-row dicts is also SuperSuit's convention
(`markov_vector_wrapper.py:92`).

## Components

### `src/commons_game_marp/env/vec_env.py` (new)

```python
class VecCommonsEnv:
    envs: list                          # underlying HarvestCommonsEnv / FrameStackEnv
    num_envs: int
    num_agents: int
    agent_ids: list[str]                # possible_agents order, shared by all envs
    observation_space, action_space     # single-agent spaces, unchanged

    def reset(self) -> tuple[np.ndarray, list[dict]]
    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]
    def compute_social_metrics(self) -> list[dict]     # one dict per env
```

It owns the dict↔flat translation in both directions: gathering each env's
`{agent_id: {"curr_obs": ...}}` into flat rows, and scattering an `(N,)` action
array back into per-env action dicts. No other component performs this
translation.

`envs` stays public because per-episode observability needs the underlying
instances: `compute_agent_step_metrics` takes an `agent` object and an `env`
(`src/commons_game_marp/train/metrics.py`), and `VideoRecorder.record` takes an
env and calls `env.render` (`src/commons_game_marp/train/video_utils.py:48-54`).

The trainer's existing `_build_env` becomes the per-instance factory the vec
env calls `num_envs` times, so frame stacking (`FrameStackEnv`) and every env
config field apply per copy exactly as they do today. `FrameStackEnv` keeps
per-instance stacks, so no change is needed there.

### Lockstep episodes

Every environment shares `ep_length` and is reset together, so the trainer keeps
an explicit episode structure rather than adopting SuperSuit's auto-reset:

```python
for iteration in range(episodes // num_envs):
    obs, infos = vec.reset()
    for step in range(ep_length):
        actions = algo.act(obs, step)
        next_obs, rewards, dones, infos = vec.step(actions)
        algo.observe(obs, actions, rewards, next_obs, dones, infos, step)
        obs = next_obs
```

Exact episode boundaries are load-bearing here in a way they are not for SB3:
social metrics are computed per completed episode
(`HarvestCommonsEnv.compute_social_metrics`), and the preference buffer stores
whole episodes as `EpisodeRecord`s. Auto-reset would force both to be
reconstructed from done flags for no benefit while episodes are fixed-length.

### Configuration

`EnvConfig` gains one field:

```python
num_envs: int = 1   # independent env copies stepped in lockstep
```

The trainer runs `iterations = episodes // num_envs`. `episodes` keeps its
meaning as the **total episode budget**, so raising `num_envs` holds the sample
budget fixed and makes runs comparable across settings; existing YAMLs are
unaffected at the default.

Validation at startup, both hard errors:

- `num_envs >= 1`.
- `episodes % num_envs == 0`. Truncating the budget silently is worse than
  refusing to start.

### Algorithm interface

`Algorithm.act` and `Algorithm.observe` move from agent-keyed dicts to the flat
arrays above:

```python
def act(self, observations: np.ndarray, step: int) -> np.ndarray:   # (N,H,W,C) -> (N,)
def observe(self, observations, actions, rewards, next_observations,
            dones, infos, step) -> None
```

`on_env_ready(env)` additionally reads `num_envs`, `num_agents` and
`agent_ids` off the vec env.

`num_envs=1` goes through the same path — there is no legacy dict branch. Two
code paths for one behaviour is how the two drift apart, and the `num_envs=1`
path is the one every existing config uses, so it must be the tested one.

`dones` is an `(N,)` array; the `"__all__"` key disappears. Because episodes are
fixed-length and lockstep, "the episode ended" is `step == ep_length - 1`,
which the trainer already knows.

### IPPO

- `SingleAgentBuffer` gains an env axis: `obs` becomes `(T, num_envs, ...)`,
  and scalars become `(T, num_envs)`.
- `compute_advantages` is replaced by the vectorized `(T, N)` form MAPPO's
  `RolloutBuffer` already implements (`mappo.py:52-68`), which runs GAE
  independently per column. The two buffers converge on one implementation
  instead of maintaining a scalar version and an array version of the same
  recurrence.
- `act` runs one forward per agent over that agent's `(num_envs, ...)` slice.
- The update flattens `(T, num_envs)` to `(T * num_envs,)` before minibatching.
  Advantage normalisation is over the whole flat batch, unchanged in kind.
- `n_steps` counts **per-env** timesteps, following SB3's convention, so the
  update batch is `n_steps * num_envs`. Under a fixed episode budget this means
  proportionally larger batches and proportionally fewer updates — which is the
  intended effect, not a side effect. `batch_size` (the minibatch) keeps its
  current meaning and default; scale it in config if desired.

### MAPPO

Its `RolloutBuffer` is already `(T, N)` with `N = num_agents`, and
`compute_advantages` already treats columns independently. `N` becomes
`num_envs * num_agents` under the same row convention and the recurrence needs
no change.

`_format_global_obs` (`mappo.py:238`) builds a centralised state from all
agents' observations within **one** env; it becomes per-env and returns
`(num_envs, ...)`. The critic emits `(num_envs, num_agents)`, and the
`np.repeat` at `mappo.py:337` adapts to the wider layout.

### DQN and random

Correct but untuned, as stated in non-goals.

- DQN scatters rows to per-agent replay buffers, pushing `num_envs`
  transitions per agent per step, and acts epsilon-greedily per row.
- The random policy emits `(N,)` uniform integers.

### Reward model and preference buffer

The per-step reward-model path already batches across agents
(`trainer.py:295-302`); the batch simply becomes `num_envs` times wider, and
one `predict_batch` per step still covers every row.

Each iteration appends **`num_envs` `EpisodeRecord`s** to the preference
buffer, one per env, each with its own `agent_trajs`. `max_episodes_in_buffer`
keeps its meaning as a cap in episodes, so projected resident size and the
`BUFFER_WARN_BYTES` arithmetic (`trainer.py:107-131`) are unchanged — only the
rate at which the buffer fills rises.

### Logging, video, per-agent detail

One record per iteration, with `reward_mean`, `reward_per_agent` and the social
metrics **averaged across the `num_envs` episodes** that iteration completed.
The payload gains `num_envs`.

`episode` remains an **episode count, not an iteration count**: it is logged as
`(iteration + 1) * num_envs - 1` so that runs with different `num_envs` overlay
directly on a shared x-axis. Downstream analysis (`src/commons_game_marp/analysis/`)
therefore needs no notion of iterations.

Video recording and `log_agent_episode_details` cover **env 0 only**. Recording
all `num_envs` would multiply per-step JSON volume by `num_envs` for data whose
summary is already in the main log.

## Seeding

`MapEnv.reset` reseeds the *global* `np.random` and `random` modules
(`map_env.py:174-175`), and every stochastic env draw reads that global state:
apple respawn (`commons_env.py:180`), spawn-point shuffling
(`map_env.py:613`), initial orientation (`map_env.py:623`). There is no
per-instance RNG.

Consequently SuperSuit's `seed + i` per-sub-env seeding
(`concat_vec_env.py:38-42`) cannot be copied — reseeding one copy reseeds the
process.

**Decision:** keep the single process-level seed set in `Trainer._seed_rngs`,
and pass no seed on vec resets (matching what the trainer does today at
`trainer.py:271`). The `num_envs` copies decorrelate because they draw
*sequentially from one shared stream*, which makes their rollouts genuinely
different, and the run as a whole stays reproducible from its one seed.

What this does not provide is per-env reproducibility or independent per-env
streams. That is accepted: the requirement is that separate runs behave
comparably, not that env *i* be reproducible in isolation. Threading a
per-instance `np.random.Generator` through `MapEnv`, `HarvestCommonsEnv` and
`Agent` is out of scope — it would rewrite files this feature otherwise only
reads, and would change the RNG stream so existing seeded results stop
reproducing.

## Testing

- `tests/test_vec_env.py`
  - Row ordering is `env_idx * num_agents + agent_idx` on both observations
    returned and actions dispatched.
  - Reset and step return the shapes and dtypes tabulated above.
  - With `num_envs=1`, a fixed seed and a fixed action sequence, the vec env
    reproduces the single-env trajectory step for step.
  - Distinct envs diverge: under `num_envs=2` and identical actions, the two
    envs' observations differ within an episode (confirming the shared-stream
    decorrelation the seeding decision relies on).
- GAE: a `done` in one column does not affect the advantages of any other
  column.
- Trainer smoke test: 2 iterations at `num_envs=2` with a short `ep_length`,
  for all four algorithms, with the reward model both on and off.
- Config validation: `episodes % num_envs != 0` and `num_envs < 1` both fail at
  startup with an actionable message.
- The existing suite passes unchanged at the `num_envs=1` default.

## Risks

- **Interface change reaches all four algorithms.** Mitigated by the trainer
  smoke test covering each, and by there being no second code path to keep in
  sync.
- **`n_steps` changes effective batch size.** This is intended and documented
  above, but it means a run at `num_envs=4` is not hyperparameter-identical to
  one at `num_envs=1`. It should be described as such when results are
  compared.
- **Per-step Python overhead grows linearly with `num_envs`.** Accepted: the
  goal is sample decorrelation, and wall-clock cost was explicitly traded away.
