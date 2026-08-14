# Real Parallel Environments — Implementation Plan

**Goal:** Make `env.num_envs` actually use more than one core, and stop paying for
per-step work nothing consumes.

**Measured starting point** (medium map, 5 agents, random policy, in-process):

| | |
|---|---|
| throughput | ~1400 agent-steps/s, flat across `num_envs` 1→8 |
| CPU | ~1 core of 20 (6%) |
| `map_to_colors` | 62% of step time |
| of which `infos["state"]` | ~48% of step time, **consumed by nothing** |

## Why not SuperSuit

The user asked to follow PettingZoo/SuperSuit's approach. The *technique* is
adopted — worker processes over pipes, envs split across workers, the same flat
env-major/agent-minor row layout `vec_env.py` already documents as SuperSuit's.
The *dependency* is not, for three concrete reasons:

1. SuperSuit is not currently a dependency. `pettingzoo` is.
2. Using it means routing through `env/pettingzoo_env.py`, the legacy
   `ParallelEnv` wrapper the trainer does **not** use. That wrapper applies the
   FIRE penalty a second time (`commons_env.step` already applies it) and
   re-introduces the `metric != "Efficiency"` terminal-reward path. Both are
   silent correctness regressions.
3. `concat_vec_envs_v1` auto-resets. `vec_env.py` deliberately does not copy
   that: the social metrics and the preference buffer's `EpisodeRecord`s both
   require exact episode boundaries.

So: same design, in-repo, no semantic change.

## What blocks subprocess workers today

Everything the main process currently reads by reaching into live env objects
must instead be *returned through the pipe*:

| Coupling | Where | Fix |
|---|---|---|
| `infos["state"]` full-map RGB | `map_env.py:151,193` | Task 1 — make opt-in |
| `agent.grid`, `agent.get_pos()`, `env.world_map`, `env.apple_points` | `Trainer._row_step_metrics` | Task 2 — compute in the env, return scalars in `infos` |
| `vec.envs[0]` for video frames | `Trainer.train` → `VideoRecorder.record` | Task 3 — a render command over the pipe |
| `env.compute_social_metrics()` | `VecCommonsEnv.compute_social_metrics` | Task 4 — a command over the pipe |

---

### Task 1: `infos["state"]` becomes opt-in

**Files:** `env/map_env.py`, `env/commons_env.py`, `train/config.py`,
`configs/env/*.yaml`; test `tests/test_env_state_info.py`

1938 bytes per agent per step, 2.9x the observation itself, rendered by a
full-map `map_to_colors` call per agent. Nothing in the repo reads it.

- [ ] Test: `infos[agent]` has no `state` key by default; has it when
      `include_state_in_info=True`; the array matches `env.state`.
- [ ] Add the constructor flag (default `False`) and gate both call sites.
- [ ] Add `EnvConfig.include_state_in_info: bool = False`.
- [ ] Benchmark single-env fps before/after.

### Task 2: per-step agent metrics move into the env

**Files:** `env/commons_env.py`, `train/trainer.py`, `train/metrics.py`;
tests `tests/test_env_step_metrics.py`, `tests/test_trainer_metrics.py`

`nearby_apples` (int) and `ate_last_apple_in_cluster` (bool) are two scalars per
agent. Computing them worker-side and shipping them in `infos` costs ~2 bytes
where the live-object reads cost a process boundary.

- [ ] Test: `infos[agent]["nearby_apples"]` equals
      `count_apples_around(agent.grid, agent.get_pos(), disc_offsets(r))`, and
      `ate_last_apple_in_cluster` matches `check_ate_last_apple_in_cluster`,
      over a scripted episode.
- [ ] Env computes both in `step`, gated on a `step_metrics` flag so an
      unused-metrics run pays nothing.
- [ ] `Trainer` reads them from `infos`; delete `_row_step_metrics`.
- [ ] Existing `tests/test_trainer_metrics.py` must pass unchanged — the
      TensorBoard numbers may not move.

### Task 3: frames come from the env, not the object

**Files:** `env/map_env.py`, `train/video_utils.py`, `env/vec_env.py`

- [ ] Test: `env.render_frame()` returns an `(H, W, 3) uint8` array equal to
      what `render` writes.
- [ ] `VecCommonsEnv.render_frame(env_idx)` delegates; `VideoRecorder.record`
      takes a frame rather than an env.

### Task 4: `SubprocVecCommonsEnv`

**Files:** `env/env_spec.py` (new), `env/subproc_vec_env.py` (new),
`env/vec_env.py`, `train/config.py`, `train/trainer.py`;
test `tests/test_subproc_vec_env.py`

- [ ] `EnvSpec` frozen dataclass with `build()` — picklable, so `spawn` works
      (a bound-method factory is not, and `fork` is unsafe once torch has
      initialised CUDA in the parent).
- [ ] Worker owns `ceil(num_envs / num_workers)` envs, handles
      `reset` / `step` / `social_metrics` / `render` / `close`.
- [ ] Identical public interface to `VecCommonsEnv`: `reset`, `step`,
      `compute_social_metrics`, `num_envs`, `num_agents`, `agent_ids`,
      `num_rows`, `observation_space`, `action_space`.
- [ ] Test: for a fixed seed and action sequence, subproc and in-process
      produce **identical** observations, rewards, dones and social metrics.
- [ ] `EnvConfig.num_workers: int = 0` (0 = in-process). Trainer picks the
      implementation.
- [ ] Workers are torn down on exit and on exception.

### Task 5: measure and document

- [ ] Benchmark fps and CPU across `num_workers` 0,2,4,8 at `num_envs=8`.
- [ ] Update `docs/metrics.md` / README with the real guidance on `num_envs`
      vs `num_workers`.
