# Expanded TensorBoard Metrics — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bring this repo's TensorBoard output up to (and past) what `DanfoaTestSOT` produced — per-action and per-apple-proximity reward-model diagnostics, action distributions, resource-use metrics, unpenalised environment reward, and throughput.

**Architecture:** A new `train/episode_stats.py` holds a pure accumulator that consumes per-step numpy arrays for one iteration and returns a flat metrics dict. The trainer feeds it inside its existing step loop, the environment gains episode-scoped resource counters, and `ResultLogger` routes the resulting sections to TensorBoard tag families. No algorithm or env dynamics change.

**Tech Stack:** Python 3.12, numpy, PyTorch (`torch.utils.tensorboard.SummaryWriter`), Hydra/OmegaConf structured configs, pytest.

**Spec:** `docs/superpowers/specs/2026-08-13-expanded-tensorboard-metrics.md`

## Global Constraints

- Existing TensorBoard tag names and semantics must not change. New tags only.
- No NaN may ever be written to TensorBoard. Empty subsets emit no scalar.
- `compute_phi` output must be unchanged for every supported `phi_key`.
- `logging.detailed_metrics: false` restores the previous tag set (plus `social/*` additions and `time/*`).
- Aggregation covers all `num_envs * num_agents` rows, not just environment 0.
- Existing spelling `fire_sucsses` is preserved; new keys use correct spelling.
- Tests run with `.venv/bin/python -m pytest`.

---

### Task 1: Vectorised nearby-apple counting

**Files:**
- Modify: `src/commons_game_marp/train/metrics.py`
- Test: `tests/test_metrics.py`

**Interfaces:**
- Produces: `disc_offsets(radius: int) -> np.ndarray` of shape `(K, 2)`, integer row/col offsets with `sqrt(dr^2+dc^2) <= radius`; `count_apples_around(world_map: np.ndarray, pos, offsets: np.ndarray) -> int`.
- `count_nearby_apples` keeps its current signature and result.

- [x] **Step 1: Write failing tests** in `tests/test_metrics.py` asserting `count_apples_around` matches `count_nearby_apples` on a hand-built map, and returns 0 for an agent at `OUTCAST_POSITION`.
- [x] **Step 2:** Run `.venv/bin/python -m pytest tests/test_metrics.py -v`. Expect `ImportError`.
- [x] **Step 3:** Implement `disc_offsets` (cached by radius) and `count_apples_around` using a bounds-masked gather over `world_map`.
- [x] **Step 4:** Run the tests. Expect PASS, including the pre-existing ones.
- [x] **Step 5:** Commit `perf(metrics): vectorise nearby-apple counting`.

---

### Task 2: Environment resource counters

**Files:**
- Modify: `src/commons_game_marp/env/commons_env.py`
- Test: `tests/test_env_metrics.py` (create)

**Interfaces:**
- Produces: `HarvestCommonsEnv.compute_social_metrics()` output gains `fire_hit_rate`, `apples_eaten`, `apples_spawned`, `apple_stock_mean`, `apple_stock_min`, `apple_stock_final`, `depletion_fraction`, `timeout_steps`, `reward_min_agent`, `reward_max_agent`, `reward_std_agent`.

- [x] **Step 1:** Write failing tests: run a short episode with a scripted action sequence, assert every new key is present, finite, and that `apples_eaten` equals the number of positive rewards; assert `compute_phi` is unchanged for all `phi_key`s.
- [x] **Step 2:** Run; expect KeyError.
- [x] **Step 3:** Track `self._apple_stock`, `self._apples_spawned`, `self._apples_eaten` in `step`/`custom_map_update`/`custom_reset`; emit the new keys from `compute_social_metrics` and reset them there.
- [x] **Step 4:** Run `tests/test_env_metrics.py tests/test_env_penalty.py`. Expect PASS.
- [x] **Step 5:** Commit `feat(env): record per-episode resource and fairness metrics`.

---

### Task 3: Episode statistics accumulator

**Files:**
- Create: `src/commons_game_marp/train/episode_stats.py`
- Test: `tests/test_episode_stats.py` (create)

**Interfaces:**
- Produces:
  - `ACTION_GROUPS: dict[str, tuple[int, ...]]` mapping `move_left|move_right|move_up|move_down|stay|turn|fire` to action ids (`turn -> (5, 6)`).
  - `NEARBY_BUCKETS: tuple[tuple[str, int, int], ...]` — `("0",0,0), ("1-2",1,2), ("3-4",3,4), ("5+",5,inf)`.
  - `class EpisodeStats(num_actions: int, agent_ids: list[str], track_reward_model: bool)` with
    `record_step(actions, env_rewards, nearby_apples, last_in_cluster, pred_rewards=None)` (all `(num_rows,)` arrays) and
    `result() -> dict[str, dict[str, float]]` keyed by section: `action`, `harvest`, `rm_on_action`, `rm_outcome_avg`, `rm_outcome_std`, `rm_outcome`, `rm_by_nearby_apples`, `rm_pred`.

- [x] **Step 1:** Write failing tests covering: action fractions sum to 1; `turn` merges 5 and 6; entropy of a uniform distribution equals `ln(7)`; harvest rate and `last_in_cluster_rate`; per-action predicted-reward means; `apple_eaten`/`no_apple_eaten` means, stds, `delta`, and `separation`; nearby-apple bucket partitioning; step correlation on a perfectly correlated input equals 1.0; **empty subsets are absent from the result, never NaN**; `track_reward_model=False` emits no `rm_*` sections.
- [x] **Step 2:** Run; expect `ModuleNotFoundError`.
- [x] **Step 3:** Implement with per-step appends into python lists of small numpy arrays, concatenated once in `result()`.
- [x] **Step 4:** Run `.venv/bin/python -m pytest tests/test_episode_stats.py -v`. Expect PASS.
- [x] **Step 5:** Commit `feat(train): add episode statistics accumulator`.

---

### Task 4: Logging config keys

**Files:**
- Modify: `src/commons_game_marp/train/config.py`
- Modify: `src/commons_game_marp/configs/logging/*.yaml` (documentation comments)
- Test: `tests/test_hydra_configs.py`

**Interfaces:**
- Produces: `LoggingConfig.detailed_metrics: bool = True`, `LoggingConfig.nearby_apple_radius: int = 2`, `LoggingConfig.histogram_every_n_episodes: int = 0`.

- [x] **Step 1:** Write a failing test asserting the three defaults compose under Hydra.
- [x] **Step 2:** Run; expect failure.
- [x] **Step 3:** Add the fields with explanatory comments.
- [x] **Step 4:** Run `tests/test_hydra_configs.py tests/test_config_num_envs.py`. Expect PASS.
- [x] **Step 5:** Commit `feat(config): add detailed-metrics logging switches`.

---

### Task 5: TensorBoard routing

**Files:**
- Modify: `src/commons_game_marp/train/logging_utils.py`
- Test: `tests/test_logging_sections.py` (create)

**Interfaces:**
- Consumes: section dicts from Task 3, `reward_env_*` payload keys, `time` payload key.
- Produces: `ResultLogger.log_episode` routes `payload["sections"]` — `{section_name: {tag: value}}` — to `add_scalar(f"{section}/{tag}", …)`, and flattens nested `algo_metrics` dicts to `algo/<name>/<agent_id>`.

- [x] **Step 1:** Write failing tests using a fake writer: sections route to the right tags, non-finite values are dropped, nested algo dicts flatten, existing tags still emitted.
- [x] **Step 2:** Run; expect failure.
- [x] **Step 3:** Implement `_log_sections` and the nested-dict branch; keep every existing line.
- [x] **Step 4:** Run `tests/test_logging_sections.py`. Expect PASS.
- [x] **Step 5:** Commit `feat(logging): route metric sections to tensorboard`.

---

### Task 6: Trainer wiring

**Files:**
- Modify: `src/commons_game_marp/train/trainer.py`
- Test: `tests/test_trainer_metrics.py` (create)

**Interfaces:**
- Consumes: everything above.
- Produces: `metrics.jsonl` records gain `sections`, `reward_env_sum`, `reward_env_mean`, `reward_env_per_agent`, `time`.

- [x] **Step 1:** Write a failing end-to-end test: two-episode run with `num_envs=2`, `detailed_metrics=True`, reward model off; assert the JSONL record carries the `action` and `harvest` sections with sane values, and `reward_env_*`; and a second test with `detailed_metrics=False` asserting they are absent.
- [x] **Step 2:** Run; expect failure.
- [x] **Step 3:** Compute `nearby_apples` and `last_in_cluster` for **all rows** each step, feed `EpisodeStats`, read `infos[row]['r']` for the unpenalised reward, time the iteration, and attach the sections to the payload. Keep the existing env-0 CSV path working by reading from the same per-row arrays.
- [x] **Step 4:** Run the full suite `.venv/bin/python -m pytest tests/ -q`. Expect PASS.
- [x] **Step 5:** Commit `feat(train): log action, harvest and reward-model diagnostics`.

---

### Task 7: Histograms and documentation

**Files:**
- Modify: `src/commons_game_marp/train/logging_utils.py`, `src/commons_game_marp/train/trainer.py`
- Create: `docs/metrics.md`
- Test: `tests/test_logging_sections.py`

- [x] **Step 1:** Write a failing test that `histogram_every_n_episodes=2` produces `rm_pred/hist` and `reward/agent_hist` on even episodes only.
- [x] **Step 2:** Run; expect failure.
- [x] **Step 3:** Implement `log_histograms`, gate it in the trainer, and write `docs/metrics.md` documenting every tag with its definition and how to read it.
- [x] **Step 4:** Run the full suite.
- [x] **Step 5:** Commit `docs(metrics): document the tensorboard tag catalog`.

---

## Self-review

- Spec §1 → Task 2. §2, §3 → Tasks 1, 3, 6. §4 → Tasks 5, 6. §5 → Tasks 3, 6.
  §6 → Task 6. §7 → Task 5. Configuration → Task 4. Histograms → Task 7.
- No placeholders: each task names exact files, exact new identifiers, and the
  exact assertions its tests must make.
- Names used consistently: `count_apples_around`, `disc_offsets`,
  `EpisodeStats.record_step`, `EpisodeStats.result`, `ACTION_GROUPS`,
  `NEARBY_BUCKETS`, `detailed_metrics`, `nearby_apple_radius`,
  `histogram_every_n_episodes`.
