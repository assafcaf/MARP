# Parallel Environments Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run `num_envs` independent copies of `HarvestCommonsEnv` in lockstep so each policy update consumes decorrelated rollouts from several environments instead of one correlated trajectory.

**Architecture:** A new `VecCommonsEnv` steps `num_envs` environment instances serially in one process and presents them as one flat batch of `N = num_envs * num_agents` rows, ordered env-major/agent-minor (`row = env_idx * num_agents + agent_idx`) — SuperSuit's layout, without the SuperSuit dependency. `Algorithm.act`/`observe` move from agent-keyed dicts to those flat arrays; per-agent consumers recover their slice through a zero-copy `reshape(num_envs, num_agents, ...)` view. Episodes stay fixed-length and lockstep with explicit resets, so social metrics and preference-buffer episode records keep exact boundaries.

**Tech Stack:** Python 3.11+, NumPy, PyTorch 2.5.1, Hydra/OmegaConf configs, pytest.

**Spec:** `docs/superpowers/specs/2026-08-13-parallel-envs-design.md`

## Global Constraints

- Row ordering is **env-major, agent-minor**: `row = env_idx * num_agents + agent_idx`, agents in `VecCommonsEnv.agent_ids` order. Every producer and consumer of a flat batch obeys this.
- Flat batch shapes and dtypes: observations `(N, H, W, C)` `uint8`, actions `(N,)` `int64`, rewards `(N,)` `float32`, dones `(N,)` `bool`, infos a Python `list` of `N` dicts.
- `episodes` in config remains the **total episode budget**. The trainer runs `episodes // num_envs` iterations.
- `num_envs` defaults to `1`, and `num_envs=1` uses the same code path as `num_envs>1` — no legacy dict branch anywhere.
- `n_steps` counts **per-env** timesteps (SB3 convention); the resulting update batch is `n_steps * num_envs`.
- No new third-party dependencies. `supersuit` and `stable-baselines3` are **not** to be added.
- No multiprocessing, no auto-reset, no per-instance RNG refactor. The environment keeps using the global `np.random`/`random` streams seeded once in `Trainer._seed_rngs`.
- Run tests with `uv run pytest`.

---

### Task 1: Shared vectorized GAE

MAPPO already has a correct column-wise GAE implementation and IPPO has a scalar one. Both need the column-wise form, so it moves into one module they share.

**Files:**
- Create: `src/commons_game_marp/train/algorithms/gae.py`
- Modify: `src/commons_game_marp/train/algorithms/mappo.py:52-68` (`RolloutBuffer.compute_advantages`)
- Test: `tests/test_gae.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `compute_gae(rewards, dones, values, next_values, gamma, gae_lambda) -> tuple[np.ndarray, np.ndarray]`, all inputs `(T, N)` float arrays, returning `(advantages, returns)` both `(T, N)` `float32`. Tasks 4 and 5 call it.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_gae.py`:

```python
"""GAE must treat every column as an independent trajectory.

With parallel environments each column of the (T, N) batch is a different
env/agent rollout. A `done` in one column terminating another column's
bootstrap would silently corrupt advantages for every environment but the
first, and the resulting policy would still train -- just on wrong targets.
"""

import numpy as np
import pytest

from commons_game_marp.train.algorithms.gae import compute_gae


def _single_column(rewards, dones, values, next_values, gamma, lam):
    """Reference scalar implementation, one trajectory."""
    T = len(rewards)
    adv = np.zeros(T, dtype=np.float32)
    last = 0.0
    for t in reversed(range(T)):
        mask = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_values[t] * mask - values[t]
        last = delta + gamma * lam * mask * last
        adv[t] = last
    return adv


def test_matches_scalar_reference_for_one_column():
    rng = np.random.default_rng(0)
    T = 12
    rewards = rng.normal(size=(T, 1)).astype(np.float32)
    dones = np.zeros((T, 1), dtype=np.float32)
    values = rng.normal(size=(T, 1)).astype(np.float32)
    next_values = rng.normal(size=(T, 1)).astype(np.float32)

    adv, ret = compute_gae(rewards, dones, values, next_values, 0.99, 0.95)
    expected = _single_column(
        rewards[:, 0], dones[:, 0], values[:, 0], next_values[:, 0], 0.99, 0.95
    )

    np.testing.assert_allclose(adv[:, 0], expected, rtol=1e-6)
    np.testing.assert_allclose(ret[:, 0], adv[:, 0] + values[:, 0], rtol=1e-6)


def test_done_in_one_column_does_not_affect_another():
    """The regression this module exists to prevent."""
    rng = np.random.default_rng(1)
    T = 10
    rewards = rng.normal(size=(T, 3)).astype(np.float32)
    values = rng.normal(size=(T, 3)).astype(np.float32)
    next_values = rng.normal(size=(T, 3)).astype(np.float32)
    dones = np.zeros((T, 3), dtype=np.float32)
    dones[4, 0] = 1.0  # only column 0 terminates

    adv, _ = compute_gae(rewards, dones, values, next_values, 0.99, 0.95)
    expected_col2 = _single_column(
        rewards[:, 2], dones[:, 2], values[:, 2], next_values[:, 2], 0.99, 0.95
    )

    np.testing.assert_allclose(adv[:, 2], expected_col2, rtol=1e-6)


def test_done_truncates_bootstrap_within_its_own_column():
    rewards = np.ones((3, 1), dtype=np.float32)
    values = np.zeros((3, 1), dtype=np.float32)
    next_values = np.full((3, 1), 100.0, dtype=np.float32)
    dones = np.array([[0.0], [1.0], [0.0]], dtype=np.float32)

    adv, _ = compute_gae(rewards, dones, values, next_values, 0.99, 0.95)

    # t=1 is terminal: reward only, no bootstrap and no carry from t=2.
    assert adv[1, 0] == pytest.approx(1.0)


def test_rejects_mismatched_shapes():
    ok = np.zeros((4, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="same shape"):
        compute_gae(ok, ok, ok, np.zeros((4, 3), dtype=np.float32), 0.99, 0.95)


def test_rejects_one_dimensional_input():
    bad = np.zeros(4, dtype=np.float32)
    with pytest.raises(ValueError, match=r"\(T, N\)"):
        compute_gae(bad, bad, bad, bad, 0.99, 0.95)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_gae.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'commons_game_marp.train.algorithms.gae'`

- [ ] **Step 3: Write the implementation**

Create `src/commons_game_marp/train/algorithms/gae.py`:

```python
"""Generalized advantage estimation over a batch of independent columns.

Shared by IPPO and MAPPO. Each column of the (T, N) batch is one independent
trajectory -- under parallel environments, one (env, agent) pair -- and the
recurrence never crosses columns. Keeping a single implementation is the point:
a scalar variant and an array variant of the same recurrence drift apart, and
the drift is invisible in training curves.
"""

from typing import Tuple

import numpy as np


def compute_gae(
    rewards: np.ndarray,
    dones: np.ndarray,
    values: np.ndarray,
    next_values: np.ndarray,
    gamma: float,
    gae_lambda: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Advantages and returns for a (T, N) batch of independent trajectories.

    Parameters
    ----------
    rewards, dones, values, next_values:
        Arrays of shape (T, N). `dones` is treated as a float mask, so bool
        arrays are accepted.
    gamma, gae_lambda:
        Discount and GAE trace decay.

    Returns
    -------
    (advantages, returns), both (T, N) float32.
    """
    rewards = np.asarray(rewards, dtype=np.float32)
    dones = np.asarray(dones, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)
    next_values = np.asarray(next_values, dtype=np.float32)

    if rewards.ndim != 2:
        raise ValueError(f"compute_gae expects (T, N) arrays, got {rewards.shape}")
    if not (rewards.shape == dones.shape == values.shape == next_values.shape):
        raise ValueError(
            "rewards, dones, values and next_values must have the same shape; got "
            f"{rewards.shape}, {dones.shape}, {values.shape}, {next_values.shape}"
        )

    T, N = rewards.shape
    advantages = np.zeros((T, N), dtype=np.float32)
    last_adv = np.zeros((N,), dtype=np.float32)
    for t in reversed(range(T)):
        mask = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_values[t] * mask - values[t]
        last_adv = delta + gamma * gae_lambda * mask * last_adv
        advantages[t] = last_adv
    returns = advantages + values
    return advantages, returns
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_gae.py -v`
Expected: PASS, 5 tests.

- [ ] **Step 5: Point MAPPO's buffer at the shared implementation**

In `src/commons_game_marp/train/algorithms/mappo.py`, add the import beside the existing ones:

```python
from .gae import compute_gae
```

Replace the whole body of `RolloutBuffer.compute_advantages` (currently `mappo.py:52-68`) with:

```python
    def compute_advantages(
        self, gamma: float, gae_lambda: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        return compute_gae(
            rewards=np.stack(self.rewards, axis=0),
            dones=np.stack(self.dones, axis=0),
            values=np.stack(self.values, axis=0),
            next_values=np.stack(self.next_values, axis=0),
            gamma=gamma,
            gae_lambda=gae_lambda,
        )
```

- [ ] **Step 6: Run the full suite — this step is a pure refactor and must not change behaviour**

Run: `uv run pytest`
Expected: PASS, same test count as before this task plus the 5 new ones.

- [ ] **Step 7: Commit**

```bash
git add src/commons_game_marp/train/algorithms/gae.py src/commons_game_marp/train/algorithms/mappo.py tests/test_gae.py
git commit -m "refactor(algorithms): extract column-wise GAE into a shared module"
```

---

### Task 2: `VecCommonsEnv` and the row conversion helpers

**Files:**
- Create: `src/commons_game_marp/env/vec_env.py`
- Test: `tests/test_vec_env.py`

**Interfaces:**
- Consumes: `HarvestCommonsEnv` (`src/commons_game_marp/env/commons_env.py`), `MAP` (`src/commons_game_marp/env/maps.py` via `commons_env`).
- Produces, used by Tasks 3-5:
  - `rows_to_agents(rows: np.ndarray, num_envs: int, agent_ids: list[str]) -> dict[str, np.ndarray]`
  - `agents_to_rows(per_agent: dict[str, np.ndarray], num_envs: int, agent_ids: list[str]) -> np.ndarray`
  - `class VecCommonsEnv` with attributes `envs`, `num_envs`, `num_agents`, `agent_ids`, `num_rows`, `observation_space`, `action_space`, and methods `reset()`, `step(actions)`, `compute_social_metrics()`, `rows_to_agents(rows)`, `agents_to_rows(per_agent)`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_vec_env.py`:

```python
"""The vec layer owns the dict<->flat-row translation, and nothing else does.

Row ordering is env-major/agent-minor -- row = env_idx * num_agents +
agent_idx -- copied from SuperSuit's ConcatVecEnv so the two codebases stay
comparable. A transposed ordering would still train; it would just attribute
every observation to the wrong agent, which no shape assertion would catch.
"""

import numpy as np
import pytest

from commons_game_marp.env.commons_env import HarvestCommonsEnv, MAP
from commons_game_marp.env.vec_env import (
    VecCommonsEnv,
    agents_to_rows,
    rows_to_agents,
)


def make_env():
    return HarvestCommonsEnv(
        ascii_map=MAP["small"],
        num_agents=2,
        render=False,
        agent_view_range=3,
        ep_length=20,
        spawn_speed="slow",
        metric="Efficiency",
        penalty=False,
    )


@pytest.fixture
def vec():
    return VecCommonsEnv(make_env, num_envs=3)


def test_rows_to_agents_uses_env_major_ordering():
    agent_ids = ["agent-0", "agent-1"]
    # 3 envs x 2 agents; value encodes (env, agent) as env * 10 + agent
    rows = np.array([0, 1, 10, 11, 20, 21], dtype=np.int64)

    per_agent = rows_to_agents(rows, num_envs=3, agent_ids=agent_ids)

    np.testing.assert_array_equal(per_agent["agent-0"], [0, 10, 20])
    np.testing.assert_array_equal(per_agent["agent-1"], [1, 11, 21])


def test_rows_to_agents_preserves_trailing_dimensions():
    rows = np.zeros((4, 5, 5, 3), dtype=np.uint8)
    per_agent = rows_to_agents(rows, num_envs=2, agent_ids=["a", "b"])
    assert per_agent["a"].shape == (2, 5, 5, 3)


def test_agents_to_rows_is_the_inverse_of_rows_to_agents():
    agent_ids = ["agent-0", "agent-1"]
    rows = np.arange(6, dtype=np.int64)
    per_agent = rows_to_agents(rows, num_envs=3, agent_ids=agent_ids)

    np.testing.assert_array_equal(
        agents_to_rows(per_agent, num_envs=3, agent_ids=agent_ids), rows
    )


def test_reset_returns_flat_rows_and_per_row_infos(vec):
    obs, infos = vec.reset()

    assert vec.num_rows == 6
    assert obs.shape == (6, *vec.observation_space["curr_obs"].shape)
    assert obs.dtype == np.uint8
    assert isinstance(infos, list) and len(infos) == 6


def test_step_returns_the_documented_shapes_and_dtypes(vec):
    vec.reset()
    actions = np.zeros(vec.num_rows, dtype=np.int64)

    obs, rewards, dones, infos = vec.step(actions)

    assert obs.shape == (6, *vec.observation_space["curr_obs"].shape)
    assert obs.dtype == np.uint8
    assert rewards.shape == (6,) and rewards.dtype == np.float32
    assert dones.shape == (6,) and dones.dtype == np.bool_
    assert isinstance(infos, list) and len(infos) == 6


def test_actions_are_dispatched_to_the_right_env_and_agent():
    """Each row's action must reach exactly one (env, agent) pair.

    Action 7 is FIRE, which the env records per agent in infos[...]['fire'].
    Firing from a single row must show up in that row's info and no other.
    """
    vec = VecCommonsEnv(make_env, num_envs=3)
    vec.reset()
    actions = np.zeros(vec.num_rows, dtype=np.int64)
    fire_row = 1 * vec.num_agents + 1  # env 1, agent 1
    actions[fire_row] = 7

    _, _, _, infos = vec.step(actions)

    assert infos[fire_row]["fire"] is True
    assert all(
        infos[r]["fire"] is False for r in range(vec.num_rows) if r != fire_row
    )


def test_single_env_reproduces_the_unwrapped_environment():
    """num_envs=1 must be the same trajectory, not merely a valid one."""
    actions = [np.array([3, 5], dtype=np.int64) for _ in range(5)]

    np.random.seed(1234)
    import random as _random

    _random.seed(1234)
    plain = make_env()
    plain.reset(seed=None)
    plain_obs = []
    for step_actions in actions:
        obs, _, _, _ = plain.step(
            {"agent-0": int(step_actions[0]), "agent-1": int(step_actions[1])}
        )
        plain_obs.append(np.stack([obs["agent-0"]["curr_obs"], obs["agent-1"]["curr_obs"]]))

    np.random.seed(1234)
    _random.seed(1234)
    vec = VecCommonsEnv(make_env, num_envs=1)
    vec.reset()
    vec_obs = []
    for step_actions in actions:
        obs, _, _, _ = vec.step(step_actions)
        vec_obs.append(obs)

    for a, b in zip(plain_obs, vec_obs):
        np.testing.assert_array_equal(a, b)


def test_parallel_envs_diverge_under_identical_actions():
    """The shared global RNG stream is what decorrelates the copies.

    The env has no per-instance RNG (MapEnv.reset reseeds the global modules),
    so the copies differ only because they draw sequentially from one stream.
    If that ever stopped being true, every env would be a duplicate and the
    whole feature would be a no-op -- which this test is here to catch.
    """
    vec = VecCommonsEnv(make_env, num_envs=2)
    vec.reset()
    actions = np.zeros(vec.num_rows, dtype=np.int64)

    diverged = False
    for _ in range(20):
        obs, _, _, _ = vec.step(actions)
        env0 = obs[: vec.num_agents]
        env1 = obs[vec.num_agents :]
        if not np.array_equal(env0, env1):
            diverged = True
            break

    assert diverged, "parallel envs produced identical observations"


def test_compute_social_metrics_returns_one_dict_per_env(vec):
    vec.reset()
    actions = np.zeros(vec.num_rows, dtype=np.int64)
    for _ in range(3):
        vec.step(actions)

    metrics = vec.compute_social_metrics()

    assert len(metrics) == 3
    assert all("efficiency" in m for m in metrics)


def test_social_metrics_are_snapshots_not_live_references(vec):
    """`HarvestCommonsEnv.get_social_metrics` returns its own mutable dict."""
    vec.reset()
    actions = np.zeros(vec.num_rows, dtype=np.int64)
    vec.step(actions)
    first = vec.compute_social_metrics()
    first_efficiency = first[0]["efficiency"]

    for _ in range(5):
        vec.step(actions)
    vec.compute_social_metrics()

    assert first[0]["efficiency"] == first_efficiency


def test_rejects_num_envs_below_one():
    with pytest.raises(ValueError, match="num_envs"):
        VecCommonsEnv(make_env, num_envs=0)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_vec_env.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'commons_game_marp.env.vec_env'`

- [ ] **Step 3: Write the implementation**

Create `src/commons_game_marp/env/vec_env.py`:

```python
"""Several environment copies stepped in lockstep, presented as flat rows.

The layout is SuperSuit's, so this repo and `DanfoaTestSOT` describe the same
thing the same way. `pettingzoo_env_to_vec_env_v1` maps the A agents of one
env onto A vector slots in `possible_agents` order, and `concat_vec_envs_v1`
stacks k of those along axis 0 -- which fixes the ordering as env-major,
agent-minor:

    row = env_idx * num_agents + agent_idx

SuperSuit is flat because SB3 trains one shared policy over every row. IPPO and
MAPPO here keep per-agent networks, so they need per-agent slices instead -- and
`rows.reshape(num_envs, num_agents, ...)` is a zero-copy view that provides
them. The flat array stays canonical; the reshape is how per-agent consumers
read it.

Deliberately *not* copied from SuperSuit: auto-reset. Episodes here are
fixed-length and lockstep, and both the social metrics and the preference
buffer's episode records depend on exact episode boundaries, so the trainer
resets explicitly.
"""

from typing import Any, Callable, Dict, List, Tuple

import numpy as np


def rows_to_agents(
    rows: np.ndarray, num_envs: int, agent_ids: List[str]
) -> Dict[str, np.ndarray]:
    """View a flat (N, ...) batch as {agent_id: (num_envs, ...)}.

    The reshape is a view, so no data is copied; each agent's entry is a
    strided view into `rows`.
    """
    num_agents = len(agent_ids)
    reshaped = rows.reshape(num_envs, num_agents, *rows.shape[1:])
    return {agent_id: reshaped[:, i] for i, agent_id in enumerate(agent_ids)}


def agents_to_rows(
    per_agent: Dict[str, np.ndarray], num_envs: int, agent_ids: List[str]
) -> np.ndarray:
    """Inverse of `rows_to_agents`: {agent_id: (num_envs, ...)} -> (N, ...)."""
    stacked = np.stack([per_agent[agent_id] for agent_id in agent_ids], axis=1)
    return stacked.reshape(num_envs * len(agent_ids), *stacked.shape[2:])


class VecCommonsEnv:
    """`num_envs` independent environments stepped serially in one process.

    Serial by design: the goal is decorrelated samples per update, not
    wall-clock throughput, so there are no worker processes and stepping does
    not get faster.
    """

    def __init__(self, env_fn: Callable[[], Any], num_envs: int) -> None:
        if num_envs < 1:
            raise ValueError(f"num_envs must be >= 1, got {num_envs}")
        self.num_envs = int(num_envs)
        self.envs = [env_fn() for _ in range(self.num_envs)]

        probe = self.envs[0]
        probe.reset(seed=None)
        self.agent_ids: List[str] = list(probe.agents.keys())
        self.num_agents = len(self.agent_ids)
        self.observation_space = probe.observation_space
        self.action_space = probe.action_space

    @property
    def num_rows(self) -> int:
        return self.num_envs * self.num_agents

    def rows_to_agents(self, rows: np.ndarray) -> Dict[str, np.ndarray]:
        return rows_to_agents(rows, self.num_envs, self.agent_ids)

    def agents_to_rows(self, per_agent: Dict[str, np.ndarray]) -> np.ndarray:
        return agents_to_rows(per_agent, self.num_envs, self.agent_ids)

    def reset(self) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """Reset every environment.

        No seed is passed: `MapEnv.reset` reseeds the *global* numpy and random
        modules, so seeding per copy would reseed the process. The copies
        decorrelate by drawing sequentially from the one stream seeded in
        `Trainer._seed_rngs`.
        """
        obs_rows: List[np.ndarray] = []
        infos: List[Dict[str, Any]] = []
        for env in self.envs:
            observations, env_infos = env.reset(seed=None)
            for agent_id in self.agent_ids:
                obs_rows.append(observations[agent_id]["curr_obs"])
                infos.append(env_infos.get(agent_id, {}))
        return np.stack(obs_rows, axis=0), infos

    def step(
        self, actions: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        actions = np.asarray(actions, dtype=np.int64)
        if actions.shape != (self.num_rows,):
            raise ValueError(
                f"expected actions of shape ({self.num_rows},), got {actions.shape}"
            )
        per_env_actions = actions.reshape(self.num_envs, self.num_agents)

        obs_rows: List[np.ndarray] = []
        rewards: List[float] = []
        dones: List[bool] = []
        infos: List[Dict[str, Any]] = []
        for env_idx, env in enumerate(self.envs):
            action_dict = {
                agent_id: int(per_env_actions[env_idx, i])
                for i, agent_id in enumerate(self.agent_ids)
            }
            observations, env_rewards, env_dones, env_infos = env.step(action_dict)
            all_done = bool(env_dones.get("__all__", False))
            for agent_id in self.agent_ids:
                obs_rows.append(observations[agent_id]["curr_obs"])
                rewards.append(float(env_rewards[agent_id]))
                dones.append(bool(env_dones.get(agent_id, False)) or all_done)
                infos.append(env_infos.get(agent_id, {}))

        return (
            np.stack(obs_rows, axis=0),
            np.asarray(rewards, dtype=np.float32),
            np.asarray(dones, dtype=bool),
            infos,
        )

    def compute_social_metrics(self) -> List[Dict[str, float]]:
        """Per-env social metrics for the episode just finished.

        Copied, not aliased: `HarvestCommonsEnv.get_social_metrics` returns the
        env's own `self.metrics` dict, which it mutates on the next episode.
        """
        metrics = []
        for env in self.envs:
            env.compute_social_metrics()
            metrics.append(dict(env.get_social_metrics()))
        return metrics
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_vec_env.py -v`
Expected: PASS, 12 tests.

- [ ] **Step 5: Run the full suite**

Run: `uv run pytest`
Expected: PASS — nothing consumes the new module yet.

- [ ] **Step 6: Commit**

```bash
git add src/commons_game_marp/env/vec_env.py tests/test_vec_env.py
git commit -m "feat(env): add VecCommonsEnv with SuperSuit's flat row layout"
```

---

### Task 3: `num_envs` config field and iteration accounting

**Files:**
- Modify: `src/commons_game_marp/train/config.py:8-21` (`EnvConfig`), and add `resolve_iterations` at module level
- Test: `tests/test_config_num_envs.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `EnvConfig.num_envs: int = 1`; `resolve_iterations(episodes: int, num_envs: int) -> int`. Task 4's trainer calls `resolve_iterations`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_config_num_envs.py`:

```python
"""`episodes` stays a total episode budget, so runs stay comparable as
`num_envs` changes. A non-divisible pair silently truncates that budget, which
is exactly the kind of quiet difference that invalidates a comparison between
two runs -- so it is refused at startup instead.
"""

import pytest

from commons_game_marp.train.config import EnvConfig, resolve_iterations


def test_num_envs_defaults_to_one():
    assert EnvConfig().num_envs == 1


def test_iterations_divide_the_episode_budget():
    assert resolve_iterations(episodes=1000, num_envs=4) == 250


def test_single_env_runs_one_iteration_per_episode():
    assert resolve_iterations(episodes=17, num_envs=1) == 17


def test_non_divisible_budget_is_refused_with_workable_values():
    with pytest.raises(ValueError) as excinfo:
        resolve_iterations(episodes=1000, num_envs=3)

    message = str(excinfo.value)
    assert "999" in message and "1002" in message


def test_num_envs_below_one_is_refused():
    with pytest.raises(ValueError, match="num_envs"):
        resolve_iterations(episodes=100, num_envs=0)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_config_num_envs.py -v`
Expected: FAIL — `ImportError: cannot import name 'resolve_iterations'`

- [ ] **Step 3: Add the config field**

In `src/commons_game_marp/train/config.py`, add to `EnvConfig` immediately after `num_frames`:

```python
    # Independent environment copies stepped in lockstep. Above 1 the trainer
    # runs `episodes // num_envs` iterations, each completing num_envs
    # episodes, so `episodes` stays a total budget rather than an iteration
    # count. Stepping is serial -- this buys decorrelated samples per update,
    # not wall-clock speed.
    num_envs: int = 1
```

- [ ] **Step 4: Add the accounting helper**

Add to `src/commons_game_marp/train/config.py`, after the dataclasses and before `register_configs`:

```python
def resolve_iterations(episodes: int, num_envs: int) -> int:
    """Iterations needed to spend an `episodes` budget `num_envs` at a time.

    Refuses a non-divisible pair rather than truncating: a run that quietly
    completes 996 of a requested 1000 episodes is not comparable with one that
    completed 1000, and nothing downstream would show the difference.
    """
    if num_envs < 1:
        raise ValueError(f"env.num_envs must be >= 1, got {num_envs}")
    remainder = episodes % num_envs
    if remainder:
        raise ValueError(
            f"episodes ({episodes}) must be divisible by env.num_envs ({num_envs}); "
            f"otherwise the episode budget is silently truncated. "
            f"Use {episodes - remainder} or {episodes + num_envs - remainder} episodes."
        )
    return episodes // num_envs
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/test_config_num_envs.py -v`
Expected: PASS, 5 tests.

- [ ] **Step 6: Run the full suite**

Run: `uv run pytest`
Expected: PASS. `tests/test_hydra_configs.py` composes the real configs against the schema, so it also confirms the new field does not break composition.

- [ ] **Step 7: Commit**

```bash
git add src/commons_game_marp/train/config.py tests/test_config_num_envs.py
git commit -m "feat(config): add env.num_envs and total-budget iteration accounting"
```

---

### Task 4: Flip the algorithm interface to flat rows and run the trainer on `VecCommonsEnv`

**This task is deliberately atomic.** `Algorithm.act`/`observe` and the trainer loop are two halves of one interface; there is no intermediate state where the suite is green, and a reviewer cannot accept one half while rejecting the other. Commits inside this task will have a red suite until its final steps — that is expected, and the task is complete only when Step 16 passes.

**Files:**
- Modify: `src/commons_game_marp/train/algorithms/base.py` (whole file)
- Modify: `src/commons_game_marp/train/algorithms/random_policy.py` (whole file)
- Modify: `src/commons_game_marp/train/algorithms/dqn.py:129-190` (`DQNAlgorithm.on_env_ready`, `_format_obs`, `act`, `observe`)
- Modify: `src/commons_game_marp/train/algorithms/ippo.py:40-92` (`SingleAgentBuffer`), `234-356` (`on_env_ready`, `_format_obs`, `act`, `observe`), `418-450` (`_update_agent` batch assembly)
- Modify: `src/commons_game_marp/train/algorithms/mappo.py:200-310` (`on_env_ready`, `_format_local_obs`, `_format_global_obs`, `act`, `observe`), `327-345` (`_update` batch assembly)
- Modify: `src/commons_game_marp/train/console.py:112` (`episode_end`)
- Modify: `src/commons_game_marp/train/trainer.py:41-43, 87-105, 215-441`
- Modify: `tests/conftest.py` (`FakeEnv`)
- Test: `tests/test_algorithm_vec.py`, `tests/test_trainer_parallel.py`

**Interfaces:**
- Consumes: `compute_gae` (Task 1); `VecCommonsEnv`, `rows_to_agents`, `agents_to_rows` (Task 2); `EnvConfig.num_envs`, `resolve_iterations` (Task 3).
- Produces:
  - `Algorithm.act(self, observations: np.ndarray, step: int) -> np.ndarray` — `(N, ...)` in, `(N,)` `int64` out.
  - `Algorithm.observe(self, observations, actions, rewards, next_observations, dones, infos, step) -> None` — all flat rows.
  - `Algorithm.on_env_ready(self, env)` reads `env.agent_ids`, `env.num_envs`, `env.num_agents`, `env.observation_space["curr_obs"].shape`, `env.action_space.n`.
  - `TrainingConsole.episode_end(self, episode: int, stats=None, advance: int = 1)`.
  - `Trainer._make_single_env()` — builds one `HarvestCommonsEnv`, wrapped in `FrameStackEnv` when `num_frames > 1`.

- [ ] **Step 1: Update the shared test double**

`FakeEnv` is what every `on_env_ready` test builds against; its docstring already asks that new duck-typed attributes be added here rather than mocked ad hoc. In `tests/conftest.py`, replace the `FakeEnv` class body with:

```python
class FakeEnv:
    """Minimal stand-in for `VecCommonsEnv`.

    Exists so algorithm construction (`Algorithm.on_env_ready`) can be tested
    without building the full env stack (no PettingZoo/gym env, no rendering,
    no GPU). `on_env_ready` touches exactly these duck-typed attributes:

    - `observation_space["curr_obs"].shape`
    - `action_space.n`
    - `agent_ids`, `num_envs`, `num_agents`

    If `on_env_ready` starts reading anything else from the env, add it here
    -- do not grow this into a general-purpose mock environment.
    """

    observation_space = {"curr_obs": SimpleNamespace(shape=(15, 15, 3))}
    action_space = SimpleNamespace(n=8)
    agent_ids = ["agent-0", "agent-1"]
    num_agents = 2
    num_envs = 1

    def __init__(self, num_envs: int = 1) -> None:
        self.num_envs = num_envs
```

- [ ] **Step 2: Write the failing algorithm tests**

Create `tests/test_algorithm_vec.py`:

```python
"""Every algorithm speaks the flat-row protocol, at num_envs 1 and above.

The protocol is the whole contract between the trainer and the algorithms:
(N, ...) observations in, (N,) int64 actions out, N = num_envs * num_agents in
env-major order. An algorithm that returns the right shape from the wrong
ordering trains happily on mismatched agent data, so ordering is asserted, not
just shape.
"""

import numpy as np
import pytest

from commons_game_marp.train.algorithms.dqn import DQNAlgorithm
from commons_game_marp.train.algorithms.ippo import IPPOAlgorithm
from commons_game_marp.train.algorithms.mappo import MAPPOAlgorithm
from commons_game_marp.train.algorithms.random_policy import RandomAlgorithm
from commons_game_marp.train.config import (
    DQNConfig,
    IPPOConfig,
    MAPPOConfig,
    RandomConfig,
)
from tests.conftest import FakeEnv


def build(algo_cls, config, num_envs):
    env = FakeEnv(num_envs=num_envs)
    algo = algo_cls(config)
    algo.on_env_ready(env)
    return algo, env


ALGORITHMS = [
    (IPPOAlgorithm, lambda: IPPOConfig(n_steps=4, batch_size=4, update_epochs=1, device="cpu")),
    (MAPPOAlgorithm, lambda: MAPPOConfig(n_steps=4, batch_size=4, update_epochs=1, device="cpu")),
    (DQNAlgorithm, lambda: DQNConfig(device="cpu", train_after=10_000)),
    (RandomAlgorithm, lambda: RandomConfig(device="cpu")),
]


@pytest.mark.parametrize("algo_cls,make_config", ALGORITHMS)
@pytest.mark.parametrize("num_envs", [1, 3])
def test_act_returns_one_int64_action_per_row(algo_cls, make_config, num_envs):
    algo, env = build(algo_cls, make_config(), num_envs)
    rows = num_envs * env.num_agents
    obs = np.zeros((rows, 15, 15, 3), dtype=np.uint8)

    actions = algo.act(obs, step=0)

    assert isinstance(actions, np.ndarray)
    assert actions.shape == (rows,)
    assert actions.dtype == np.int64
    assert np.all((actions >= 0) & (actions < env.action_space.n))


@pytest.mark.parametrize("algo_cls,make_config", ALGORITHMS)
@pytest.mark.parametrize("num_envs", [1, 3])
def test_observe_accepts_flat_rows(algo_cls, make_config, num_envs):
    algo, env = build(algo_cls, make_config(), num_envs)
    rows = num_envs * env.num_agents
    obs = np.zeros((rows, 15, 15, 3), dtype=np.uint8)
    actions = algo.act(obs, step=0)

    algo.observe(
        observations=obs,
        actions=actions,
        rewards=np.zeros(rows, dtype=np.float32),
        next_observations=obs,
        dones=np.zeros(rows, dtype=bool),
        infos=[{} for _ in range(rows)],
        step=0,
    )


def test_ippo_routes_each_agents_observations_to_its_own_actor():
    """Agent-0's actor must never see agent-1's rows.

    Both actors are replaced with deterministic stand-ins keyed to a marker
    value planted in each agent's observations; a transposed reshape would
    hand the markers to the wrong actor.
    """
    import torch
    from torch import nn

    algo, env = build(IPPOAlgorithm, IPPOConfig(device="cpu", flatten_obs=True), 3)

    class MarkerActor(nn.Module):
        def __init__(self, expected: float, num_actions: int) -> None:
            super().__init__()
            self.expected = expected
            self.num_actions = num_actions
            self.seen = []

        def forward(self, obs: torch.Tensor) -> torch.Tensor:
            self.seen.append(float(obs.flatten()[0].item()))
            return torch.zeros(obs.shape[0], self.num_actions)

    actors = {
        "agent-0": MarkerActor(1.0, env.action_space.n),
        "agent-1": MarkerActor(2.0, env.action_space.n),
    }
    algo.actors = actors

    obs = np.zeros((6, 15, 15, 3), dtype=np.uint8)
    for env_idx in range(3):
        obs[env_idx * 2 + 0] = 1  # agent-0 rows
        obs[env_idx * 2 + 1] = 2  # agent-1 rows

    algo.act(obs, step=0)

    assert actors["agent-0"].seen and actors["agent-1"].seen
    assert actors["agent-0"].seen[0] != actors["agent-1"].seen[0]


def test_ippo_buffer_keeps_one_column_per_env():
    algo, env = build(
        IPPOAlgorithm, IPPOConfig(n_steps=1000, device="cpu"), num_envs=3
    )
    rows = 6
    obs = np.zeros((rows, 15, 15, 3), dtype=np.uint8)

    for step in range(4):
        actions = algo.act(obs, step=step)
        algo.observe(
            obs, actions, np.zeros(rows, dtype=np.float32), obs,
            np.zeros(rows, dtype=bool), [{}] * rows, step,
        )

    buffer = algo.buffers["agent-0"]
    assert buffer.size() == 4, "size counts per-env timesteps, not rows"
    assert np.stack(buffer.rewards, axis=0).shape == (4, 3)


def test_ippo_updates_after_n_steps_per_env_not_per_row():
    """`n_steps` is per-env, so the update batch is n_steps * num_envs."""
    algo, env = build(
        IPPOAlgorithm,
        IPPOConfig(n_steps=3, batch_size=2, update_epochs=1, device="cpu"),
        num_envs=2,
    )
    rows = 4
    obs = np.zeros((rows, 15, 15, 3), dtype=np.uint8)

    for step in range(2):
        actions = algo.act(obs, step=step)
        algo.observe(
            obs, actions, np.zeros(rows, dtype=np.float32), obs,
            np.zeros(rows, dtype=bool), [{}] * rows, step,
        )
    assert algo.buffers["agent-0"].size() == 2, "must not update before n_steps"

    actions = algo.act(obs, step=2)
    algo.observe(
        obs, actions, np.zeros(rows, dtype=np.float32), obs,
        np.zeros(rows, dtype=bool), [{}] * rows, 2,
    )
    assert algo.buffers["agent-0"].size() == 0, "buffer clears after the update"
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `uv run pytest tests/test_algorithm_vec.py -v`
Expected: FAIL — algorithms still return dicts, and `FakeEnv(num_envs=...)` attributes are unused.

- [ ] **Step 4: Update the abstract interface**

Replace `src/commons_game_marp/train/algorithms/base.py` with:

```python
from abc import ABC, abstractmethod
from typing import Any, Dict, List

import numpy as np


class Algorithm(ABC):
    """Policy interface over a flat batch of `num_envs * num_agents` rows.

    Every array is ordered env-major, agent-minor -- row = env_idx *
    num_agents + agent_idx -- matching `VecCommonsEnv`. There is no separate
    single-environment path: `num_envs=1` is the same code, so the path every
    existing config uses is the path the tests cover.
    """

    def __init__(self, config: Any):
        self.config = config

    @abstractmethod
    def on_env_ready(self, env) -> None:
        """Build networks and buffers.

        `env` is a `VecCommonsEnv`, exposing `agent_ids`, `num_envs`,
        `num_agents`, `observation_space` and `action_space`.
        """

    @abstractmethod
    def act(self, observations: np.ndarray, step: int) -> np.ndarray:
        """(N, ...) observations -> (N,) int64 actions."""

    @abstractmethod
    def observe(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_observations: np.ndarray,
        dones: np.ndarray,
        infos: List[Dict[str, Any]],
        step: int,
    ) -> None:
        """Record one transition for every row."""

    @abstractmethod
    def on_episode_end(self, episode: int) -> Dict[str, Any]:
        pass

    def uses_external_loop(self) -> bool:
        return True

    def save(self, path: str) -> None:
        return None
```

- [ ] **Step 5: Convert the random policy**

Replace `src/commons_game_marp/train/algorithms/random_policy.py` with:

```python
from typing import Any, Dict, List

import numpy as np

from .base import Algorithm


class RandomAlgorithm(Algorithm):
    def __init__(self, config: Any):
        super().__init__(config)
        self._env = None

    def on_env_ready(self, env) -> None:
        self._env = env

    def act(self, observations: np.ndarray, step: int) -> np.ndarray:
        return np.array(
            [self._env.action_space.sample() for _ in range(observations.shape[0])],
            dtype=np.int64,
        )

    def observe(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_observations: np.ndarray,
        dones: np.ndarray,
        infos: List[Dict[str, Any]],
        step: int,
    ) -> None:
        return None

    def on_episode_end(self, episode: int) -> Dict[str, Any]:
        return {}
```

- [ ] **Step 6: Convert DQN**

In `src/commons_game_marp/train/algorithms/dqn.py`, add to the imports:

```python
from ...env.vec_env import agents_to_rows, rows_to_agents
```

Replace `DQNAlgorithm.on_env_ready`, `_format_obs`, `act` and `observe` (currently `dqn.py:134-188`) with:

```python
    def on_env_ready(self, env) -> None:
        obs_space = env.observation_space["curr_obs"]
        obs_shape = obs_space.shape
        num_actions = int(env.action_space.n)
        self.agent_ids = list(env.agent_ids)
        self.num_envs = int(env.num_envs)
        for agent_id in self.agent_ids:
            self.agents[agent_id] = DQNAgent(
                obs_shape=obs_shape,
                num_actions=num_actions,
                learning_rate=self.config.learning_rate,
                gamma=self.config.gamma,
                epsilon_start=self.config.epsilon_start,
                epsilon_end=self.config.epsilon_end,
                epsilon_decay=self.config.epsilon_decay,
                batch_size=self.config.batch_size,
                replay_buffer_size=self.config.replay_buffer_size,
                target_update_freq=self.config.target_update_freq,
                train_after=self.config.train_after,
                train_every=self.config.train_every,
                max_grad_norm=self.config.max_grad_norm,
                device=self.config.device,
            )

    def _format_obs(self, obs_batch: np.ndarray) -> np.ndarray:
        """(num_envs, ...) uint8 rows -> float32, scaled as configured."""
        img = obs_batch.astype(np.float32)
        if self.config.normalize_obs:
            img = img / 255.0
        return img

    def act(self, observations: np.ndarray, step: int) -> np.ndarray:
        per_agent = rows_to_agents(observations, self.num_envs, self.agent_ids)
        actions = {}
        for agent_id, agent in self.agents.items():
            obs_batch = self._format_obs(per_agent[agent_id])
            actions[agent_id] = np.array(
                [agent.act(obs_batch[e], training=True) for e in range(self.num_envs)],
                dtype=np.int64,
            )
        return agents_to_rows(actions, self.num_envs, self.agent_ids)

    def observe(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_observations: np.ndarray,
        dones: np.ndarray,
        infos: List[Dict[str, Any]],
        step: int,
    ) -> None:
        obs_by_agent = rows_to_agents(observations, self.num_envs, self.agent_ids)
        next_by_agent = rows_to_agents(next_observations, self.num_envs, self.agent_ids)
        act_by_agent = rows_to_agents(actions, self.num_envs, self.agent_ids)
        rew_by_agent = rows_to_agents(rewards, self.num_envs, self.agent_ids)
        done_by_agent = rows_to_agents(dones, self.num_envs, self.agent_ids)

        for agent_id, agent in self.agents.items():
            obs_batch = self._format_obs(obs_by_agent[agent_id])
            next_batch = self._format_obs(next_by_agent[agent_id])
            # One transition per environment: the replay buffer takes all
            # num_envs of them, which is the whole benefit DQN gets here.
            for e in range(self.num_envs):
                agent.remember(
                    obs_batch[e],
                    int(act_by_agent[agent_id][e]),
                    float(rew_by_agent[agent_id][e]),
                    next_batch[e],
                    bool(done_by_agent[agent_id][e]),
                )
                loss_info = agent.train_step()
                if loss_info.get("loss") is not None:
                    self.last_losses[agent_id] = loss_info["loss"]
```

Also add `self.agent_ids: List[str] = []` and `self.num_envs = 1` to `DQNAlgorithm.__init__` beside the existing `self.agents` and `self.last_losses`.

- [ ] **Step 7: Convert IPPO's buffer to carry an env axis**

In `src/commons_game_marp/train/algorithms/ippo.py`, add to the imports:

```python
from ...env.vec_env import agents_to_rows, rows_to_agents
from .gae import compute_gae
```

Replace `SingleAgentBuffer` (currently `ippo.py:40-92`) with:

```python
class SingleAgentBuffer:
    """Rollout buffer for one agent across `num_envs` parallel environments.

    Every entry carries a leading env axis, so the stored arrays are (T,
    num_envs). `size()` reports T -- per-env timesteps -- which is what
    `n_steps` counts, following SB3's convention; the update batch is
    therefore n_steps * num_envs.
    """

    def __init__(self, num_envs: int = 1) -> None:
        self.num_envs = num_envs
        self.clear()

    def clear(self) -> None:
        self.obs: List[np.ndarray] = []
        self.actions: List[np.ndarray] = []
        self.logprobs: List[np.ndarray] = []
        self.rewards: List[np.ndarray] = []
        self.dones: List[np.ndarray] = []
        self.values: List[np.ndarray] = []
        self.next_values: List[np.ndarray] = []

    def add_step(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        logprob: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        value: np.ndarray,
        next_value: np.ndarray,
    ) -> None:
        """Record one timestep. `obs` is (num_envs, ...); the rest (num_envs,)."""
        self.obs.append(obs)
        self.actions.append(action)
        self.logprobs.append(logprob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)
        self.next_values.append(next_value)

    def size(self) -> int:
        """Timesteps per environment, not rows."""
        return len(self.rewards)

    def compute_advantages(
        self, gamma: float, gae_lambda: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        return compute_gae(
            rewards=np.stack(self.rewards, axis=0),
            dones=np.stack(self.dones, axis=0),
            values=np.stack(self.values, axis=0),
            next_values=np.stack(self.next_values, axis=0),
            gamma=gamma,
            gae_lambda=gae_lambda,
        )
```

- [ ] **Step 8: Convert IPPO's interface**

In `src/commons_game_marp/train/algorithms/ippo.py`, add `self.num_envs = 1` and `self.num_agents = 0` to `__init__` beside `self.agent_ids`.

In `on_env_ready` (currently `ippo.py:234-274`), replace the first four lines with:

```python
        obs_space = env.observation_space["curr_obs"]
        self.obs_shape = obs_space.shape
        self.num_actions = int(env.action_space.n)
        self.agent_ids = list(env.agent_ids)
        self.num_envs = int(env.num_envs)
        self.num_agents = len(self.agent_ids)
```

and change the buffer construction inside the per-agent loop to:

```python
            self.buffers[agent_id] = SingleAgentBuffer(self.num_envs)
```

Replace `_format_obs`, `act` and `observe` (currently `ippo.py:276-356`) with:

```python
    def _format_obs(self, obs_batch: np.ndarray) -> np.ndarray:
        """One agent's (num_envs, ...) uint8 rows -> the policy's input dtype."""
        img = obs_batch.astype(np.float32)
        if self.config.normalize_obs:
            img = img / 255.0
        if self.config.flatten_obs:
            return img.reshape(obs_batch.shape[0], -1)
        return img

    def act(self, observations: np.ndarray, step: int) -> np.ndarray:
        per_agent = rows_to_agents(observations, self.num_envs, self.agent_ids)
        actions: Dict[str, np.ndarray] = {}
        self._last_step = {}

        for agent_id in self.agent_ids:
            obs = self._format_obs(per_agent[agent_id])
            obs_tensor = torch.from_numpy(obs).float().to(self.device)

            with torch.no_grad():
                logits = self.actors[agent_id](obs_tensor)
                dist = Categorical(logits=logits)
                action = dist.sample()
                logprob = dist.log_prob(action)
                value = self.critics[agent_id](obs_tensor)

            actions_np = action.cpu().numpy().astype(np.int64)
            actions[agent_id] = actions_np
            self._last_step[agent_id] = {
                "obs": obs,
                "action": actions_np,
                "logprob": logprob.cpu().numpy().astype(np.float32),
                "value": value.cpu().numpy().astype(np.float32),
            }

        return agents_to_rows(actions, self.num_envs, self.agent_ids)

    def observe(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_observations: np.ndarray,
        dones: np.ndarray,
        infos: List[Dict[str, Any]],
        step: int,
    ) -> None:
        if not self._last_step:
            return

        next_by_agent = rows_to_agents(next_observations, self.num_envs, self.agent_ids)
        rew_by_agent = rows_to_agents(rewards, self.num_envs, self.agent_ids)
        done_by_agent = rows_to_agents(dones, self.num_envs, self.agent_ids)

        for agent_id in self.agent_ids:
            if agent_id not in self._last_step:
                continue

            last = self._last_step[agent_id]
            next_obs = self._format_obs(next_by_agent[agent_id])
            next_obs_tensor = torch.from_numpy(next_obs).float().to(self.device)
            with torch.no_grad():
                next_value = (
                    self.critics[agent_id](next_obs_tensor)
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )

            self.buffers[agent_id].add_step(
                obs=last["obs"],
                action=last["action"],
                logprob=last["logprob"],
                reward=np.asarray(rew_by_agent[agent_id], dtype=np.float32),
                done=np.asarray(done_by_agent[agent_id], dtype=np.float32),
                value=last["value"],
                next_value=next_value,
            )

        self._last_step = {}

        # The `or done_all` trigger the dict interface carried is gone: episodes
        # are lockstep and fixed-length, and `on_episode_end` already flushes
        # whatever is left in the buffers at the boundary.
        min_buffer_size = min(buf.size() for buf in self.buffers.values())
        if min_buffer_size >= self.config.n_steps:
            self._update_all()
```

- [ ] **Step 9: Flatten IPPO's update batch across the env axis**

In `_update_agent` (currently `ippo.py:418-450`), replace the batch assembly — from `obs = np.stack(buffer.obs, axis=0)` down to and including `T = len(buffer.obs)` and `batch_size = min(...)` — with:

```python
        advantages, returns = buffer.compute_advantages(
            gamma=self.config.gamma, gae_lambda=self.config.gae_lambda
        )

        # (T, num_envs, ...) -> (T * num_envs, ...). Advantages were computed
        # per column first, so no gradient path crosses an environment.
        stacked_obs = np.stack(buffer.obs, axis=0)
        T, E = stacked_obs.shape[0], stacked_obs.shape[1]
        total = T * E
        obs = stacked_obs.reshape(total, *stacked_obs.shape[2:])
        actions = np.stack(buffer.actions, axis=0).reshape(total).astype(np.int64)
        old_logprobs = np.stack(buffer.logprobs, axis=0).reshape(total).astype(np.float32)
        old_values = np.stack(buffer.values, axis=0).reshape(total).astype(np.float32)
        advantages = advantages.reshape(total)
        returns = returns.reshape(total)

        # Normalize advantages
        adv_mean = advantages.mean()
        adv_std = advantages.std() + 1e-8
        advantages = (advantages - adv_mean) / adv_std

        obs_tensor = torch.from_numpy(obs).float().to(self.device)
        actions_tensor = torch.from_numpy(actions).long().to(self.device)
        old_logprobs_tensor = torch.from_numpy(old_logprobs).float().to(self.device)
        old_values_tensor = torch.from_numpy(old_values).float().to(self.device)
        advantages_tensor = torch.from_numpy(advantages).float().to(self.device)
        returns_tensor = torch.from_numpy(returns).float().to(self.device)

        batch_size = min(int(self.config.batch_size), total)
```

Then in the two minibatch loops below, replace `np.random.permutation(T)` with `np.random.permutation(total)` and `for start in range(0, T, batch_size)` with `for start in range(0, total, batch_size)`. Delete the now-duplicated advantage-normalisation and tensor-construction lines that followed the old assembly.

- [ ] **Step 10: Convert MAPPO**

In `src/commons_game_marp/train/algorithms/mappo.py`, add to the imports:

```python
from ...env.vec_env import rows_to_agents
```

Add `self.num_envs = 1` to `MAPPOAlgorithm.__init__` beside `self.agent_ids`, and in `on_env_ready` replace the agent-id line with:

```python
        self.agent_ids = list(env.agent_ids)
        self.num_envs = int(env.num_envs)
```

Replace `_format_local_obs`, `_format_global_obs`, `act` and `observe` (currently `mappo.py:228-310`) with:

```python
    def _format_local_obs(self, obs_batch: np.ndarray) -> np.ndarray:
        """One agent's (num_envs, ...) uint8 rows -> the policy's input dtype."""
        img = obs_batch.astype(np.float32)
        if self.config.normalize_obs:
            img = img / 255.0
        if self.config.flatten_obs:
            return img.reshape(obs_batch.shape[0], -1)
        return img

    def _format_global_obs(self, rows: np.ndarray) -> np.ndarray:
        """Centralized state per environment: every agent's view, concatenated.

        Returns (num_envs, obs_dim * num_agents) when flattened, otherwise
        (num_envs, H, W, C * num_agents). The concatenation axis is one higher
        than in the single-env form because of the leading env axis.
        """
        per_agent = rows_to_agents(rows, self.num_envs, self.agent_ids)
        imgs = [self._format_local_obs(per_agent[a]) for a in self.agent_ids]
        if self.config.flatten_obs:
            return np.concatenate(imgs, axis=1)
        return np.concatenate(imgs, axis=3)

    def _local_rows(self, rows: np.ndarray) -> np.ndarray:
        """Formatted local observations, back in env-major row order."""
        per_agent = rows_to_agents(rows, self.num_envs, self.agent_ids)
        stacked = np.stack(
            [self._format_local_obs(per_agent[a]) for a in self.agent_ids], axis=1
        )
        return stacked.reshape(
            self.num_envs * len(self.agent_ids), *stacked.shape[2:]
        )

    def act(self, observations: np.ndarray, step: int) -> np.ndarray:
        local_rows = self._local_rows(observations)
        global_obs = self._format_global_obs(observations)
        obs_tensor = torch.from_numpy(local_rows).float().to(self.device)
        global_tensor = torch.from_numpy(global_obs).float().to(self.device)

        with torch.no_grad():
            logits = self.actor(obs_tensor)
            dist = Categorical(logits=logits)
            actions = dist.sample()
            logprobs = dist.log_prob(actions)
            values = self.critic(global_tensor)  # (num_envs, num_agents)

        actions_np = actions.cpu().numpy().astype(np.int64)
        self._last_step = {
            "local_obs": local_rows,
            "global_obs": global_obs,
            "actions": actions_np,
            "logprobs": logprobs.cpu().numpy().astype(np.float32),
            # (num_envs, num_agents) -> rows, which is already env-major.
            "values": values.cpu().numpy().reshape(-1).astype(np.float32),
        }
        return actions_np

    def observe(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_observations: np.ndarray,
        dones: np.ndarray,
        infos: List[Dict[str, Any]],
        step: int,
    ) -> None:
        if not self._last_step:
            return

        next_global = self._format_global_obs(next_observations)
        next_global_tensor = torch.from_numpy(next_global).float().to(self.device)
        with torch.no_grad():
            next_values = (
                self.critic(next_global_tensor).cpu().numpy().reshape(-1).astype(np.float32)
            )

        self.buffer.add_step(
            local_obs=self._last_step["local_obs"],
            global_obs=self._last_step["global_obs"],
            actions=self._last_step["actions"],
            logprobs=self._last_step["logprobs"],
            rewards=np.asarray(rewards, dtype=np.float32),
            dones=np.asarray(dones, dtype=np.float32),
            values=self._last_step["values"],
            next_values=next_values,
        )

        self._last_step = {}
        # Lockstep episodes: `on_episode_end` flushes the boundary, so the
        # per-env timestep count is the only trigger needed here.
        if self.buffer.size() >= self.config.n_steps:
            self._update()
            self.buffer.clear()
```

- [ ] **Step 11: Widen MAPPO's update batch**

In `_update` (currently `mappo.py:327-345`), replace the batch assembly from `local_obs = np.stack(...)` through `agent_idx = np.tile(...)` with:

```python
        local_obs = np.stack(self.buffer.local_obs, axis=0)     # (T, E*A, ...)
        global_obs = np.stack(self.buffer.global_obs, axis=0)   # (T, E, ...)
        actions = np.stack(self.buffer.actions, axis=0)
        logprobs = np.stack(self.buffer.logprobs, axis=0)
        values = np.stack(self.buffer.values, axis=0)
        T, N = actions.shape
        num_agents = len(self.agent_ids)

        local_obs = local_obs.reshape(T * N, *local_obs.shape[2:])
        # Each env's centralized state serves that env's num_agents rows, which
        # sit consecutively because rows are env-major/agent-minor.
        global_obs = np.repeat(
            global_obs.reshape(T * self.num_envs, *global_obs.shape[2:]),
            num_agents,
            axis=0,
        )
        actions = actions.reshape(T * N)
        logprobs = logprobs.reshape(T * N)
        values = values.reshape(T * N)
        advantages = advantages.reshape(T * N)
        returns = returns.reshape(T * N)
        # The critic head is indexed by agent, and a row's agent is row % A.
        agent_idx = np.tile(np.arange(num_agents, dtype=np.int64), T * self.num_envs)
```

- [ ] **Step 12: Run the algorithm tests**

Run: `uv run pytest tests/test_algorithm_vec.py -v`
Expected: PASS, 21 tests.

- [ ] **Step 13: Let the console advance by more than one episode at a time**

In `src/commons_game_marp/train/console.py`, change the `episode_end` signature and the bar update:

```python
    def episode_end(
        self,
        episode: int,
        stats: Optional[Mapping[str, Any]] = None,
        advance: int = 1,
    ) -> None:
        """Report finished episodes. `episode` is the 0-based index of the last
        one completed; `advance` is how many completed at once, which is
        `num_envs` when environments run in parallel."""
        if not self.enabled:
            return
        summary = format_metrics(stats)
        if self._bar is not None:
            if summary:
                self._bar.set_postfix_str(summary, refresh=False)
            self._bar.update(advance)
            return
```

The rest of the method is unchanged.

- [ ] **Step 14: Write the failing trainer test**

Create `tests/test_trainer_parallel.py`:

```python
"""End-to-end: a short run must complete on every algorithm, at num_envs 1 and
above, and must log the episode budget it was asked for.

A vec trainer that quietly runs `episodes` iterations instead of `episodes //
num_envs` would 4x the compute of every existing config without failing
anything, so the logged episode count is asserted directly.
"""

import json
import os

import pytest

from commons_game_marp.train.config import (
    DQNConfig,
    EnvConfig,
    IPPOConfig,
    LoggingConfig,
    MAPPOConfig,
    RandomConfig,
    RewardModelConfig,
    TrainerConfig,
)
from commons_game_marp.train.trainer import Trainer


def make_config(tmp_path, algorithm, num_envs, episodes=4, reward_model=False):
    return TrainerConfig(
        episodes=episodes,
        seed=0,
        env=EnvConfig(
            map_type="small",
            num_agents=2,
            agent_view_range=3,
            ep_length=5,
            num_envs=num_envs,
        ),
        algorithm=algorithm,
        logging=LoggingConfig(
            log_dir=str(tmp_path),
            run_dir=str(tmp_path / "run"),
            console="quiet",
            video_enabled=False,
            log_agent_episode_details=True,
        ),
        reward_model=RewardModelConfig(
            enabled=reward_model,
            warmup_episodes=1,
            update_every_env_steps=1,
            batch_pairs=2,
            train_steps_per_update=1,
            max_episodes_in_buffer=8,
            device="cpu",
        ),
    )


ALGORITHMS = [
    ("ippo", lambda: IPPOConfig(n_steps=4, batch_size=4, update_epochs=1, device="cpu")),
    ("mappo", lambda: MAPPOConfig(n_steps=4, batch_size=4, update_epochs=1, device="cpu")),
    ("dqn", lambda: DQNConfig(device="cpu", train_after=10_000)),
    ("random", lambda: RandomConfig(device="cpu")),
]


def read_episodes(run_dir):
    path = os.path.join(run_dir, "metrics.jsonl")
    with open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


@pytest.mark.parametrize("name,make_algorithm", ALGORITHMS)
@pytest.mark.parametrize("num_envs", [1, 2])
def test_short_run_completes(tmp_path, name, make_algorithm, num_envs):
    config = make_config(tmp_path, make_algorithm(), num_envs)
    Trainer(config).train()

    records = read_episodes(tmp_path / "run")
    assert records, "the run logged nothing"


def test_episode_budget_is_spent_not_multiplied(tmp_path):
    config = make_config(
        tmp_path, IPPOConfig(n_steps=4, batch_size=4, update_epochs=1, device="cpu"),
        num_envs=2, episodes=4,
    )
    Trainer(config).train()

    records = read_episodes(tmp_path / "run")
    assert len(records) == 2, "4 episodes at num_envs=2 is 2 iterations"
    # `episode` stays an episode count so curves overlay across num_envs.
    assert [r["episode"] for r in records] == [1, 3]
    assert all(r["num_envs"] == 2 for r in records)


def test_non_divisible_episode_budget_fails_at_construction(tmp_path):
    config = make_config(
        tmp_path, RandomConfig(device="cpu"), num_envs=3, episodes=4
    )
    with pytest.raises(ValueError, match="divisible"):
        Trainer(config)


def test_reward_model_run_completes_with_parallel_envs(tmp_path):
    config = make_config(
        tmp_path, IPPOConfig(n_steps=4, batch_size=4, update_epochs=1, device="cpu"),
        num_envs=2, reward_model=True,
    )
    Trainer(config).train()

    records = read_episodes(tmp_path / "run")
    assert all("reward_pred_mean" in r for r in records)


def test_agent_detail_logs_cover_env_zero_only(tmp_path):
    """One episode's detail file per agent per iteration, not num_envs of them."""
    config = make_config(
        tmp_path, RandomConfig(device="cpu"), num_envs=2, episodes=4
    )
    Trainer(config).train()

    details_dir = os.path.join(tmp_path, "run", "agent_episodes")
    assert os.path.isdir(details_dir)
    files = [f for f in os.listdir(details_dir) if f.endswith(".jsonl")]
    assert files, "no per-agent detail files written"
```

Before running it, confirm the detail-log directory name and `metrics.jsonl` filename against `src/commons_game_marp/train/logging_utils.py` and adjust `read_episodes`/`details_dir` to match what `ResultLogger` actually writes.

- [ ] **Step 15: Rewrite the trainer**

In `src/commons_game_marp/train/trainer.py`, add the imports:

```python
from ..env.vec_env import VecCommonsEnv
from .config import TrainerConfig, resolve_iterations
```

Replace `_build_env` (currently `trainer.py:87-105`) with:

```python
    def _make_single_env(self):
        """Build one environment copy, exactly as configured."""
        env_cfg = self.config.env
        ascii_map = MAP[env_cfg.map_type]
        env = HarvestCommonsEnv(
            ascii_map=ascii_map,
            num_agents=env_cfg.num_agents,
            render=env_cfg.render,
            agent_view_range=env_cfg.agent_view_range,
            ep_length=env_cfg.ep_length,
            spawn_speed=env_cfg.spawn_speed,
            metric=env_cfg.metric,
            penalty=env_cfg.penalty,
        )
        num_frames = int(env_cfg.num_frames)
        if num_frames < 1:
            raise ValueError(f"env.num_frames must be >= 1, got {num_frames}")
        if num_frames > 1:
            return FrameStackEnv(env, num_frames)
        return env

    def _build_env(self) -> VecCommonsEnv:
        return VecCommonsEnv(self._make_single_env, int(self.config.env.num_envs))
```

In `__init__`, add the iteration resolution before `self.env = self._build_env()` so a bad pair fails before any environment is constructed:

```python
        self.iterations = resolve_iterations(
            self.config.episodes, int(self.config.env.num_envs)
        )
        self.env = self._build_env()
```

Add a line to `_announce_setup` after the `environment` line:

```python
        if env_cfg.num_envs > 1:
            self.console.info(
                f"parallel envs  : {env_cfg.num_envs}"
                f" ({self.iterations} iterations x {env_cfg.num_envs} episodes)"
            )
```

Replace `_format_reward_obs` (currently `trainer.py:199-209`) with a row-based version:

```python
    def _format_reward_obs(self, observations: np.ndarray, row: int) -> np.ndarray:
        """Format one row's observation for the reward model.

        Returns the frame in its native `uint8` dtype. Scaling happens inside
        `RewardModel.forward`, on device, via `obs_scale` -- so the preference
        buffer holds a quarter of the bytes a float32 copy would need, and the
        same saving applies to every host-to-device transfer. The effective
        input scale is unchanged: `_reward_obs_scale() * frame`.
        """
        return np.ascontiguousarray(observations[row])
```

Add a module-level helper beside the class:

```python
def _average_social_metrics(per_env: list) -> dict:
    """Mean of each social metric across the environments of one iteration."""
    if not per_env:
        return {}
    return {
        key: float(np.mean([metrics[key] for metrics in per_env]))
        for key in per_env[0]
    }
```

Replace the training loop (currently `trainer.py:266-419`, from `self.console.section("Training")` to the end of the episode loop) with:

```python
        self.console.section("Training")
        self.console.start_episodes(self.config.episodes)
        recent_rewards: list = []

        vec = self.env
        num_envs = vec.num_envs
        num_agents = vec.num_agents
        agent_ids = vec.agent_ids
        num_rows = vec.num_rows

        for iteration in range(self.iterations):
            # `episode` stays an episode count, not an iteration count, so runs
            # with different num_envs overlay on one x-axis.
            episode = (iteration + 1) * num_envs - 1

            obs, infos = vec.reset()
            episode_rewards = np.zeros(num_rows, dtype=np.float64)
            episode_pred_rewards = (
                np.zeros(num_rows, dtype=np.float64) if rm_cfg.enabled else None
            )
            # One trajectory set per environment: each becomes its own
            # EpisodeRecord, so the preference buffer keeps exact episodes.
            episode_agent_trajs = (
                [{agent_id: [] for agent_id in agent_ids} for _ in range(num_envs)]
                if rm_cfg.enabled
                else None
            )
            # Per-step detail is env 0 only: logging all num_envs would multiply
            # the JSON volume for data already summarized in metrics.jsonl.
            agent_episode_details = (
                {agent_id: [] for agent_id in agent_ids}
                if self.config.logging.log_agent_episode_details
                else None
            )
            step_count = 0
            video_recorder.start(episode)

            for step in range(self.config.env.ep_length):
                actions = self.algorithm.act(obs, step)
                next_obs, rewards, dones, infos = vec.step(actions)

                if rm_cfg.enabled:
                    obs_frames = [self._format_reward_obs(obs, r) for r in range(num_rows)]
                    predicted = reward_model.predict_batch(
                        np.stack(obs_frames, axis=0), [int(a) for a in actions]
                    )
                    pred_rewards = np.asarray(predicted, dtype=np.float32).reshape(num_rows)
                    for row in range(num_rows):
                        env_idx, agent_idx = divmod(row, num_agents)
                        episode_agent_trajs[env_idx][agent_ids[agent_idx]].append(
                            (obs_frames[row], int(actions[row]))
                        )
                    episode_pred_rewards += pred_rewards
                    self.algorithm.observe(
                        obs, actions, pred_rewards, next_obs, dones, infos, step
                    )
                else:
                    pred_rewards = None
                    self.algorithm.observe(
                        obs, actions, rewards, next_obs, dones, infos, step
                    )

                video_recorder.record(vec.envs[0], step)
                episode_rewards += rewards

                if agent_episode_details is not None:
                    for agent_idx, agent_id in enumerate(agent_ids):
                        row = agent_idx  # env 0
                        apple_eaten = bool(rewards[row] > 0)
                        agent = vec.envs[0].agents[agent_id]
                        metrics = compute_agent_step_metrics(
                            agent=agent,
                            env=vec.envs[0],
                            reward=float(rewards[row]),
                            apple_eaten=apple_eaten,
                            nearby_radius=2,
                        )
                        step_data = {
                            "step": step,
                            "action": int(actions[row]),
                            "reward": float(rewards[row]),
                            "done": bool(dones[row]),
                            "apple_eaten": apple_eaten,
                            "nearby_apples": metrics["nearby_apples"],
                            "ate_last_apple_in_cluster": metrics["ate_last_apple_in_cluster"],
                        }
                        if pred_rewards is not None:
                            step_data["predicted_reward"] = float(pred_rewards[row])
                        agent_episode_details[agent_id].append(step_data)

                obs = next_obs
                step_count = step + 1
                global_step += num_envs

            per_env_metrics = vec.compute_social_metrics()
            metrics = _average_social_metrics(per_env_metrics)
            if rm_cfg.enabled:
                for env_idx in range(num_envs):
                    pref_buffer.add_episode(
                        EpisodeRecord(
                            agent_trajs=episode_agent_trajs[env_idx],
                            metrics=per_env_metrics[env_idx],
                        )
                    )

            rm_metrics = {}
            if rm_cfg.enabled and (episode + 1) >= rm_cfg.warmup_episodes:
                self.console.info_once(
                    "rm-warmup-done",
                    f"warmup complete after episode {episode + 1}"
                    f" -- reward model training starts, buffer holds {len(pref_buffer)} episodes",
                )
                if (global_step - last_rm_update_step) >= rm_cfg.update_every_env_steps:
                    rm_metrics = rm_trainer.train(
                        pref_buffer,
                        phi_key=rm_cfg.phi,
                        mode=rm_cfg.mode,
                        batch_pairs=rm_cfg.batch_pairs,
                        train_steps=rm_cfg.train_steps_per_update,
                    )
                    last_rm_update_step = global_step
                    self.console.info_once(
                        "rm-first-update",
                        "first reward model update: "
                        + (_format_metrics(rm_metrics) or "no metrics returned"),
                    )
            if rm_cfg.enabled and (episode + 1) % rm_cfg.save_every_episodes == 0:
                reward_model_path = os.path.join(self.logger.run_dir, "reward_model.pt")
                reward_model.save(reward_model_path)
                self.console.info(f"reward model checkpoint saved at episode {episode + 1}")

            algo_metrics = self.algorithm.on_episode_end(iteration)
            if algo_metrics is None:
                algo_metrics = {}
            if rm_metrics:
                algo_metrics = dict(algo_metrics)
                algo_metrics["reward_model"] = rm_metrics
            self._watch_entropy_saturation(algo_metrics)

            # Averaged across the num_envs episodes this iteration completed.
            by_env_agent = episode_rewards.reshape(num_envs, num_agents)
            reward_per_agent = {
                agent_id: float(by_env_agent[:, i].mean())
                for i, agent_id in enumerate(agent_ids)
            }
            payload = {
                "episode": episode,
                "num_envs": num_envs,
                "steps": step_count,
                "reward_sum": float(by_env_agent.sum(axis=1).mean()),
                "reward_mean": float(episode_rewards.mean()),
                "reward_per_agent": reward_per_agent,
                "social_metrics": metrics,
                "algo_metrics": algo_metrics,
            }
            if rm_cfg.enabled and episode_pred_rewards is not None:
                pred_by_env_agent = episode_pred_rewards.reshape(num_envs, num_agents)
                payload["reward_pred_sum"] = float(pred_by_env_agent.sum(axis=1).mean())
                payload["reward_pred_mean"] = float(episode_pred_rewards.mean())
                payload["reward_pred_per_agent"] = {
                    agent_id: float(pred_by_env_agent[:, i].mean())
                    for i, agent_id in enumerate(agent_ids)
                }
            if iteration % self.config.logging.log_interval == 0:
                self.logger.log_episode(payload)

            if agent_episode_details is not None:
                for agent_id, details in agent_episode_details.items():
                    episode_summary = {
                        "total_steps": step_count,
                        "total_reward": float(by_env_agent[0, agent_ids.index(agent_id)]),
                        "steps": details,
                    }
                    if rm_cfg.enabled and episode_pred_rewards is not None:
                        episode_summary["total_predicted_reward"] = float(
                            episode_pred_rewards.reshape(num_envs, num_agents)[
                                0, agent_ids.index(agent_id)
                            ]
                        )
                    if metrics:
                        episode_summary["social_metrics"] = metrics
                    self.logger.log_agent_episode_details(agent_id, episode, episode_summary)

            video_path = video_recorder.finish()
            if video_path is not None:
                self.console.info(f"video saved: {video_path}")
            self.console.episode_end(episode, self._episode_stats(payload), advance=num_envs)
            recent_rewards.append(payload["reward_mean"])
```

Two details to carry over deliberately:

- `self.algorithm.on_episode_end(iteration)` is passed the **iteration** index, because the entropy controllers' anneal schedule counts updates against `set_total_episodes`. Also change the `set_total_episodes` call at the top of `train()` to `self.algorithm.set_total_episodes(self.iterations)` so the schedule spans the run.
- `global_step += num_envs` per step, because `num_envs` environment transitions occur per loop iteration and `reward_model.update_every_env_steps` is denominated in env steps.

- [ ] **Step 16: Run the whole suite**

Run: `uv run pytest`
Expected: PASS. Every pre-existing test must still pass — `tests/test_trainer_obs.py` in particular exercises the observation path and will catch a wrong reshape.

- [ ] **Step 17: Commit**

```bash
git add -A src/commons_game_marp tests
git commit -m "feat(train): run parallel environments through a flat-row algorithm interface"
```

---

### Task 5: Document the new knob and verify a real run

**Files:**
- Modify: `src/commons_game_marp/configs/experiment/example.yaml` (env section)
- Test: manual run, plus the suite

**Interfaces:**
- Consumes: everything from Tasks 1-4.
- Produces: no new code interfaces.

- [ ] **Step 1: Document `num_envs` in the reference config**

In `src/commons_game_marp/configs/experiment/example.yaml`, add to the `env:` block, beside `num_frames`:

```yaml
  # Independent environment copies stepped in lockstep. Each policy update then
  # draws on num_envs decorrelated rollouts instead of one long correlated
  # trajectory. `episodes` stays the TOTAL budget: at num_envs=4 a 1000-episode
  # run is 250 iterations of 4 episodes, so the sample budget is unchanged and
  # runs stay comparable. `episodes` must be divisible by num_envs.
  #
  # Stepping is serial -- this buys sample decorrelation, not wall-clock speed,
  # and per-step cost grows linearly with num_envs. Note that `n_steps` counts
  # per-env timesteps, so the update batch is n_steps * num_envs: a run at
  # num_envs=4 is not hyperparameter-identical to one at num_envs=1.
  num_envs: 1
```

- [ ] **Step 2: Verify the documented config still composes**

Run: `uv run pytest tests/test_hydra_configs.py -v`
Expected: PASS.

- [ ] **Step 3: Run a real short training run with parallel envs**

Run:

```bash
uv run commons-game-train experiment=ippo episodes=8 env.num_envs=4 \
  env.ep_length=20 env.map_type=small env.num_agents=2 \
  logging.video_enabled=false
```

Expected: the run completes; the setup banner reports `parallel envs  : 4 (2 iterations x 4 episodes)`; `metrics.jsonl` in the run directory holds 2 records with `"num_envs": 4` and `"episode"` values `3` and `7`.

- [ ] **Step 4: Confirm the single-env default is untouched**

Run:

```bash
uv run commons-game-train experiment=ippo episodes=4 env.ep_length=20 \
  env.map_type=small env.num_agents=2 logging.video_enabled=false
```

Expected: completes; `metrics.jsonl` holds 4 records with `"num_envs": 1` and `"episode"` values `0,1,2,3`.

- [ ] **Step 5: Run the full suite one last time**

Run: `uv run pytest`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/commons_game_marp/configs/experiment/example.yaml
git commit -m "docs(configs): document env.num_envs and its effect on batch size"
```

---

## Self-Review Notes

Spec coverage checked section by section:

| Spec section | Task |
|---|---|
| Data layout, row ordering | Task 2 |
| `VecCommonsEnv` | Task 2 |
| Lockstep episodes, explicit reset | Task 2 (no auto-reset), Task 4 (loop) |
| Configuration, `num_envs`, divisibility | Task 3 |
| Algorithm interface | Task 4 (Steps 4-11) |
| IPPO env axis, shared GAE, `n_steps` per env | Task 1, Task 4 (Steps 7-9) |
| MAPPO widened `N`, per-env global obs | Task 4 (Steps 10-11) |
| DQN / random adaptation | Task 4 (Steps 5-6) |
| Reward model, per-env EpisodeRecords | Task 4 (Step 15) |
| Logging averaging, `episode` as episode count | Task 4 (Steps 13-15) |
| Video and detail logs from env 0 | Task 4 (Step 15) |
| Seeding: shared global stream | Task 2 (`reset` docstring and divergence test) |
| Testing | Tasks 1-5 |
