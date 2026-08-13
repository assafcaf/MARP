# Wider View, Frame Stacking, and Adaptive Exploration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Widen the agent view to 7, add opt-in observation frame stacking, and replace IPPO/MAPPO's fixed entropy-coefficient schedule with a controller that targets policy entropy directly.

**Architecture:** Three independent threads. (1) A config-default change for view range. (2) A new `EntropyController` class owning all three coefficient modes, adopted by IPPO (one per agent) and MAPPO (one shared). (3) A `FrameStackEnv` wrapper applied by the trainer only when `num_frames > 1`, so the default path stays byte-identical.

**Tech Stack:** Python 3.12, PyTorch 2.5, Hydra 1.3 (structured configs via `ConfigStore`), pytest, `uv` for running.

**Spec:** `docs/superpowers/specs/2026-08-13-observation-and-exploration-design.md`

## Global Constraints

- Observation arrays are **HWC** `uint8`, shape `(2·view+1, 2·view+1, 3·num_frames)`. Every CNN permutes to CHW internally; do not change that convention.
- `num_frames` defaults to **1**, and at 1 the trainer must not wrap the env at all — the default code path stays unchanged, not merely equivalent.
- `ent_coef_mode` defaults to **`adaptive`** for both IPPO and MAPPO. `anneal` must reproduce the current linear schedule numerically.
- Preference-buffer memory warning threshold: **8 GB**.
- `target_entropy = target_entropy_frac * ln(num_actions)`. At 8 actions and `0.6`, that is **1.2478 nats**.
- Run tests with `uv run pytest`. Run the full suite before the final commit of each task.
- Every commit message ends with:
  ```
  Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
  ```

---

### Task 1: Widen the default agent view range to 7

**Files:**
- Modify: `src/commons_game_marp/train/config.py:11` (`EnvConfig.agent_view_range`)
- Modify: `src/commons_game_marp/configs/env/medium.yaml:5`
- Modify: `src/commons_game_marp/configs/env/small.yaml:5`
- Test: `tests/test_hydra_configs.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `EnvConfig.agent_view_range == 7` by default. Observation shape becomes `(15, 15, 3)`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_hydra_configs.py`:

```python
@pytest.mark.parametrize("env_group", ["medium", "small"])
def test_view_range_matches_the_reference_implementation(env_group):
    """DanfoaTestSOT runs agent_view_range: 7 (src/configs/prm.yaml). The
    observation is (2*view+1, 2*view+1, 3), so this is 15x15x3 rather than
    the 11x11x3 the earlier runs used."""
    config = _compose(f"env={env_group}")
    assert config.env.agent_view_range == 7


def test_env_config_dataclass_default_view_range():
    """The dataclass default must agree with the YAML: a programmatic
    Trainer(TrainerConfig()) bypasses Hydra entirely."""
    from commons_game_marp.train.config import EnvConfig

    assert EnvConfig().agent_view_range == 7
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_hydra_configs.py -k view_range -v`
Expected: FAIL — `assert 5 == 7`, three times.

- [ ] **Step 3: Write minimal implementation**

In `src/commons_game_marp/train/config.py`, `EnvConfig`:

```python
    agent_view_range: int = 7
```

In both `src/commons_game_marp/configs/env/medium.yaml` and `src/commons_game_marp/configs/env/small.yaml`:

```yaml
agent_view_range: 7
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_hydra_configs.py -v && uv run pytest -q`
Expected: PASS. The full suite must stay green — `tests/conftest.py` already hardcodes `shape=(15, 15, 3)` in `FakeEnv`, which now matches the real env instead of contradicting it.

- [ ] **Step 5: Commit**

```bash
git add src/commons_game_marp/train/config.py \
        src/commons_game_marp/configs/env/medium.yaml \
        src/commons_game_marp/configs/env/small.yaml \
        tests/test_hydra_configs.py
git commit -m "feat(env): widen default agent view range to 7

Matches DanfoaTestSOT's prm.yaml. Observations go 11x11x3 -> 15x15x3;
every CNN sizes its linear layer from a dummy forward pass, so no network
code changes.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 2: `EntropyController`

**Files:**
- Create: `src/commons_game_marp/train/entropy_control.py`
- Test: `tests/test_entropy_controller.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `EntropyController(config: Any, num_actions: int, device: torch.device)`
  - `.target_entropy: float`
  - `.set_total_episodes(total: int) -> None`
  - `.set_episode(episode: int) -> None`
  - `.coefficient() -> float`
  - `.observe_entropy(entropy: float) -> None`

  Tasks 3 and 4 depend on exactly these names. `config` is duck-typed — it reads `ent_coef_mode`, `ent_coef`, `ent_coef_end`, `target_entropy_frac`, `ent_coef_lr`, `ent_coef_min`, `ent_coef_max` via `getattr` with defaults, so a `SimpleNamespace` works in tests.

- [ ] **Step 1: Write the failing test**

Create `tests/test_entropy_controller.py`:

```python
"""The entropy coefficient is a means; policy entropy is the end.

Run 20260813-125003-seed=0 collapsed to 0.64 nats while ent_coef was still
0.068 -- a schedule floor of 0.03 would never have been reached, let alone
helped. These tests pin the controller that targets entropy directly.
"""

import math
from types import SimpleNamespace

import pytest
import torch

from commons_game_marp.train.entropy_control import EntropyController

CPU = torch.device("cpu")


def _config(**overrides):
    base = dict(
        ent_coef_mode="adaptive",
        ent_coef=0.1,
        ent_coef_end=0.01,
        target_entropy_frac=0.6,
        ent_coef_lr=0.01,
        ent_coef_min=0.001,
        ent_coef_max=0.5,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_target_entropy_is_a_fraction_of_maximum():
    """Expressed as a fraction so it transfers across action-space sizes."""
    controller = EntropyController(_config(), num_actions=8, device=CPU)
    assert controller.target_entropy == pytest.approx(0.6 * math.log(8))


def test_fixed_mode_never_moves():
    controller = EntropyController(_config(ent_coef_mode="fixed"), 8, CPU)
    controller.set_total_episodes(100)
    controller.set_episode(50)
    controller.observe_entropy(0.1)
    assert controller.coefficient() == pytest.approx(0.1)


def test_anneal_mode_reproduces_the_previous_linear_schedule():
    """Regression guard against the schedule that ran in the analysed run:
    ent_coef 0.1 -> 0.01 linear in episode/total. The logged value at episode
    350 of 1000 was 0.0684."""
    controller = EntropyController(_config(ent_coef_mode="anneal"), 8, CPU)
    controller.set_total_episodes(1000)

    controller.set_episode(0)
    assert controller.coefficient() == pytest.approx(0.1)

    controller.set_episode(350)
    assert controller.coefficient() == pytest.approx(0.0685)

    controller.set_episode(1000)
    assert controller.coefficient() == pytest.approx(0.01)


def test_anneal_mode_clamps_progress_past_the_end():
    controller = EntropyController(_config(ent_coef_mode="anneal"), 8, CPU)
    controller.set_total_episodes(100)
    controller.set_episode(500)
    assert controller.coefficient() == pytest.approx(0.01)


def test_adaptive_raises_the_coefficient_when_entropy_is_below_target():
    """The failure this exists to catch: entropy pinned at 0.5 nats against a
    1.25 target must drive the bonus up, not let it anneal away."""
    controller = EntropyController(_config(), 8, CPU)
    before = controller.coefficient()
    for _ in range(20):
        controller.observe_entropy(0.5)
    assert controller.coefficient() > before


def test_adaptive_lowers_the_coefficient_when_entropy_is_above_target():
    controller = EntropyController(_config(), 8, CPU)
    before = controller.coefficient()
    for _ in range(20):
        controller.observe_entropy(2.0)
    assert controller.coefficient() < before


def test_adaptive_respects_the_upper_clamp_without_windup():
    """Clamping only the read value would let log_ent_coef integrate far past
    the ceiling and then take just as long to come back. The parameter itself
    must be clamped."""
    controller = EntropyController(_config(ent_coef_lr=0.5), 8, CPU)
    for _ in range(200):
        controller.observe_entropy(0.0)
    assert controller.coefficient() == pytest.approx(0.5)

    for _ in range(5):
        controller.observe_entropy(2.079)
    assert controller.coefficient() < 0.5


def test_adaptive_respects_the_lower_clamp():
    controller = EntropyController(_config(ent_coef_lr=0.5), 8, CPU)
    for _ in range(200):
        controller.observe_entropy(2.079)
    assert controller.coefficient() == pytest.approx(0.001)


def test_observe_entropy_is_inert_outside_adaptive_mode():
    for mode in ("fixed", "anneal"):
        controller = EntropyController(_config(ent_coef_mode=mode), 8, CPU)
        controller.set_total_episodes(100)
        controller.set_episode(0)
        controller.observe_entropy(0.0)
        assert controller.coefficient() == pytest.approx(0.1)


def test_unknown_mode_is_rejected_at_construction():
    with pytest.raises(ValueError, match="ent_coef_mode"):
        EntropyController(_config(ent_coef_mode="linear"), 8, CPU)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_entropy_controller.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'commons_game_marp.train.entropy_control'`.

- [ ] **Step 3: Write minimal implementation**

Create `src/commons_game_marp/train/entropy_control.py`:

```python
"""Entropy-coefficient control for the PPO-family algorithms.

Three modes share one interface so IPPO and MAPPO carry no schedule logic of
their own.

`adaptive` exists because the coefficient turned out to be the wrong variable
to control. In run 20260813-125003-seed=0 the policy entropy had already
collapsed to 0.64 nats while `ent_coef` was still 0.068 -- roughly thirty times
the policy-loss magnitude in the total loss, and still losing. No reachable
floor on a fixed schedule would have helped, because the schedule never got
low enough to be the binding constraint. Targeting the entropy itself does.
"""

import math
from typing import Any, Optional

import torch


class EntropyController:
    """Supplies the entropy coefficient for one policy's PPO update.

    IPPO builds one per agent (its networks are per-agent, so its exploration
    pressure should be too, and a diverging agent then shows up directly as a
    diverging coefficient). MAPPO builds one, matching its shared actor.
    """

    MODES = ("fixed", "anneal", "adaptive")

    def __init__(self, config: Any, num_actions: int, device: torch.device) -> None:
        self.mode = str(getattr(config, "ent_coef_mode", "anneal"))
        if self.mode not in self.MODES:
            raise ValueError(
                f"Unknown ent_coef_mode '{self.mode}'. Available: {list(self.MODES)}"
            )

        self.start = float(getattr(config, "ent_coef", 0.01))
        self.end = float(getattr(config, "ent_coef_end", self.start))
        self.minimum = float(getattr(config, "ent_coef_min", 1e-3))
        self.maximum = float(getattr(config, "ent_coef_max", 0.5))
        frac = float(getattr(config, "target_entropy_frac", 0.6))
        self.target_entropy = frac * math.log(num_actions)

        self._total_episodes = 0
        self._current_episode = 0
        self._log_ent_coef: Optional[torch.Tensor] = None
        self._optimizer: Optional[torch.optim.Optimizer] = None
        self._log_min = math.log(self.minimum)
        self._log_max = math.log(self.maximum)

        if self.mode == "adaptive":
            initial = min(max(self.start, self.minimum), self.maximum)
            self._log_ent_coef = torch.tensor(
                math.log(initial), dtype=torch.float32, device=device, requires_grad=True
            )
            self._optimizer = torch.optim.Adam(
                [self._log_ent_coef], lr=float(getattr(config, "ent_coef_lr", 3e-4))
            )

    def set_total_episodes(self, total: int) -> None:
        self._total_episodes = int(total)

    def set_episode(self, episode: int) -> None:
        self._current_episode = int(episode)

    def coefficient(self) -> float:
        """The coefficient to multiply the entropy bonus by, as a plain float.

        Returned detached in every mode: the controller trains its own
        parameter through `observe_entropy`, and letting the policy loss
        backpropagate into it as well would make the two objectives fight.
        """
        if self.mode == "fixed":
            return self.start
        if self.mode == "anneal":
            if self._total_episodes <= 0 or self.start == self.end:
                return self.start
            progress = min(1.0, self._current_episode / self._total_episodes)
            return self.start + (self.end - self.start) * progress
        assert self._log_ent_coef is not None
        return float(self._log_ent_coef.detach().exp().clamp(self.minimum, self.maximum))

    def observe_entropy(self, entropy: float) -> None:
        """One dual-ascent step from an observed policy entropy. Adaptive only.

        `loss = log_ent_coef * (entropy - target)` has gradient `(entropy -
        target)` with respect to the parameter, so entropy below target gives a
        negative gradient and the descent step raises the coefficient. The
        parameter is clamped after the step rather than only on read: clamping
        the read value alone lets the parameter integrate far past the bound
        and take just as long to unwind.
        """
        if self.mode != "adaptive":
            return
        assert self._log_ent_coef is not None and self._optimizer is not None
        loss = self._log_ent_coef * (float(entropy) - self.target_entropy)
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()
        with torch.no_grad():
            self._log_ent_coef.clamp_(self._log_min, self._log_max)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_entropy_controller.py -v`
Expected: PASS, 10 tests.

- [ ] **Step 5: Commit**

```bash
git add src/commons_game_marp/train/entropy_control.py tests/test_entropy_controller.py
git commit -m "feat(train): add EntropyController with fixed/anneal/adaptive modes

Adaptive mode targets a policy entropy in nats via SAC-style dual ascent on
log_ent_coef, clamping the parameter (not just the read value) to avoid
integral windup. Not yet wired into any algorithm.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 3: Adopt the controller in IPPO, split the gradient clip

**Files:**
- Modify: `src/commons_game_marp/train/config.py` (`IPPOConfig`)
- Modify: `src/commons_game_marp/configs/algorithm/ippo.yaml`
- Modify: `src/commons_game_marp/train/algorithms/ippo.py` (`on_env_ready`, `_get_entropy_coef`, `on_episode_end`, `_update_all`, `_update_agent`)
- Test: `tests/test_algorithm_init.py`

**Interfaces:**
- Consumes: `EntropyController` from Task 2, with the exact method names listed there.
- Produces: `IPPOAlgorithm.ent_controllers: Dict[str, EntropyController]`, keyed by agent id. `on_episode_end` returns `algo_metrics` containing `ent_coef` (float, mean), `ent_coef_per_agent` (dict), `target_entropy` (float), `entropy` (float, mean), `entropy_per_agent` (dict).

- [ ] **Step 1: Write the failing test**

Append to `tests/test_algorithm_init.py`:

```python
def test_ippo_builds_one_entropy_controller_per_agent(fake_env):
    """IPPO's networks are per-agent, so its exploration pressure is too --
    which is what makes a diverging agent visible in ent_coef_per_agent."""
    from commons_game_marp.train.algorithms.ippo import IPPOAlgorithm
    from commons_game_marp.train.config import IPPOConfig

    algorithm = IPPOAlgorithm(IPPOConfig())
    algorithm.on_env_ready(fake_env)

    assert set(algorithm.ent_controllers) == set(fake_env.agents)
    controllers = list(algorithm.ent_controllers.values())
    assert len({id(c) for c in controllers}) == len(controllers)


def test_ippo_defaults_to_adaptive_entropy(fake_env):
    from commons_game_marp.train.algorithms.ippo import IPPOAlgorithm
    from commons_game_marp.train.config import IPPOConfig

    algorithm = IPPOAlgorithm(IPPOConfig())
    algorithm.on_env_ready(fake_env)

    for controller in algorithm.ent_controllers.values():
        assert controller.mode == "adaptive"
        assert controller.target_entropy == pytest.approx(0.6 * math.log(8))


def test_ippo_reports_per_agent_entropy_metrics(fake_env):
    """Per-agent divergence is the failure mode this change exists to catch,
    so the per-agent series must survive into algo_metrics rather than being
    averaged away."""
    from commons_game_marp.train.algorithms.ippo import IPPOAlgorithm
    from commons_game_marp.train.config import IPPOConfig

    algorithm = IPPOAlgorithm(IPPOConfig())
    algorithm.on_env_ready(fake_env)
    algorithm._last_metrics = {
        "entropy": 1.0,
        "entropy_per_agent": {"agent-0": 1.5, "agent-1": 0.5},
    }

    metrics = algorithm.on_episode_end(0)

    assert set(metrics["ent_coef_per_agent"]) == set(fake_env.agents)
    assert metrics["ent_coef"] == pytest.approx(
        sum(metrics["ent_coef_per_agent"].values()) / len(fake_env.agents)
    )
    assert metrics["target_entropy"] == pytest.approx(0.6 * math.log(8))
    assert metrics["entropy_per_agent"] == {"agent-0": 1.5, "agent-1": 0.5}
```

Add `import math` and `import pytest` at the top of the file if not already present.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_algorithm_init.py -k entropy -v`
Expected: FAIL — `AttributeError: 'IPPOAlgorithm' object has no attribute 'ent_controllers'`.

- [ ] **Step 3: Write minimal implementation**

In `src/commons_game_marp/train/config.py`, `IPPOConfig` — replace the `ent_coef` / `ent_coef_end` pair with the full set:

```python
    ent_coef_mode: str = "adaptive"  # "fixed" | "anneal" | "adaptive"
    ent_coef: float = 0.1  # initial value (adaptive) / schedule start (anneal)
    ent_coef_end: float = 0.01  # anneal only
    target_entropy_frac: float = 0.6  # adaptive: target = frac * ln(num_actions)
    ent_coef_lr: float = 3e-4
    ent_coef_min: float = 1e-3
    ent_coef_max: float = 0.5
```

In `src/commons_game_marp/configs/algorithm/ippo.yaml`, replace the `ent_coef` / `ent_coef_end` lines with:

```yaml
ent_coef_mode: adaptive
ent_coef: 0.1
ent_coef_end: 0.01
target_entropy_frac: 0.6
ent_coef_lr: 0.0003
ent_coef_min: 0.001
ent_coef_max: 0.5
```

In `src/commons_game_marp/train/algorithms/ippo.py`, add the import:

```python
from ..entropy_control import EntropyController
```

Add to `IPPOAlgorithm.__init__`, beside the other per-agent dicts:

```python
        self.ent_controllers: Dict[str, EntropyController] = {}
```

Delete `_get_entropy_coef` entirely. Replace `set_total_episodes` with:

```python
    def set_total_episodes(self, total: int) -> None:
        """Set total episodes for the anneal-mode entropy schedule."""
        self._total_episodes = total
        for controller in self.ent_controllers.values():
            controller.set_total_episodes(total)
```

At the end of the per-agent loop in `on_env_ready`, beside `self.buffers[agent_id] = SingleAgentBuffer()`:

```python
            self.ent_controllers[agent_id] = EntropyController(
                self.config, self.num_actions, self.device
            )
            self.ent_controllers[agent_id].set_total_episodes(self._total_episodes)
```

Replace `on_episode_end` with:

```python
    def on_episode_end(self, episode: int) -> Dict[str, Any]:
        # Update any remaining data in buffers
        if any(buf.size() > 0 for buf in self.buffers.values()):
            self._update_all()

        # Track episode for the anneal-mode schedule
        self._current_episode = episode + 1
        for controller in self.ent_controllers.values():
            controller.set_episode(self._current_episode)

        metrics = dict(self._last_metrics)
        per_agent = {
            agent_id: controller.coefficient()
            for agent_id, controller in self.ent_controllers.items()
        }
        metrics["ent_coef_per_agent"] = per_agent
        metrics["ent_coef"] = sum(per_agent.values()) / len(per_agent) if per_agent else 0.0
        if self.ent_controllers:
            metrics["target_entropy"] = next(
                iter(self.ent_controllers.values())
            ).target_entropy
        self._last_metrics = {}
        return metrics
```

In `_update_all`, collect the per-agent entropies. Add before the loop:

```python
        entropy_per_agent: Dict[str, float] = {}
```

Inside the loop, after `total_entropy += metrics.get("entropy", 0.0)`:

```python
            entropy_per_agent[agent_id] = metrics.get("entropy", 0.0)
```

And add the key to `self._last_metrics`:

```python
                "entropy_per_agent": entropy_per_agent,
```

In `_update_agent`, delete this line:

```python
        current_ent_coef = self._get_entropy_coef()
```

and replace it with:

```python
        controller = self.ent_controllers[agent_id]
```

Inside the minibatch loop, replace `- current_ent_coef * entropy` in the loss with a per-minibatch read, since the adaptive coefficient moves within an update. Change:

```python
                loss = (
                    policy_loss
                    + self.config.vf_coef * value_loss
                    - current_ent_coef * entropy
                )
```

to:

```python
                current_ent_coef = controller.coefficient()
                loss = (
                    policy_loss
                    + self.config.vf_coef * value_loss
                    - current_ent_coef * entropy
                )
```

Replace the joint gradient clip:

```python
                nn.utils.clip_grad_norm_(
                    list(actor.parameters()) + list(critic.parameters()),
                    self.config.max_grad_norm,
                )
```

with separate clips, so a critic-loss spike cannot throttle actor learning:

```python
                # Clipped separately: a joint norm lets a critic-loss spike
                # scale the actor's gradient down with it. value_loss went
                # 0.02 -> 0.14 around episode 100 of the analysed run, exactly
                # where entropy first dropped.
                nn.utils.clip_grad_norm_(actor.parameters(), self.config.max_grad_norm)
                nn.utils.clip_grad_norm_(critic.parameters(), self.config.max_grad_norm)
```

Immediately after `optimizer.step()`, feed the observed entropy back:

```python
                controller.observe_entropy(float(entropy.item()))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_algorithm_init.py -v && uv run pytest -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/commons_game_marp/train/config.py \
        src/commons_game_marp/configs/algorithm/ippo.yaml \
        src/commons_game_marp/train/algorithms/ippo.py \
        tests/test_algorithm_init.py
git commit -m "feat(ippo): target policy entropy instead of scheduling ent_coef

One EntropyController per agent, defaulting to adaptive. Logs ent_coef and
entropy per agent so a diverging agent is visible as it happens rather than
only in retrospect. Also splits the actor/critic gradient clip, which was a
single joint norm.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 4: Adopt the controller in MAPPO, split the gradient clip

**Files:**
- Modify: `src/commons_game_marp/train/config.py` (`MAPPOConfig`)
- Modify: `src/commons_game_marp/configs/algorithm/mappo.yaml`
- Modify: `src/commons_game_marp/train/algorithms/mappo.py` (`__init__`, `on_env_ready`, `on_episode_end`, `_update`)
- Test: `tests/test_algorithm_init.py`

**Interfaces:**
- Consumes: `EntropyController` from Task 2.
- Produces: `MAPPOAlgorithm.ent_controller: EntropyController` (singular — MAPPO has one shared actor), `MAPPOAlgorithm.set_total_episodes(total: int)`. `on_episode_end` returns `algo_metrics` containing `ent_coef` and `target_entropy`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_algorithm_init.py`:

```python
def test_mappo_builds_one_shared_entropy_controller(fake_env):
    """MAPPO has a single shared actor, so a single controller."""
    from commons_game_marp.train.algorithms.mappo import MAPPOAlgorithm
    from commons_game_marp.train.config import MAPPOConfig

    algorithm = MAPPOAlgorithm(MAPPOConfig())
    algorithm.on_env_ready(fake_env)

    assert algorithm.ent_controller.mode == "adaptive"
    assert algorithm.ent_controller.target_entropy == pytest.approx(0.6 * math.log(8))


def test_mappo_accepts_total_episodes_for_annealing(fake_env):
    """Trainer.train() calls this behind a hasattr check. MAPPO had no such
    method, so anneal mode would have silently held at the start value."""
    from commons_game_marp.train.algorithms.mappo import MAPPOAlgorithm
    from commons_game_marp.train.config import MAPPOConfig

    config = MAPPOConfig()
    config.ent_coef_mode = "anneal"
    config.ent_coef = 0.1
    config.ent_coef_end = 0.01

    algorithm = MAPPOAlgorithm(config)
    algorithm.on_env_ready(fake_env)
    algorithm.set_total_episodes(1000)

    assert algorithm.on_episode_end(-1)["ent_coef"] == pytest.approx(0.1)
    assert algorithm.on_episode_end(349)["ent_coef"] == pytest.approx(0.0685)


def test_mappo_reports_entropy_coefficient(fake_env):
    from commons_game_marp.train.algorithms.mappo import MAPPOAlgorithm
    from commons_game_marp.train.config import MAPPOConfig

    algorithm = MAPPOAlgorithm(MAPPOConfig())
    algorithm.on_env_ready(fake_env)

    metrics = algorithm.on_episode_end(0)

    assert metrics["ent_coef"] == pytest.approx(0.1)
    assert metrics["target_entropy"] == pytest.approx(0.6 * math.log(8))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_algorithm_init.py -k mappo -v`
Expected: FAIL — `AttributeError: 'MAPPOAlgorithm' object has no attribute 'ent_controller'`.

- [ ] **Step 3: Write minimal implementation**

In `src/commons_game_marp/train/config.py`, `MAPPOConfig` — replace `ent_coef: float = 0.01` with the same block IPPO now has:

```python
    ent_coef_mode: str = "adaptive"  # "fixed" | "anneal" | "adaptive"
    ent_coef: float = 0.1  # initial value (adaptive) / schedule start (anneal)
    ent_coef_end: float = 0.01  # anneal only
    target_entropy_frac: float = 0.6  # adaptive: target = frac * ln(num_actions)
    ent_coef_lr: float = 3e-4
    ent_coef_min: float = 1e-3
    ent_coef_max: float = 0.5
```

In `src/commons_game_marp/configs/algorithm/mappo.yaml`, replace `ent_coef: 0.01` with:

```yaml
ent_coef_mode: adaptive
ent_coef: 0.1
ent_coef_end: 0.01
target_entropy_frac: 0.6
ent_coef_lr: 0.0003
ent_coef_min: 0.001
ent_coef_max: 0.5
```

In `src/commons_game_marp/train/algorithms/mappo.py`, add the import:

```python
from ..entropy_control import EntropyController
```

Widen the `typing` import on line 3 — `Optional` is not currently imported in this module:

```python
from typing import Any, Dict, List, Optional, Tuple
```

Add to `MAPPOAlgorithm.__init__`, beside `self._last_metrics`:

```python
        self.ent_controller: Optional[EntropyController] = None
        self._total_episodes = 0
        self._current_episode = 0
```

Add a new method after `uses_external_loop`:

```python
    def set_total_episodes(self, total: int) -> None:
        """Set total episodes for the anneal-mode entropy schedule.

        Trainer.train() calls this behind a `hasattr` check. Without it, anneal
        mode would hold at the start value for the whole run.
        """
        self._total_episodes = total
        if self.ent_controller is not None:
            self.ent_controller.set_total_episodes(total)
```

At the end of `on_env_ready`, after the optimizer is built:

```python
        self.ent_controller = EntropyController(
            self.config, self.num_actions, self.device
        )
        self.ent_controller.set_total_episodes(self._total_episodes)
```

Replace `on_episode_end` with:

```python
    def on_episode_end(self, episode: int) -> Dict[str, Any]:
        if self.buffer.size() > 0:
            self._update()
            self.buffer.clear()

        self._current_episode = episode + 1
        metrics = dict(self._last_metrics)
        if self.ent_controller is not None:
            self.ent_controller.set_episode(self._current_episode)
            metrics["ent_coef"] = self.ent_controller.coefficient()
            metrics["target_entropy"] = self.ent_controller.target_entropy
        self._last_metrics = {}
        return metrics
```

In `_update`, inside the minibatch loop, replace:

```python
                loss = policy_loss + self.config.vf_coef * value_loss - self.config.ent_coef * entropy
```

with:

```python
                current_ent_coef = self.ent_controller.coefficient()
                loss = (
                    policy_loss
                    + self.config.vf_coef * value_loss
                    - current_ent_coef * entropy
                )
```

Replace the joint gradient clip:

```python
                nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) + list(self.critic.parameters()),
                    self.config.max_grad_norm,
                )
```

with:

```python
                # Clipped separately: a joint norm lets a critic-loss spike
                # scale the actor's gradient down with it.
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.max_grad_norm)
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.max_grad_norm)
```

Immediately after `self.optimizer.step()`:

```python
                self.ent_controller.observe_entropy(float(entropy.item()))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_algorithm_init.py -v && uv run pytest -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/commons_game_marp/train/config.py \
        src/commons_game_marp/configs/algorithm/mappo.yaml \
        src/commons_game_marp/train/algorithms/mappo.py \
        tests/test_algorithm_init.py
git commit -m "feat(mappo): adopt EntropyController and split the gradient clip

MAPPO applied a flat ent_coef with no schedule, no episode tracking, and no
ent_coef in its metrics. It now has the same three modes IPPO does, plus the
set_total_episodes hook Trainer.train() already looks for.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 5: `FrameStackEnv`

**Files:**
- Create: `src/commons_game_marp/env/frame_stack.py`
- Test: `tests/test_frame_stack.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `FrameStackEnv(env, num_frames: int)` with `.reset(seed=None) -> (obs, infos)`, `.step(actions) -> (obs, rewards, dones, infos)`, `.observation_space` property, and `__getattr__` delegation. Task 6 constructs it.

- [ ] **Step 1: Write the failing test**

Create `tests/test_frame_stack.py`:

```python
"""Frame stacking mirrors DanfoaTestSOT, which applies it as an env wrapper
(ss.frame_stack_v1 in src/experiment_runner/runners.py:120) so the policy and
the reward predictor both see the stack."""

import gymnasium
import numpy as np
import pytest

from commons_game_marp.env.frame_stack import FrameStackEnv


class _StubEnv:
    """Emits a distinct constant frame per step so stacking order is checkable."""

    def __init__(self, agent_ids=("agent-0", "agent-1"), shape=(3, 3, 3)):
        self.agent_ids = list(agent_ids)
        self.shape = shape
        self.counter = 0
        self.agents = {agent_id: object() for agent_id in self.agent_ids}
        self.action_space = gymnasium.spaces.Discrete(8)
        self.rendered = []

    @property
    def observation_space(self):
        return {
            "curr_obs": gymnasium.spaces.Box(
                low=0, high=255, shape=self.shape, dtype=np.uint8
            )
        }

    def _frame(self):
        return np.full(self.shape, self.counter, dtype=np.uint8)

    def reset(self, seed=None):
        self.counter = 1
        obs = {a: {"curr_obs": self._frame()} for a in self.agent_ids}
        return obs, {a: {} for a in self.agent_ids}

    def step(self, actions):
        self.counter += 1
        obs = {a: {"curr_obs": self._frame()} for a in self.agent_ids}
        rewards = {a: 0.0 for a in self.agent_ids}
        dones = {a: False for a in self.agent_ids}
        return obs, rewards, dones, {a: {} for a in self.agent_ids}

    def render(self, path, mod="human"):
        self.rendered.append(path)

    def get_social_metrics(self):
        return {"efficiency": 1.0}


def test_observation_space_widens_along_the_channel_axis():
    wrapped = FrameStackEnv(_StubEnv(), num_frames=2)
    assert wrapped.observation_space["curr_obs"].shape == (3, 3, 6)
    assert wrapped.observation_space["curr_obs"].dtype == np.uint8


def test_reset_fills_the_stack_by_repeating_the_first_frame():
    """The first step must already have a full stack, not a zero-padded one:
    a zero half-frame is a real observation the policy would have to learn to
    ignore."""
    wrapped = FrameStackEnv(_StubEnv(), num_frames=2)
    obs, _ = wrapped.reset()

    frame = obs["agent-0"]["curr_obs"]
    assert frame.shape == (3, 3, 6)
    assert np.all(frame == 1)


def test_step_appends_the_newest_frame_last():
    wrapped = FrameStackEnv(_StubEnv(), num_frames=2)
    wrapped.reset()
    obs, _, _, _ = wrapped.step({"agent-0": 0, "agent-1": 0})

    frame = obs["agent-0"]["curr_obs"]
    assert np.all(frame[:, :, 0:3] == 1)  # oldest
    assert np.all(frame[:, :, 3:6] == 2)  # newest


def test_oldest_frame_is_evicted_once_the_stack_is_full():
    wrapped = FrameStackEnv(_StubEnv(), num_frames=2)
    wrapped.reset()
    wrapped.step({"agent-0": 0, "agent-1": 0})
    obs, _, _, _ = wrapped.step({"agent-0": 0, "agent-1": 0})

    frame = obs["agent-0"]["curr_obs"]
    assert np.all(frame[:, :, 0:3] == 2)
    assert np.all(frame[:, :, 3:6] == 3)


def test_each_agent_keeps_its_own_stack():
    wrapped = FrameStackEnv(_StubEnv(), num_frames=3)
    wrapped.reset()
    obs, _, _, _ = wrapped.step({"agent-0": 0, "agent-1": 0})

    assert obs["agent-0"]["curr_obs"].shape == (3, 3, 9)
    assert obs["agent-1"]["curr_obs"].shape == (3, 3, 9)


def test_reset_clears_state_from_the_previous_episode():
    """reset() rebuilds the env's agents, so a stack carried across the
    boundary would splice two episodes into one observation."""
    wrapped = FrameStackEnv(_StubEnv(), num_frames=2)
    wrapped.reset()
    wrapped.step({"agent-0": 0, "agent-1": 0})
    obs, _ = wrapped.reset()

    assert np.all(obs["agent-0"]["curr_obs"] == 1)


def test_frames_stay_uint8():
    """PreferenceBuffer sizing assumes uint8; a float32 stack would quadruple
    resident memory."""
    wrapped = FrameStackEnv(_StubEnv(), num_frames=2)
    obs, _ = wrapped.reset()
    assert obs["agent-0"]["curr_obs"].dtype == np.uint8


def test_unknown_attributes_delegate_to_the_wrapped_env():
    """Trainer reaches through for env.agents, compute_social_metrics(), and
    the video recorder's env.render()."""
    inner = _StubEnv()
    wrapped = FrameStackEnv(inner, num_frames=2)

    assert wrapped.agents is inner.agents
    assert wrapped.get_social_metrics() == {"efficiency": 1.0}
    assert wrapped.action_space is inner.action_space
    wrapped.render("/tmp/x.png", mod="human")
    assert inner.rendered == ["/tmp/x.png"]


def test_num_frames_below_two_is_rejected():
    """The trainer must not wrap at all for num_frames == 1; a wrapper that
    silently accepted it would make the default path non-identical."""
    with pytest.raises(ValueError, match="num_frames"):
        FrameStackEnv(_StubEnv(), num_frames=1)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_frame_stack.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'commons_game_marp.env.frame_stack'`.

- [ ] **Step 3: Write minimal implementation**

Create `src/commons_game_marp/env/frame_stack.py`:

```python
"""Observation frame stacking, applied as an environment wrapper.

Wrapping the env rather than each algorithm's `_format_obs` means every
consumer picks the stack up from `observation_space` on its own: the four
algorithms, `RewardModel` (built from `env.observation_space["curr_obs"].shape`
in `Trainer.train`), and the preference buffer that stores what
`_format_reward_obs` returns. That mirrors the reference implementation, where
`ss.frame_stack_v1` sits on the env and the reward predictor reads the stack
depth straight off the observation space.

Note the memory consequence: `PreferenceBuffer` holds raw frames, so its
resident size scales linearly with `num_frames`. See `Trainer._warn_if_buffer_large`.
"""

from collections import deque
from typing import Any, Deque, Dict, Optional, Tuple

import gymnasium
import numpy as np


class FrameStackEnv:
    """Stacks the last `num_frames` observations along the channel axis.

    Only constructed when `num_frames > 1`; at 1 the trainer uses the bare env
    so the default code path is unchanged rather than merely equivalent.
    """

    def __init__(self, env: Any, num_frames: int) -> None:
        if num_frames < 2:
            raise ValueError(
                f"num_frames must be >= 2 to stack, got {num_frames}. "
                "Use the unwrapped env for num_frames == 1."
            )
        self.env = env
        self.num_frames = int(num_frames)
        self._stacks: Dict[str, Deque[np.ndarray]] = {}

    @property
    def observation_space(self) -> Dict[str, gymnasium.spaces.Box]:
        inner = self.env.observation_space["curr_obs"]
        height, width, channels = inner.shape
        return {
            "curr_obs": gymnasium.spaces.Box(
                low=0,
                high=255,
                shape=(height, width, channels * self.num_frames),
                dtype=np.uint8,
            )
        }

    def _stack_for(self, agent_id: str) -> np.ndarray:
        # Oldest first, newest last -- the ordering the tests pin and the one a
        # reader of a rendered stack expects.
        return np.concatenate(list(self._stacks[agent_id]), axis=-1)

    def _seed_stacks(self, observations: Dict[str, Any]) -> Dict[str, Any]:
        """Start each agent's stack by repeating its first frame.

        Zero padding would hand the policy a half-blank observation on the
        first steps of every episode -- a real input it would have to learn to
        ignore.
        """
        self._stacks = {}
        stacked = {}
        for agent_id, agent_obs in observations.items():
            frame = np.asarray(agent_obs["curr_obs"], dtype=np.uint8)
            self._stacks[agent_id] = deque(
                [frame] * self.num_frames, maxlen=self.num_frames
            )
            stacked[agent_id] = {**agent_obs, "curr_obs": self._stack_for(agent_id)}
        return stacked

    def _append(self, observations: Dict[str, Any]) -> Dict[str, Any]:
        stacked = {}
        for agent_id, agent_obs in observations.items():
            frame = np.asarray(agent_obs["curr_obs"], dtype=np.uint8)
            if agent_id not in self._stacks:
                self._stacks[agent_id] = deque(
                    [frame] * self.num_frames, maxlen=self.num_frames
                )
            else:
                self._stacks[agent_id].append(frame)
            stacked[agent_id] = {**agent_obs, "curr_obs": self._stack_for(agent_id)}
        return stacked

    def reset(self, seed: Optional[int] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        observations, infos = self.env.reset(seed=seed)
        return self._seed_stacks(observations), infos

    def step(self, actions: Dict[str, int]) -> Tuple[Dict[str, Any], Any, Any, Any]:
        observations, rewards, dones, infos = self.env.step(actions)
        return self._append(observations), rewards, dones, infos

    def __getattr__(self, name: str) -> Any:
        # Only called for attributes this wrapper does not define, so `env`,
        # `num_frames` and the methods above never reach here. Guarded against
        # recursion during unpickling, when `env` may not be set yet.
        if name == "env":
            raise AttributeError(name)
        return getattr(self.env, name)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_frame_stack.py -v`
Expected: PASS, 9 tests.

- [ ] **Step 5: Commit**

```bash
git add src/commons_game_marp/env/frame_stack.py tests/test_frame_stack.py
git commit -m "feat(env): add FrameStackEnv wrapper

Stacks the last N observations along the channel axis, mirroring
ss.frame_stack_v1 in the reference. Not yet wired to config.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 6: Wire `num_frames` through config and the trainer

**Files:**
- Modify: `src/commons_game_marp/train/config.py` (`EnvConfig`)
- Modify: `src/commons_game_marp/configs/env/medium.yaml`
- Modify: `src/commons_game_marp/configs/env/small.yaml`
- Modify: `src/commons_game_marp/train/trainer.py` (`_build_env`, `_announce_setup`)
- Test: `tests/test_hydra_configs.py`, `tests/test_trainer_obs.py`

**Interfaces:**
- Consumes: `FrameStackEnv` from Task 5.
- Produces: `EnvConfig.num_frames: int = 1`. `Trainer._build_env()` returns a bare `HarvestCommonsEnv` when `num_frames == 1` and a `FrameStackEnv` otherwise.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_hydra_configs.py`:

```python
@pytest.mark.parametrize("env_group", ["medium", "small"])
def test_num_frames_defaults_to_one(env_group):
    """Opt-in by design: at 1 the trainer skips the wrapper entirely, so the
    default code path is unchanged rather than merely equivalent."""
    config = _compose(f"env={env_group}")
    assert config.env.num_frames == 1


def test_num_frames_is_overridable():
    config = _compose("env=medium", "env.num_frames=2")
    assert config.env.num_frames == 2
```

Append to `tests/test_trainer_obs.py`:

```python
def test_build_env_returns_the_bare_env_at_one_frame():
    from commons_game_marp.env.commons_env import HarvestCommonsEnv
    from commons_game_marp.train.config import TrainerConfig

    config = TrainerConfig()
    config.env.num_frames = 1
    config.env.num_agents = 2
    config.env.map_type = "small"

    stub = object.__new__(Trainer)
    stub.config = config
    env = Trainer._build_env(stub)

    assert isinstance(env, HarvestCommonsEnv)


def test_build_env_wraps_and_widens_the_observation_space_above_one_frame():
    from commons_game_marp.env.frame_stack import FrameStackEnv
    from commons_game_marp.train.config import TrainerConfig

    config = TrainerConfig()
    config.env.num_frames = 2
    config.env.num_agents = 2
    config.env.map_type = "small"

    stub = object.__new__(Trainer)
    stub.config = config
    env = Trainer._build_env(stub)

    assert isinstance(env, FrameStackEnv)
    height, width, channels = env.observation_space["curr_obs"].shape
    assert (height, width) == (15, 15)
    assert channels == 6
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_hydra_configs.py -k num_frames tests/test_trainer_obs.py -k build_env -v`
Expected: FAIL — `ConfigAttributeError: Key 'num_frames' is not in struct`.

- [ ] **Step 3: Write minimal implementation**

In `src/commons_game_marp/train/config.py`, `EnvConfig`, after `agent_view_range`:

```python
    # Observations stacked along the channel axis. 1 leaves the env unwrapped;
    # above 1 the trainer applies FrameStackEnv and every consumer -- policies,
    # reward model, preference buffer -- widens with it. Note that the
    # preference buffer's resident size scales linearly with this.
    num_frames: int = 1
```

In both `configs/env/medium.yaml` and `configs/env/small.yaml`, after `agent_view_range: 7`:

```yaml
num_frames: 1
```

In `src/commons_game_marp/train/trainer.py`, add the import:

```python
from ..env.frame_stack import FrameStackEnv
```

Replace `_build_env` with:

```python
    def _build_env(self):
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
        num_frames = int(getattr(env_cfg, "num_frames", 1))
        if num_frames > 1:
            return FrameStackEnv(env, num_frames)
        return env
```

The return annotation is dropped because the method now returns either type.

In `_announce_setup`, extend the environment line so the stack depth is visible in the run log:

```python
        stack = f" frames={env_cfg.num_frames}" if env_cfg.num_frames > 1 else ""
        self.console.info(
            f"environment    : map={env_cfg.map_type} agents={env_cfg.num_agents}"
            f" view={env_cfg.agent_view_range}{stack} spawn={env_cfg.spawn_speed}"
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_hydra_configs.py tests/test_trainer_obs.py -v && uv run pytest -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/commons_game_marp/train/config.py \
        src/commons_game_marp/configs/env/medium.yaml \
        src/commons_game_marp/configs/env/small.yaml \
        src/commons_game_marp/train/trainer.py \
        tests/test_hydra_configs.py tests/test_trainer_obs.py
git commit -m "feat(env): wire num_frames through config and the trainer

Defaults to 1, where the env is left unwrapped. Above 1 the reward model and
preference buffer widen automatically, since both derive their shape from
env.observation_space.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 7: Warn when the preference buffer will be large

**Files:**
- Modify: `src/commons_game_marp/train/trainer.py` (`_announce_setup`, new `_projected_buffer_bytes` and `_warn_if_buffer_large`)
- Test: `tests/test_trainer_obs.py`

**Interfaces:**
- Consumes: `EnvConfig.num_frames` from Task 6.
- Produces: `Trainer._projected_buffer_bytes() -> int`, `Trainer.BUFFER_WARN_BYTES: int` (8 GB).

- [ ] **Step 1: Write the failing test**

Append to `tests/test_trainer_obs.py`:

```python
class _WarnCapture:
    def __init__(self):
        self.messages = []

    def warn(self, message):
        self.messages.append(message)

    def info(self, message):
        pass


def _buffer_stub(num_frames, view, max_episodes, store_cap=None, enabled=True):
    from commons_game_marp.train.config import TrainerConfig

    config = TrainerConfig()
    config.env.num_frames = num_frames
    config.env.agent_view_range = view
    config.env.num_agents = 5
    config.env.ep_length = 600
    config.reward_model.enabled = enabled
    config.reward_model.max_episodes_in_buffer = max_episodes
    config.reward_model.store_max_steps_per_agent = store_cap

    stub = object.__new__(Trainer)
    stub.config = config
    stub.console = _WarnCapture()
    return stub


def test_projected_buffer_bytes_matches_the_spec_arithmetic():
    """view 7, 2 frames: 15*15*3*2 = 1350 bytes/frame, x 5000 x 600 x 5."""
    stub = _buffer_stub(num_frames=2, view=7, max_episodes=5000)
    assert Trainer._projected_buffer_bytes(stub) == 1350 * 5000 * 600 * 5


def test_projection_honours_the_per_agent_step_cap():
    stub = _buffer_stub(num_frames=2, view=7, max_episodes=5000, store_cap=100)
    assert Trainer._projected_buffer_bytes(stub) == 1350 * 5000 * 100 * 5


def test_warns_above_the_threshold():
    stub = _buffer_stub(num_frames=2, view=7, max_episodes=5000)
    Trainer._warn_if_buffer_large(stub)
    assert len(stub.console.messages) == 1
    message = stub.console.messages[0]
    assert "max_episodes_in_buffer" in message
    assert "store_max_steps_per_agent" in message


def test_stays_quiet_below_the_threshold():
    stub = _buffer_stub(num_frames=1, view=5, max_episodes=1000)
    Trainer._warn_if_buffer_large(stub)
    assert stub.console.messages == []


def test_stays_quiet_when_the_reward_model_is_off():
    """No reward model means no preference buffer to size."""
    stub = _buffer_stub(num_frames=2, view=7, max_episodes=5000, enabled=False)
    Trainer._warn_if_buffer_large(stub)
    assert stub.console.messages == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_trainer_obs.py -k buffer -v`
Expected: FAIL — `AttributeError: type object 'Trainer' has no attribute '_projected_buffer_bytes'`.

- [ ] **Step 3: Write minimal implementation**

In `src/commons_game_marp/train/trainer.py`, add a class attribute at the top of `Trainer`:

```python
class Trainer:
    # Above this projected resident size, the preference buffer gets a warning
    # at startup. PreferenceBuffer's docstring records a real OOM from this
    # arithmetic; widening the view and stacking frames both push straight
    # into it (view 7 + 2 frames is ~20 GB at the default buffer length).
    BUFFER_WARN_BYTES = 8 * 1024**3
```

Add these two methods next to `_build_env`:

```python
    def _projected_buffer_bytes(self) -> int:
        """Resident size the preference buffer will reach once full.

        Frames are stored raw, so this is the frame size times the number of
        frames retained: one per agent per step, for every episode the ring
        buffer holds. `store_max_steps_per_agent` subsamples at insertion time
        and so replaces `ep_length` when it is set.
        """
        env_cfg = self.config.env
        rm_cfg = self.config.reward_model
        side = 2 * env_cfg.agent_view_range + 1
        bytes_per_frame = side * side * 3 * int(getattr(env_cfg, "num_frames", 1))
        steps = rm_cfg.store_max_steps_per_agent or env_cfg.ep_length
        return bytes_per_frame * rm_cfg.max_episodes_in_buffer * steps * env_cfg.num_agents

    def _warn_if_buffer_large(self) -> None:
        if not self.config.reward_model.enabled:
            return
        projected = self._projected_buffer_bytes()
        if projected <= self.BUFFER_WARN_BYTES:
            return
        self.console.warn(
            f"preference buffer will reach ~{projected / 1024**3:.1f} GB when full. "
            "Lower reward_model.max_episodes_in_buffer or set "
            "reward_model.store_max_steps_per_agent to bound it."
        )
```

Call it at the end of `_announce_setup`:

```python
        self._warn_if_buffer_large()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_trainer_obs.py -v && uv run pytest -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/commons_game_marp/train/trainer.py tests/test_trainer_obs.py
git commit -m "feat(train): warn when the preference buffer will exceed 8 GB

View 7 plus 2 stacked frames projects to ~20 GB at the default buffer length,
against the ~5 GB the previous shape used. PreferenceBuffer's docstring
records a real OOM from exactly this arithmetic.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 8: Update the documented example config

**Files:**
- Modify: `src/commons_game_marp/configs/experiment/example.yaml` (IPPO block ~lines 109-130, commented MAPPO block ~lines 165-180, env block)
- Test: `tests/test_hydra_configs.py`

**Interfaces:**
- Consumes: the config fields added in Tasks 1, 3, 4, and 6.
- Produces: nothing consumed by later tasks.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_hydra_configs.py`:

```python
def test_example_experiment_documents_the_entropy_fields():
    """example.yaml is the reference users copy from. A field that exists in
    the schema but not here is a field nobody discovers."""
    import os

    from commons_game_marp import configs

    path = os.path.join(os.path.dirname(configs.__file__), "experiment", "example.yaml")
    with open(path, encoding="utf-8") as handle:
        text = handle.read()

    for field in (
        "ent_coef_mode",
        "target_entropy_frac",
        "ent_coef_lr",
        "ent_coef_min",
        "ent_coef_max",
        "num_frames",
    ):
        assert field in text, f"{field} is undocumented in example.yaml"


def test_example_experiment_does_not_claim_mappo_lacks_ent_coef_end():
    """MAPPO gained the full entropy field set; the old comment is now false."""
    import os

    from commons_game_marp import configs

    path = os.path.join(os.path.dirname(configs.__file__), "experiment", "example.yaml")
    with open(path, encoding="utf-8") as handle:
        text = handle.read()

    assert "MAPPO has no ent_coef_end" not in text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_hydra_configs.py -k example -v`
Expected: FAIL — `AssertionError: ent_coef_mode is undocumented in example.yaml`.

- [ ] **Step 3: Write minimal implementation**

Read `src/commons_game_marp/configs/experiment/example.yaml` first — the line numbers above are from the pre-change file and the surrounding prose must stay in the file's existing voice.

In the active IPPO `algorithm:` block, replace the `ent_coef` / `ent_coef_end` lines with:

```yaml
  # Exploration. `adaptive` tunes the coefficient to hold policy entropy at
  # `target_entropy_frac * ln(num_actions)` -- 1.25 nats at 8 actions. It
  # replaced a fixed schedule that could not do the job: in the 2026-08-13
  # narrow_view run, entropy had collapsed to 0.64 nats while ent_coef was
  # still 0.068, so no reachable floor would have bitten.
  # `anneal` is the old linear ent_coef -> ent_coef_end schedule, kept so
  # earlier runs stay reproducible. `fixed` holds ent_coef constant.
  ent_coef_mode: adaptive    # adaptive | anneal | fixed
  ent_coef: 0.1              # initial value (adaptive) / schedule start (anneal)
  ent_coef_end: 0.01         # anneal only
  target_entropy_frac: 0.6   # adaptive only
  ent_coef_lr: 0.0003        # adaptive only
  ent_coef_min: 0.001
  ent_coef_max: 0.5
```

In the commented-out MAPPO block, replace the line
`#   ent_coef: 0.01           # Constant -- MAPPO has no ent_coef_end.` with:

```yaml
#   ent_coef_mode: adaptive  # Same three modes as IPPO, one shared controller.
#   ent_coef: 0.1
#   ent_coef_end: 0.01
#   target_entropy_frac: 0.6
#   ent_coef_lr: 0.0003
#   ent_coef_min: 0.001
#   ent_coef_max: 0.5
```

In the `env:` block, replace the `agent_view_range` entry (lines 72-74, whose
comment still says "so 5 gives an 11x11x3 RGB observation") with:

```yaml
  # Half-width of an agent's egocentric view, so 7 gives a 15x15x3 RGB
  # observation. Matches the reference implementation.
  agent_view_range: 7

  # Observations stacked along the channel axis, so 2 gives 15x15x6. 1 leaves
  # the env unwrapped. Above 1 the policies, reward model and preference
  # buffer all widen with it -- and the buffer's resident size scales
  # linearly, so check the startup warning before raising this with a long
  # buffer.
  num_frames: 1
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_hydra_configs.py -v && uv run pytest -q`
Expected: PASS. `test_experiment_preset_composes[example]` covers that the edited YAML still type-checks against the schema.

- [ ] **Step 5: Commit**

```bash
git add src/commons_game_marp/configs/experiment/example.yaml tests/test_hydra_configs.py
git commit -m "docs(configs): document num_frames and the entropy control fields

example.yaml is what users copy from, and its claim that MAPPO has no
ent_coef_end is no longer true.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Verification

After Task 8, confirm the whole thing runs end to end rather than only in unit tests:

- [ ] **Full suite:** `uv run pytest -q` — all green.
- [ ] **Default smoke run:** `uv run commons-game train episodes=2 env=small logging.video_enabled=false` — completes, and the run's `config.yaml` shows `agent_view_range: 7`, `num_frames: 1`, `ent_coef_mode: adaptive`.
- [ ] **Frame-stacked run with the reward model on:** `uv run commons-game train episodes=2 env=medium algorithm=ippo reward_model=narrow_view env.num_frames=2 reward_model.max_episodes_in_buffer=50 logging.video_enabled=false` — completes, and the buffer warning does **not** fire at 50 episodes.
- [ ] **Controller responds:** in that run's `metrics.jsonl`, confirm `ent_coef_per_agent`, `entropy_per_agent`, and `target_entropy` are present and that `target_entropy` is ~1.2478.
- [ ] **Anneal reproduces the old path:** `uv run commons-game train episodes=2 env=medium algorithm=ippo algorithm.ent_coef_mode=anneal logging.video_enabled=false` — `ent_coef` in `metrics.jsonl` starts at ~0.1 and decreases.
- [ ] **MAPPO still trains:** `uv run commons-game train episodes=2 env=small algorithm=mappo logging.video_enabled=false` — completes with `ent_coef` and `target_entropy` in its metrics.
