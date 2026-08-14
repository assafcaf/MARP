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


def _steps_to_leave_ceiling(controller, limit=100):
    """Steps of maximum-entropy feedback before the coefficient leaves its ceiling."""
    for step in range(1, limit + 1):
        controller.observe_entropy(2.079)
        if controller.coefficient() < 0.5:
            return step
    raise AssertionError(f"still pinned at the ceiling after {limit} steps")


def test_adaptive_recovery_does_not_depend_on_how_long_it_saturated():
    """The anti-windup guarantee. Clamping only the value `coefficient()`
    returns would let log_ent_coef keep integrating while pinned, so a longer
    saturation would take proportionally longer to unwind. Clamping the
    parameter itself bounds it, and recovery becomes constant -- Adam's
    momentum contributes a few steps of lag either way, which is why this
    asserts equality between two runs rather than a fixed step count."""
    brief = EntropyController(_config(ent_coef_lr=0.5), 8, CPU)
    for _ in range(200):
        brief.observe_entropy(0.0)
    assert brief.coefficient() == pytest.approx(0.5)

    long_haul = EntropyController(_config(ent_coef_lr=0.5), 8, CPU)
    for _ in range(2000):
        long_haul.observe_entropy(0.0)
    assert long_haul.coefficient() == pytest.approx(0.5)

    assert _steps_to_leave_ceiling(long_haul) == _steps_to_leave_ceiling(brief)


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


def test_mode_defaults_to_adaptive_when_the_field_is_absent():
    """The duck-typed getattr path is a last resort, and it must not land on
    the strategy this module exists to replace: a config that predates
    ent_coef_mode gets the controller, not the schedule that let entropy
    collapse unnoticed."""
    controller = EntropyController(SimpleNamespace(), 8, CPU)
    assert controller.mode == "adaptive"


def test_is_saturated_when_pinned_at_the_ceiling():
    """Entropy held well below target drives the coefficient up to ent_coef_max."""
    controller = EntropyController(_config(ent_coef_lr=0.5), 8, CPU)
    for _ in range(200):
        controller.observe_entropy(0.0)
    assert controller.coefficient() == pytest.approx(0.5)
    assert controller.is_saturated() is True


def test_is_saturated_when_pinned_at_the_floor():
    """Entropy held above target drives the coefficient down to ent_coef_min."""
    controller = EntropyController(_config(ent_coef_lr=0.5), 8, CPU)
    for _ in range(200):
        controller.observe_entropy(2.079)
    assert controller.coefficient() == pytest.approx(0.001)
    assert controller.is_saturated() is True


def test_is_not_saturated_away_from_the_clamps():
    controller = EntropyController(_config(), 8, CPU)
    assert controller.is_saturated() is False


def test_is_saturated_is_always_false_outside_adaptive_mode():
    for mode in ("fixed", "anneal"):
        controller = EntropyController(_config(ent_coef_mode=mode), 8, CPU)
        assert controller.is_saturated() is False


OBSERVATIONS_PER_EPISODE = 10  # IPPO defaults: ~10 minibatches per episode per agent


def test_shipped_default_responds_within_the_window_a_collapse_takes():
    """Calibration guard, not a unit test of the control law.

    `observe_entropy` runs once per minibatch -- about ten times an episode at
    IPPO's defaults, not the thousands per episode SAC's 3e-4 convention
    assumes. At 3e-4 the coefficient moved 0.3% an episode, which against the
    diagnostic run's collapse (entropy 2.08 -> 0.64 across roughly a hundred
    episodes) would have been close to a no-op. This pins the shipped default
    to a response fast enough to matter on that timescale.
    """
    from commons_game_marp.train.config import IPPOConfig

    controller = EntropyController(IPPOConfig(), 8, CPU)
    start = controller.coefficient()

    for _ in range(50 * OBSERVATIONS_PER_EPISODE):
        controller.observe_entropy(0.5)

    assert controller.coefficient() > 2 * start, (
        f"coefficient only reached {controller.coefficient():.4f} from {start:.4f} "
        "after 50 episodes of collapsed entropy -- too slow to catch a collapse"
    )


def test_getattr_fallbacks_match_the_declared_config_defaults():
    """Every duck-typed fallback must equal what IPPOConfig declares.

    The fallbacks are a last resort for a config that predates a field, and a
    last resort must not quietly select different behaviour from the default.
    This has now bitten twice -- once when the mode fallback was `anneal` while
    the default was `adaptive`, and once when the learning-rate fallback stayed
    at the value proven too slow to catch an entropy collapse. Comparing the
    two constructions directly retires the whole class rather than the two
    instances of it.
    """
    from commons_game_marp.train.config import IPPOConfig

    declared = EntropyController(IPPOConfig(), 8, CPU)
    fallback = EntropyController(SimpleNamespace(), 8, CPU)

    assert fallback.mode == declared.mode
    assert fallback.start == declared.start
    assert fallback.end == declared.end
    assert fallback.minimum == declared.minimum
    assert fallback.maximum == declared.maximum
    assert fallback.target_entropy == declared.target_entropy
    assert fallback.coefficient() == declared.coefficient()
