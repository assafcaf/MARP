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
