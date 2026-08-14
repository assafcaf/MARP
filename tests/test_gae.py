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
