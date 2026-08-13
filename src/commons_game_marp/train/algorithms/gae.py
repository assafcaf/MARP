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
