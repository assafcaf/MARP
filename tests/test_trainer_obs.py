"""The reward model must normalize observations the same way the policy does."""

import numpy as np
import pytest

from commons_game_marp.train.config import DQNConfig, IPPOConfig, MAPPOConfig, RandomConfig
from commons_game_marp.train.trainer import Trainer


class _ConfigStub:
    def __init__(self, algorithm):
        self.algorithm = algorithm


def _format(algorithm) -> np.ndarray:
    stub = object.__new__(Trainer)
    stub.config = _ConfigStub(algorithm)
    obs = {"agent-0": {"curr_obs": np.full((3, 3, 3), 255, dtype=np.uint8)}}
    return Trainer._format_reward_obs(stub, obs, "agent-0")


@pytest.mark.parametrize("cls", [DQNConfig, IPPOConfig, MAPPOConfig, RandomConfig])
def test_normalization_follows_the_selected_algorithm(cls):
    algorithm = cls()

    algorithm.normalize_obs = True
    assert _format(algorithm).max() == pytest.approx(1.0)

    algorithm.normalize_obs = False
    assert _format(algorithm).max() == pytest.approx(255.0)


def test_random_policy_normalizes_by_default():
    """Regression guard: `random` has no hyperparameters of its own and used to
    fall through to the dqn section. Its default must stay True so reward-model
    inputs match the data the model was trained on."""
    assert _format(RandomConfig()).max() == pytest.approx(1.0)
