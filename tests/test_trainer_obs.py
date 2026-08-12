"""The reward model must normalize observations the same way the policy does."""

import numpy as np
import pytest

from src.train.config import AlgorithmConfig
from src.train.trainer import Trainer


class _ConfigStub:
    def __init__(self, algorithm: AlgorithmConfig):
        self.algorithm = algorithm


def _format(trainer_self, algo_name: str, normalize: bool) -> np.ndarray:
    algorithm = AlgorithmConfig(name=algo_name)
    getattr(algorithm, algo_name).normalize_obs = normalize
    trainer_self.config = _ConfigStub(algorithm)
    obs = {"agent-0": {"curr_obs": np.full((3, 3, 3), 255, dtype=np.uint8)}}
    return Trainer._format_reward_obs(trainer_self, obs, "agent-0")


@pytest.mark.parametrize("algo_name", ["dqn", "ippo", "mappo"])
def test_normalization_follows_the_active_algorithm(algo_name):
    stub = object.__new__(Trainer)

    normalized = _format(stub, algo_name, normalize=True)
    assert normalized.max() == pytest.approx(1.0)

    raw = _format(stub, algo_name, normalize=False)
    assert raw.max() == pytest.approx(255.0)


def test_dqn_setting_does_not_leak_into_ippo():
    """Regression guard: normalization used to read the dqn section always."""
    stub = object.__new__(Trainer)
    algorithm = AlgorithmConfig(name="ippo")
    algorithm.dqn.normalize_obs = False   # must be ignored
    algorithm.ippo.normalize_obs = True   # must be honoured
    stub.config = _ConfigStub(algorithm)

    obs = {"agent-0": {"curr_obs": np.full((3, 3, 3), 255, dtype=np.uint8)}}
    result = Trainer._format_reward_obs(stub, obs, "agent-0")
    assert result.max() == pytest.approx(1.0)
