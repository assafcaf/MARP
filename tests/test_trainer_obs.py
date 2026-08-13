"""The reward model must normalize observations the same way the policy does."""

import numpy as np
import pytest

from commons_game_marp.train.config import DQNConfig, IPPOConfig, MAPPOConfig, RandomConfig
from commons_game_marp.train.trainer import Trainer


class _ConfigStub:
    def __init__(self, algorithm):
        self.algorithm = algorithm


def _stub(algorithm) -> Trainer:
    stub = object.__new__(Trainer)
    stub.config = _ConfigStub(algorithm)
    return stub


def _format(algorithm) -> np.ndarray:
    obs = {"agent-0": {"curr_obs": np.full((3, 3, 3), 255, dtype=np.uint8)}}
    return Trainer._format_reward_obs(_stub(algorithm), obs, "agent-0")


def _effective_max(algorithm) -> float:
    """The scale the model actually sees: stored frame x `obs_scale`.

    Frames are stored raw (uint8) and scaled inside `RewardModel.forward`, so
    the invariant this module guards has to be checked on the product, not on
    the stored array alone.
    """
    return float(_format(algorithm).max()) * Trainer._reward_obs_scale(_stub(algorithm))


@pytest.mark.parametrize("cls", [DQNConfig, IPPOConfig, MAPPOConfig, RandomConfig])
def test_normalization_follows_the_selected_algorithm(cls):
    algorithm = cls()

    algorithm.normalize_obs = True
    assert _effective_max(algorithm) == pytest.approx(1.0)

    algorithm.normalize_obs = False
    assert _effective_max(algorithm) == pytest.approx(255.0)


def test_random_policy_normalizes_by_default():
    """Regression guard: `random` has no hyperparameters of its own and used to
    fall through to the dqn section. Its default must stay True so reward-model
    inputs match the data the model was trained on."""
    assert _effective_max(RandomConfig()) == pytest.approx(1.0)


@pytest.mark.parametrize("cls", [DQNConfig, IPPOConfig, MAPPOConfig, RandomConfig])
def test_stored_observations_stay_uint8(cls):
    """Buffer residency is the binding memory constraint, so the frame handed
    to the preference buffer must not be widened to float32 on the way in."""
    assert _format(cls()).dtype == np.uint8


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
