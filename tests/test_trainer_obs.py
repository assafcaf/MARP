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
    # One flat row per (env, agent); row 0 is env 0's first agent.
    rows = np.full((1, 3, 3, 3), 255, dtype=np.uint8)
    return Trainer._format_reward_obs(_stub(algorithm), rows, 0)


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


def test_make_single_env_returns_the_bare_env_at_one_frame():
    from commons_game_marp.env.commons_env import HarvestCommonsEnv
    from commons_game_marp.train.config import TrainerConfig

    config = TrainerConfig()
    config.env.num_frames = 1
    config.env.num_agents = 2
    config.env.map_type = "small"

    stub = object.__new__(Trainer)
    stub.config = config
    env = Trainer._make_single_env(stub)

    assert isinstance(env, HarvestCommonsEnv)


@pytest.mark.parametrize("num_frames", [0, -1])
def test_make_single_env_rejects_non_positive_num_frames(num_frames):
    """A typo'd 0 or -1 used to fall through the getattr default and return
    the bare env with no complaint. num_frames is a declared dataclass field
    now, so read it directly and reject anything below 1."""
    from commons_game_marp.train.config import TrainerConfig

    config = TrainerConfig()
    config.env.num_frames = num_frames
    config.env.num_agents = 2
    config.env.map_type = "small"

    stub = object.__new__(Trainer)
    stub.config = config

    with pytest.raises(ValueError, match="num_frames"):
        Trainer._make_single_env(stub)


def test_make_single_env_wraps_and_widens_the_observation_space_above_one_frame():
    from commons_game_marp.env.frame_stack import FrameStackEnv
    from commons_game_marp.train.config import TrainerConfig

    config = TrainerConfig()
    config.env.num_frames = 2
    config.env.num_agents = 2
    config.env.map_type = "small"

    stub = object.__new__(Trainer)
    stub.config = config
    env = Trainer._make_single_env(stub)

    assert isinstance(env, FrameStackEnv)
    height, width, channels = env.observation_space["curr_obs"].shape
    assert (height, width) == (15, 15)
    assert channels == 6


def test_build_env_wraps_num_envs_copies_of_the_configured_env():
    """`_build_env` is now the vec layer; `_make_single_env` is the copy."""
    from commons_game_marp.env.frame_stack import FrameStackEnv
    from commons_game_marp.env.vec_env import VecCommonsEnv
    from commons_game_marp.train.config import TrainerConfig

    config = TrainerConfig()
    config.env.num_frames = 2
    config.env.num_agents = 2
    config.env.map_type = "small"
    config.env.num_envs = 3

    stub = object.__new__(Trainer)
    stub.config = config
    env = Trainer._build_env(stub)

    assert isinstance(env, VecCommonsEnv)
    assert env.num_envs == 3
    assert env.num_rows == 6
    # Each copy is wrapped exactly as a single env would be.
    assert all(isinstance(e, FrameStackEnv) for e in env.envs)


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


def _saturation_stub():
    stub = object.__new__(Trainer)
    stub.console = _WarnCapture()
    stub.SATURATION_WARN_EPISODES = Trainer.SATURATION_WARN_EPISODES
    stub._saturated_episodes = 0
    return stub


def test_watch_entropy_saturation_warns_exactly_once_at_the_threshold():
    stub = _saturation_stub()
    metrics = {"ent_coef_saturated": 1.0, "entropy": 0.5, "target_entropy": 1.25}

    for episode in range(Trainer.SATURATION_WARN_EPISODES + 5):
        Trainer._watch_entropy_saturation(stub, metrics)

    assert len(stub.console.messages) == 1
    message = stub.console.messages[0]
    assert "0.50" in message
    assert "1.25" in message


def test_watch_entropy_saturation_resets_when_entropy_recovers_above_target():
    stub = _saturation_stub()
    below = {"ent_coef_saturated": 1.0, "entropy": 0.5, "target_entropy": 1.25}
    above = {"ent_coef_saturated": 1.0, "entropy": 1.5, "target_entropy": 1.25}

    for _ in range(Trainer.SATURATION_WARN_EPISODES - 1):
        Trainer._watch_entropy_saturation(stub, below)
    Trainer._watch_entropy_saturation(stub, above)
    assert stub._saturated_episodes == 0

    for _ in range(Trainer.SATURATION_WARN_EPISODES - 1):
        Trainer._watch_entropy_saturation(stub, below)
    assert stub.console.messages == []


def test_watch_entropy_saturation_silent_when_not_saturated():
    stub = _saturation_stub()
    metrics = {"ent_coef_saturated": 0.0, "entropy": 0.5, "target_entropy": 1.25}

    for _ in range(Trainer.SATURATION_WARN_EPISODES + 5):
        Trainer._watch_entropy_saturation(stub, metrics)

    assert stub.console.messages == []


def test_watch_entropy_saturation_silent_when_keys_are_absent():
    stub = _saturation_stub()

    for _ in range(Trainer.SATURATION_WARN_EPISODES + 5):
        Trainer._watch_entropy_saturation(stub, {})

    assert stub.console.messages == []
