"""Orthogonal initialization must use layer-wise gains, and PPO-family
optimizers/DQN gradient clipping must be configured as intended.

These tests assert on constructed objects (real tensors, real optimizer
param groups, real mock-call arguments) rather than on module source text or
module attributes, so a future refactor that silently drops or misindents
the behavior they guard will actually fail the suite.
"""

import math
from unittest.mock import patch

import numpy as np
import pytest
import torch.nn as nn

from commons_game_marp.train.algorithms.dqn import DQNAgent
from commons_game_marp.train.algorithms.ippo import IPPOAlgorithm, orthogonal_init
from commons_game_marp.train.algorithms.mappo import MAPPOAlgorithm
from commons_game_marp.train.config import IPPOConfig, MAPPOConfig
from tests.conftest import FakeEnv


def _three_layer_module() -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(3, 8, 3),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(8, 16),
        nn.ReLU(),
        nn.Linear(16, 4),
    )


def test_head_gain_applies_only_to_the_final_layer():
    module = _three_layer_module()
    with patch("torch.nn.init.orthogonal_") as mock_init:
        orthogonal_init(module, head_gain=0.01, trunk_gain=np.sqrt(2))

    gains = [call.kwargs["gain"] for call in mock_init.call_args_list]
    assert len(gains) == 3, "expected one init per Conv2d/Linear layer"
    assert gains[-1] == 0.01, "final layer must get the head gain"
    assert all(g == np.sqrt(2) for g in gains[:-1]), "trunk must get the trunk gain"


def test_trunk_is_not_initialised_with_the_head_gain():
    """Regression guard: the whole actor used to be initialised at gain 0.01."""
    module = _three_layer_module()
    with patch("torch.nn.init.orthogonal_") as mock_init:
        orthogonal_init(module, head_gain=0.01)

    gains = [call.kwargs["gain"] for call in mock_init.call_args_list]
    assert gains.count(0.01) == 1, "only the head may use gain 0.01"


def test_biases_are_zeroed():
    module = _three_layer_module()
    for m in module.modules():
        if isinstance(m, (nn.Linear, nn.Conv2d)) and m.bias is not None:
            nn.init.constant_(m.bias, 5.0)

    orthogonal_init(module, head_gain=0.01)

    for m in module.modules():
        if isinstance(m, (nn.Linear, nn.Conv2d)) and m.bias is not None:
            assert m.bias.abs().sum().item() == 0.0


def test_dqn_config_exposes_max_grad_norm():
    from commons_game_marp.train.config import DQNConfig

    assert hasattr(DQNConfig(), "max_grad_norm")
    assert DQNConfig().max_grad_norm == 10.0, "SB3's DQN default is 10"


# ---------------------------------------------------------------------------
# MAPPO orthogonal initialization: real tensors, both flatten_obs branches.
#
# `MAPPOAlgorithm.on_env_ready` calls `orthogonal_init(actor, head_gain=0.01)`
# and `orthogonal_init(critic, head_gain=1.0)`. The predecessor test only
# asserted `hasattr(mappo, "orthogonal_init")`, which would still pass if
# those two calls were deleted, mis-indented into only one of the two obs
# branches, or had their gains swapped -- exactly the kind of regression a
# future parameter-sharing refactor of this block could introduce silently
# (no error, just a worse learning curve, invisible without a training run).
#
# For an orthogonally-initialized weight tensor flattened to (rows, cols),
# nn.init.orthogonal_ makes either the rows or the columns exactly
# orthonormal (whichever axis is smaller), scaled by `gain`. Either way,
# trace(W @ W.T) == gain**2 * min(rows, cols) exactly (up to QR float
# precision), so mean(row_norm**2) == gain**2 * min(rows, cols) / rows is an
# exact, closed-form quantity that holds for both Linear (2D) and Conv2d
# (4D, flattened to (out_channels, in_channels*kh*kw)) layers alike --
# verified numerically against the real network shapes before writing this
# assertion (e.g. a (32, 3, 3, 3) Conv2d with gain sqrt(2) gives exactly
# 1.6875, matching the reviewer's independently measured value).
# ---------------------------------------------------------------------------


def _mean_row_norm_sq(layer: nn.Module) -> float:
    weight = layer.weight.detach()
    rows = weight.shape[0]
    flat = weight.reshape(rows, -1)
    return float((flat ** 2).sum() / rows)


def _expected_mean_row_norm_sq(layer: nn.Module, gain: float) -> float:
    weight = layer.weight.detach()
    rows = weight.shape[0]
    cols = weight.numel() // rows
    return gain ** 2 * min(rows, cols) / rows


def _assert_layer_wise_gains(net: nn.Module, head_gain: float, trunk_gain: float) -> None:
    layers = [m for m in net.modules() if isinstance(m, (nn.Linear, nn.Conv2d))]
    assert len(layers) >= 2, f"{net.__class__.__name__} should have trunk + head layers"
    for index, layer in enumerate(layers):
        gain = head_gain if index == len(layers) - 1 else trunk_gain
        actual = _mean_row_norm_sq(layer)
        expected = _expected_mean_row_norm_sq(layer, gain)
        assert actual == pytest.approx(expected, rel=1e-3), (
            f"{net.__class__.__name__} layer {index} ({layer}) does not match "
            f"gain {gain}: mean row-norm^2 = {actual}, expected {expected}"
        )


@pytest.mark.parametrize("flatten_obs", [True, False])
def test_mappo_orthogonal_init_applies_layer_wise_gains(flatten_obs):
    config = MAPPOConfig(flatten_obs=flatten_obs)
    algo = MAPPOAlgorithm(config)
    algo.on_env_ready(FakeEnv())

    _assert_layer_wise_gains(algo.actor, head_gain=0.01, trunk_gain=np.sqrt(2))
    _assert_layer_wise_gains(algo.critic, head_gain=1.0, trunk_gain=np.sqrt(2))


@pytest.mark.parametrize("flatten_obs", [True, False])
def test_ippo_orthogonal_init_applies_layer_wise_gains(flatten_obs):
    """Same guard as above, for IPPO's per-agent actor/critic networks."""
    config = IPPOConfig(flatten_obs=flatten_obs)
    algo = IPPOAlgorithm(config)
    algo.on_env_ready(FakeEnv())

    for agent_id in algo.agent_ids:
        _assert_layer_wise_gains(algo.actors[agent_id], head_gain=0.01, trunk_gain=np.sqrt(2))
        _assert_layer_wise_gains(algo.critics[agent_id], head_gain=1.0, trunk_gain=np.sqrt(2))


# ---------------------------------------------------------------------------
# PPO-family Adam eps: read back the real optimizer's param groups.
# ---------------------------------------------------------------------------


def test_ippo_optimizer_uses_ppo_adam_eps():
    config = IPPOConfig()
    algo = IPPOAlgorithm(config)
    algo.on_env_ready(FakeEnv())

    for optimizer in algo.optimizers.values():
        assert optimizer.param_groups[0]["eps"] == 1e-5


def test_mappo_optimizer_uses_ppo_adam_eps():
    config = MAPPOConfig()
    algo = MAPPOAlgorithm(config)
    algo.on_env_ready(FakeEnv())

    assert algo.optimizer.param_groups[0]["eps"] == 1e-5


def test_dqn_does_not_use_ppo_adam_eps():
    """DQN keeps PyTorch's default 1e-8; SB3 does not override it for DQN."""
    agent = DQNAgent(
        obs_shape=(15, 15, 3),
        num_actions=8,
        learning_rate=1e-3,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.1,
        epsilon_decay=0.995,
        batch_size=4,
        replay_buffer_size=100,
        target_update_freq=200,
        train_after=100,
        train_every=1,
        max_grad_norm=10.0,
        device="cpu",
    )
    assert agent.model.optimizer.param_groups[0]["eps"] == 1e-8


# ---------------------------------------------------------------------------
# DQN gradient clipping: spy on clip_grad_norm_ during a real train_step().
# ---------------------------------------------------------------------------


def test_dqn_train_step_clips_gradients():
    """`patch` wraps the real `clip_grad_norm_` (via `wraps=`) rather than
    replacing it, so gradients are still actually clipped during the step --
    this asserts both that clipping is invoked AND with what max_norm,
    without disabling the clipping it's supposed to be testing.
    """
    max_grad_norm = 0.5
    batch_size = 4
    agent = DQNAgent(
        obs_shape=(15, 15, 3),
        num_actions=8,
        learning_rate=1e-3,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.1,
        epsilon_decay=0.995,
        batch_size=batch_size,
        replay_buffer_size=100,
        target_update_freq=200,
        train_after=batch_size,
        train_every=1,
        max_grad_norm=max_grad_norm,
        device="cpu",
    )

    obs = np.zeros((15, 15, 3), dtype=np.float32)
    next_obs = np.zeros((15, 15, 3), dtype=np.float32)
    for _ in range(batch_size):
        agent.remember(obs, 0, 1.0, next_obs, False)

    with patch(
        "commons_game_marp.train.algorithms.dqn.nn.utils.clip_grad_norm_",
        wraps=nn.utils.clip_grad_norm_,
    ) as mock_clip:
        info = agent.train_step()

    assert info, "train_step should have run given enough transitions"
    mock_clip.assert_called_once()
    call = mock_clip.call_args
    called_max_norm = call.args[1] if len(call.args) > 1 else call.kwargs.get("max_norm")
    assert called_max_norm == max_grad_norm


def test_ippo_builds_one_entropy_controller_per_agent(fake_env):
    """IPPO's networks are per-agent, so its exploration pressure is too --
    which is what makes a diverging agent visible in ent_coef_per_agent."""
    from commons_game_marp.train.algorithms.ippo import IPPOAlgorithm
    from commons_game_marp.train.config import IPPOConfig

    algorithm = IPPOAlgorithm(IPPOConfig())
    algorithm.on_env_ready(fake_env)

    assert set(algorithm.ent_controllers) == set(fake_env.agent_ids)
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

    assert set(metrics["ent_coef_per_agent"]) == set(fake_env.agent_ids)
    assert metrics["ent_coef"] == pytest.approx(
        sum(metrics["ent_coef_per_agent"].values()) / len(fake_env.agent_ids)
    )
    assert metrics["target_entropy"] == pytest.approx(0.6 * math.log(8))
    assert metrics["entropy_per_agent"] == {"agent-0": 1.5, "agent-1": 0.5}


def test_ippo_per_agent_coefficients_diverge_with_per_agent_entropy(fake_env):
    """The central claim of IPPO's per-agent controller design: a freeloading
    agent's entropy diverging from its peers must show up as its coefficient
    diverging too, not as an averaged-away scalar. Drives agent-0's entropy
    below target and agent-1's above target directly through their own
    controllers, then checks the coefficients moved in opposite directions."""
    from commons_game_marp.train.algorithms.ippo import IPPOAlgorithm
    from commons_game_marp.train.config import IPPOConfig

    algorithm = IPPOAlgorithm(IPPOConfig())
    algorithm.on_env_ready(fake_env)

    target = next(iter(algorithm.ent_controllers.values())).target_entropy
    below_target = target - 0.5
    above_target = target + 0.5
    assert below_target > 0  # sanity: still a valid entropy value

    start = {
        agent_id: controller.coefficient()
        for agent_id, controller in algorithm.ent_controllers.items()
    }

    for _ in range(50):
        algorithm.ent_controllers["agent-0"].observe_entropy(below_target)
        algorithm.ent_controllers["agent-1"].observe_entropy(above_target)

    metrics = algorithm.on_episode_end(0)
    per_agent = metrics["ent_coef_per_agent"]

    # Below target -> coefficient rises. Above target -> coefficient falls.
    assert per_agent["agent-0"] > start["agent-0"]
    assert per_agent["agent-1"] < start["agent-1"]
    # The two controllers must actually have diverged from each other, not
    # just from their own starting points.
    assert per_agent["agent-0"] > per_agent["agent-1"]


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
