"""Orthogonal initialization must use layer-wise gains."""

import inspect
from unittest.mock import patch

import numpy as np
import torch.nn as nn

from src.train.algorithms.ippo import orthogonal_init


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


def test_mappo_imports_orthogonal_init():
    """MAPPO must use the same initialization scheme as IPPO."""
    import src.train.algorithms.mappo as mappo

    assert hasattr(mappo, "orthogonal_init"), (
        "mappo.py should import orthogonal_init from ippo.py"
    )


PPO_ADAM_EPS = 1e-5


def test_ippo_optimizer_uses_ppo_adam_eps():
    source = inspect.getsource(__import__("src.train.algorithms.ippo", fromlist=["x"]))
    assert "eps=" in source, "IPPO's Adam should set eps explicitly"
    assert str(PPO_ADAM_EPS) in source or "1e-5" in source


def test_mappo_optimizer_uses_ppo_adam_eps():
    source = inspect.getsource(__import__("src.train.algorithms.mappo", fromlist=["x"]))
    assert "eps=" in source, "MAPPO's Adam should set eps explicitly"
    assert str(PPO_ADAM_EPS) in source or "1e-5" in source


def test_dqn_does_not_use_ppo_adam_eps():
    """DQN keeps PyTorch's default 1e-8; SB3 does not override it for DQN."""
    source = inspect.getsource(__import__("src.train.algorithms.dqn", fromlist=["x"]))
    assert "1e-5" not in source


def test_dqn_config_exposes_max_grad_norm():
    from src.train.config import DQNConfig

    assert hasattr(DQNConfig(), "max_grad_norm")
    assert DQNConfig().max_grad_norm == 10.0, "SB3's DQN default is 10"


def test_dqn_train_step_clips_gradients():
    source = inspect.getsource(__import__("src.train.algorithms.dqn", fromlist=["x"]))
    assert "clip_grad_norm_" in source, "DQN's train_step must clip gradients"
