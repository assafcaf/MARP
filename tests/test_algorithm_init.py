"""Orthogonal initialization must use layer-wise gains."""

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
