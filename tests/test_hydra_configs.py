"""The Hydra config groups must compose into the dataclasses Trainer expects."""

import pytest
from hydra import compose, initialize_config_module
from omegaconf import OmegaConf

from commons_game_marp.train.config import TrainerConfig, register_configs

register_configs()


def _compose(*overrides):
    with initialize_config_module(version_base="1.3", config_module="commons_game_marp.configs"):
        cfg = compose(config_name="config", overrides=list(overrides))
    return OmegaConf.to_object(cfg)


@pytest.mark.parametrize("algo", ["dqn", "ippo", "mappo", "random"])
def test_algorithm_group_selects_one_node(algo):
    """The selected algorithm node carries its own name and nothing else's."""
    config = _compose(f"algorithm={algo}")

    assert isinstance(config, TrainerConfig)
    assert config.algorithm.name == algo
    # The old container held all three sections simultaneously. It must not.
    for other in ("dqn", "ippo", "mappo"):
        if other != algo:
            assert not hasattr(config.algorithm, other)


def test_every_algorithm_defines_normalize_obs():
    """`_format_reward_obs` reads this off the selected node with no fallback,
    so every algorithm -- including `random`, which used to borrow the dqn
    section -- must define it explicitly."""
    for algo in ("dqn", "ippo", "mappo", "random"):
        config = _compose(f"algorithm={algo}")
        assert isinstance(config.algorithm.normalize_obs, bool)


def test_random_normalizes_observations():
    """Regression guard: `random` has no config section of its own and used to
    silently resolve to the dqn section's scale. It now sets normalize_obs
    itself, and the value must stay True to match what the reward model was
    trained on."""
    config = _compose("algorithm=random")
    assert config.algorithm.normalize_obs is True


def test_unknown_key_is_rejected():
    """Structured configs mean a typo is a startup error, not a silent default."""
    with pytest.raises(Exception):
        _compose("algorithm.lerning_rate=0.1")
