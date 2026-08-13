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


def _experiment_names():
    import os

    from commons_game_marp import configs

    directory = os.path.join(os.path.dirname(configs.__file__), "experiment")
    return sorted(f[: -len(".yaml")] for f in os.listdir(directory) if f.endswith(".yaml"))


@pytest.mark.parametrize("preset", _experiment_names())
def test_experiment_preset_composes(preset):
    """Every preset must still compose against the schema.

    `example.yaml` is the annotated template users copy, so it spells out every
    key: a renamed or dropped field breaks it here rather than in someone's
    first run.
    """
    config = _compose(f"+experiment={preset}")

    assert isinstance(config, TrainerConfig)
    assert config.algorithm.name in ("dqn", "ippo", "mappo", "random")


def test_unknown_key_is_rejected():
    """Structured configs mean a typo is a startup error, not a silent default."""
    with pytest.raises(Exception):
        _compose("algorithm.lerning_rate=0.1")


def test_config_snapshot_round_trips(tmp_path):
    """The config.yaml written into each run directory must load back into an
    equivalent config -- it is the record of what a run actually used."""
    from omegaconf import OmegaConf

    config = _compose("algorithm=ippo", "env=medium")
    path = tmp_path / "config.yaml"
    OmegaConf.save(OmegaConf.structured(config), path)

    reloaded = OmegaConf.load(path)
    assert OmegaConf.to_container(reloaded) == OmegaConf.to_container(
        OmegaConf.structured(config)
    )


@pytest.mark.parametrize("env_group", ["medium", "small"])
def test_view_range_matches_the_reference_implementation(env_group):
    """DanfoaTestSOT runs agent_view_range: 7 (src/configs/prm.yaml). The
    observation is (2*view+1, 2*view+1, 3), so this is 15x15x3 rather than
    the 11x11x3 the earlier runs used."""
    config = _compose(f"env={env_group}")
    assert config.env.agent_view_range == 7


def test_env_config_dataclass_default_view_range():
    """The dataclass default must agree with the YAML: a programmatic
    Trainer(TrainerConfig()) bypasses Hydra entirely."""
    from commons_game_marp.train.config import EnvConfig

    assert EnvConfig().agent_view_range == 7


@pytest.mark.parametrize("env_group", ["medium", "small"])
def test_num_frames_defaults_to_one(env_group):
    """Opt-in by design: at 1 the trainer skips the wrapper entirely, so the
    default code path is unchanged rather than merely equivalent."""
    config = _compose(f"env={env_group}")
    assert config.env.num_frames == 1


def test_num_frames_is_overridable():
    config = _compose("env=medium", "env.num_frames=2")
    assert config.env.num_frames == 2


def test_example_experiment_documents_the_entropy_fields():
    """example.yaml is the reference users copy from. A field that exists in
    the schema but not here is a field nobody discovers."""
    import os

    from commons_game_marp import configs

    path = os.path.join(os.path.dirname(configs.__file__), "experiment", "example.yaml")
    with open(path, encoding="utf-8") as handle:
        text = handle.read()

    for field in (
        "ent_coef_mode",
        "target_entropy_frac",
        "ent_coef_lr",
        "ent_coef_min",
        "ent_coef_max",
        "num_frames",
    ):
        assert field in text, f"{field} is undocumented in example.yaml"


def test_example_experiment_does_not_claim_mappo_lacks_ent_coef_end():
    """MAPPO gained the full entropy field set; the old comment is now false."""
    import os

    from commons_game_marp import configs

    path = os.path.join(os.path.dirname(configs.__file__), "experiment", "example.yaml")
    with open(path, encoding="utf-8") as handle:
        text = handle.read()

    assert "MAPPO has no ent_coef_end" not in text
