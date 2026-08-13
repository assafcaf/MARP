"""A run's Hydra output and its artifacts share one directory, grouped by config.

The layout is `logs/<config-dir>/<timestamp>[-seed=N]/`, so every repeat of one
configuration can be gathered with a single glob.
"""

import os

import pytest
from hydra import compose, initialize_config_module

from commons_game_marp.train.config import register_configs
from commons_game_marp.train.run_paths import extras_from_overrides, launch_timestamp

register_configs()


def _run_dir(*overrides):
    """`hydra.run.dir` as Hydra would create it for these CLI overrides."""
    with initialize_config_module(version_base="1.3", config_module="commons_game_marp.configs"):
        cfg = compose(
            config_name="config", overrides=list(overrides), return_hydra_config=True
        )
    # Only this node: the rest of the hydra config interpolates values that do
    # not exist until a job actually runs.
    return str(cfg.hydra.run.dir)


def _split(run_dir):
    """(config dir, run leaf) of a path under `logs/`."""
    relative = os.path.relpath(run_dir, "logs")
    group, leaf = os.path.split(relative)
    return group, leaf


def test_seeds_of_one_configuration_share_a_parent():
    """The point of the layout: `plot runs logs/<config>/*` gathers the seeds."""
    group_a, leaf_a = _split(_run_dir("seed=0"))
    group_b, leaf_b = _split(_run_dir("seed=1"))

    assert group_a == group_b
    assert leaf_a != leaf_b
    assert leaf_a.endswith("-seed=0") and leaf_b.endswith("-seed=1")


def test_config_dir_names_the_configuration():
    group, _ = _split(_run_dir("algorithm=ippo", "env=medium", "env.num_agents=5"))

    assert group == "ippo-map=medium-agents=5-rm=off"


def test_reward_model_mode_and_phi_appear_when_enabled():
    group, _ = _split(_run_dir("reward_model=narrow_view"))

    assert "rm=narrow_view" in group
    assert "phi=efficiency_x_peace" in group


def test_differing_algorithms_do_not_share_a_directory():
    ippo, _ = _split(_run_dir("algorithm=ippo"))
    mappo, _ = _split(_run_dir("algorithm=mappo"))

    assert ippo != mappo


def test_unencoded_override_splits_the_configuration():
    """`-m algorithm.learning_rate=1e-3,1e-4` must not write both jobs to one
    directory: the name does not otherwise mention the learning rate."""
    base, _ = _split(_run_dir())
    tweaked, _ = _split(_run_dir("algorithm.learning_rate=0.001"))

    assert tweaked != base
    assert "algorithm.learning_rate=0.001" in tweaked


def test_logging_overrides_do_not_split_the_configuration():
    """A run is not a different configuration for having had its videos off."""
    base, _ = _split(_run_dir())
    quiet, _ = _split(_run_dir("logging.video_enabled=false"))

    assert quiet == base


def test_run_name_replaces_the_derived_group():
    group, leaf = _split(_run_dir("logging.run_name=my_exp", "seed=2"))

    assert group == "my_exp"
    assert leaf.endswith("-seed=2")


def test_null_seed_leaves_the_seed_out_of_the_name():
    """A null seed is drawn after the directory name is fixed; config.yaml and
    run_info.json record what was actually used."""
    _, leaf = _split(_run_dir("seed=null"))

    assert leaf == launch_timestamp()


def test_log_dir_override_moves_the_whole_tree(tmp_path):
    run_dir = _run_dir(f"logging.log_dir={tmp_path}")

    assert run_dir.startswith(str(tmp_path))


@pytest.mark.parametrize(
    "override, expected",
    [
        ("algorithm=ippo", []),
        ("env.map_type=medium", []),
        ("reward_model.phi=efficiency", []),
        ("hydra.verbose=true", []),
        ("logging.log_dir=/tmp/x", []),
        ("episodes=5", ["episodes=5"]),
        ("+extra=1", ["extra=1"]),
        ("algorithm.hidden_size=64", ["algorithm.hidden_size=64"]),
    ],
)
def test_extras_keep_only_what_the_name_omits(override, expected):
    assert extras_from_overrides([override]) == expected


def test_extras_stay_within_a_sane_path_length():
    overrides = [f"episodes{i}={'x' * 60}" for i in range(10)]

    assert len("-".join(extras_from_overrides(overrides))) < 120
