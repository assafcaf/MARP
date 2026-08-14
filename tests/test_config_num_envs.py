"""`episodes` stays a total episode budget, so runs stay comparable as
`num_envs` changes. A non-divisible pair silently truncates that budget, which
is exactly the kind of quiet difference that invalidates a comparison between
two runs -- so it is refused at startup instead.
"""

import pytest

from commons_game_marp.train.config import EnvConfig, resolve_iterations


def test_num_envs_defaults_to_one():
    assert EnvConfig().num_envs == 1


def test_iterations_divide_the_episode_budget():
    assert resolve_iterations(episodes=1000, num_envs=4) == 250


def test_single_env_runs_one_iteration_per_episode():
    assert resolve_iterations(episodes=17, num_envs=1) == 17


def test_non_divisible_budget_is_refused_with_workable_values():
    with pytest.raises(ValueError) as excinfo:
        resolve_iterations(episodes=1000, num_envs=3)

    message = str(excinfo.value)
    assert "999" in message and "1002" in message


def test_num_envs_below_one_is_refused():
    with pytest.raises(ValueError, match="num_envs"):
        resolve_iterations(episodes=100, num_envs=0)
