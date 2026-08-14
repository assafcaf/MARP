"""`infos["state"]` is opt-in.

It is a whole-map RGB render produced once per agent per step. Nothing in this
repo reads it, it measured at roughly half of total step time, and once the
environments run in worker processes it also has to cross a pipe. So it is off
unless asked for -- and these tests pin both halves of that: absent by default,
and still correct when enabled.
"""

import numpy as np

from commons_game_marp.env.commons_env import HarvestCommonsEnv
from commons_game_marp.env.maps import SMALL_HARVEST_MAP

STAND_STILL = 4


def _make(include_state):
    env = HarvestCommonsEnv(
        ascii_map=SMALL_HARVEST_MAP,
        num_agents=2,
        ep_length=20,
        include_state_in_info=include_state,
    )
    return env


def test_absent_by_default():
    env = _make(include_state=False)
    _, reset_infos = env.reset()
    _, _, _, step_infos = env.step({a: STAND_STILL for a in env.agents})
    for infos in (reset_infos, step_infos):
        for agent_id, info in infos.items():
            assert "state" not in info, agent_id


def test_present_and_correct_when_enabled():
    env = _make(include_state=True)
    _, reset_infos = env.reset()
    for info in reset_infos.values():
        assert "state" in info
        assert info["state"].dtype == np.uint8
        assert info["state"].ndim == 3

    _, _, _, step_infos = env.step({a: STAND_STILL for a in env.agents})
    for info in step_infos.values():
        # The env's own `state` property is the reference.
        np.testing.assert_array_equal(info["state"], env.state)


def test_other_info_keys_are_unaffected():
    """The FIRE and reward keys the trainer depends on must still be there."""
    env = _make(include_state=False)
    env.reset()
    _, _, _, infos = env.step({a: STAND_STILL for a in env.agents})
    for info in infos.values():
        assert "r" in info
        assert "fire" in info


def test_observations_are_identical_either_way():
    """The flag must change only the info dict, never what the agent sees.

    Two separately seeded envs are directly comparable now that
    `reset(seed=...)` is reproducible (see tests/test_env_seeding.py).
    """
    frames = []
    for include_state in (False, True):
        env = _make(include_state=include_state)
        observations, _ = env.reset(seed=7)
        for _ in range(5):
            observations, _, _, _ = env.step({a: STAND_STILL for a in env.agents})
        frames.append(observations["agent-0"]["curr_obs"].copy())
    np.testing.assert_array_equal(frames[0], frames[1])

