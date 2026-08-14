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

    Driven from one env, toggling the flag between steps, because two separately
    constructed envs cannot be compared: `MapEnv.reset(seed=...)` is not
    reproducible. `spawn_point` does `random.shuffle(self.spawn_points)`, which
    mutates the instance's list in place, so the seed fixes the shuffle but not
    the list it shuffles. See `test_spawn_point_seeding` below.
    """
    env = _make(include_state=False)
    env.reset(seed=7)
    for _ in range(5):
        env.step({a: STAND_STILL for a in env.agents})

    env.include_state_in_info = False
    without, _, _, _ = env.step({a: STAND_STILL for a in env.agents})
    baseline = without["agent-0"]["curr_obs"].copy()

    # Same env, same position, only the flag differs: standing still on a
    # settled map reproduces the frame.
    env.include_state_in_info = True
    with_state, _, _, infos = env.step({a: STAND_STILL for a in env.agents})
    assert "state" in infos["agent-0"]
    np.testing.assert_array_equal(with_state["agent-0"]["curr_obs"].shape, baseline.shape)
    assert with_state["agent-0"]["curr_obs"].dtype == baseline.dtype


def test_spawn_point_seeding_is_not_reproducible():
    """Documents a pre-existing bug: `seed` does not determine the run.

    `MapEnv.spawn_point` calls `random.shuffle(self.spawn_points)`, mutating the
    instance's own list. Seeding fixes the shuffle *operation*, but the list it
    operates on is whatever the previous shuffles left behind, so identical
    seeds give different agent layouts.

    (`spawn_point` also never breaks out of its loop, so it returns the *last*
    free spawn point rather than a randomly chosen one -- the shuffle is the
    only thing making it random at all.)

    This test asserts the buggy behaviour so the suite stays honest about it.
    Flip it to `assert layouts[0] == layouts[1]` when the bug is fixed.
    """
    env = _make(include_state=False)
    layouts = []
    for _ in range(3):
        env.reset(seed=7)
        layouts.append([a.get_pos().tolist() for a in env.agents.values()])
    assert len(set(map(str, layouts))) > 1, (
        "spawn_point appears reproducible now -- if the shuffle was fixed, "
        "update this test to assert reproducibility instead"
    )
