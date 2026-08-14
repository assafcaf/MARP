"""`reset(seed=...)` must determine the episode.

This is the property the whole `seed` config key promises, and it did not hold:
`spawn_point` shuffled the environment's own spawn-point list in place, so the
seed fixed the shuffle but not the order it was shuffling. Reproducibility is
also what makes the subprocess vector env verifiable at all -- without it,
worker output cannot be compared against in-process output.
"""

import numpy as np
import pytest

from commons_game_marp.env.commons_env import HarvestCommonsEnv
from commons_game_marp.env.maps import MEDIUM_HARVEST_MAP, SMALL_HARVEST_MAP

STAND_STILL = 4


def _make(num_agents=3, ascii_map=SMALL_HARVEST_MAP):
    return HarvestCommonsEnv(
        ascii_map=ascii_map, num_agents=num_agents, agent_view_range=3, ep_length=30
    )


def _layout(env):
    return [agent.get_pos().tolist() for agent in env.agents.values()]


class TestResetIsReproducible:
    def test_same_env_same_seed_gives_the_same_layout(self):
        env = _make()
        layouts = []
        for _ in range(4):
            env.reset(seed=7)
            layouts.append(_layout(env))
        assert all(layout == layouts[0] for layout in layouts), layouts

    def test_two_envs_same_seed_agree(self):
        """Separately constructed envs must agree -- construction consumes the
        global RNG stream, so this only holds if the seed fully determines the
        reset."""
        first, second = _make(), _make()
        first.reset(seed=11)
        second.reset(seed=11)
        assert _layout(first) == _layout(second)

    def test_different_seeds_still_differ(self):
        """Reproducible must not mean constant."""
        env = _make()
        env.reset(seed=1)
        one = _layout(env)
        env.reset(seed=2)
        two = _layout(env)
        assert one != two

    def test_observations_reproduce(self):
        first, second = _make(), _make()
        a, _ = first.reset(seed=3)
        b, _ = second.reset(seed=3)
        for agent_id in first.agents:
            np.testing.assert_array_equal(
                a[agent_id]["curr_obs"], b[agent_id]["curr_obs"]
            )

    def test_whole_episode_reproduces(self):
        """Rewards and social metrics, not just the first frame."""
        rng = np.random.default_rng(0)
        actions = [
            {f"agent-{i}": int(x) for i, x in enumerate(rng.integers(0, 8, 3))}
            for _ in range(30)
        ]
        traces = []
        for _ in range(2):
            env = _make()
            env.reset(seed=5)
            rewards = []
            for action in actions:
                _, step_rewards, _, _ = env.step(action)
                rewards.append(sorted(step_rewards.items()))
            env.compute_social_metrics()
            traces.append((rewards, env.get_social_metrics()))
        assert traces[0][0] == traces[1][0]
        assert traces[0][1] == traces[1][1]


class TestSpawnPointBehaviour:
    def test_spawn_points_list_is_not_mutated(self):
        """The in-place shuffle was the root cause; guard against its return."""
        env = _make()
        before = [list(p) for p in env.spawn_points]
        for seed in range(5):
            env.reset(seed=seed)
        assert [list(p) for p in env.spawn_points] == before

    def test_returns_a_free_cell(self):
        env = _make(num_agents=3)
        env.reset(seed=0)
        positions = _layout(env)
        assert len(positions) == len(set(map(tuple, positions))), positions

    def test_spawning_is_not_pinned_to_one_cell(self):
        """The missing `break` meant it always returned the last free point.
        Across seeds, the first agent must land in more than one place."""
        env = _make(ascii_map=MEDIUM_HARVEST_MAP)
        seen = set()
        for seed in range(15):
            env.reset(seed=seed)
            seen.add(tuple(_layout(env)[0]))
        assert len(seen) > 1, seen

    def test_raises_when_no_free_spawn_point_remains(self):
        env = _make(num_agents=2)
        env.reset(seed=0)
        # Park a fake agent on every spawn point.
        class _Occupied:
            def __init__(self, pos):
                self._pos = np.array(pos)

            def get_pos(self):
                return self._pos

        env.agents = {
            f"blocker-{i}": _Occupied(point)
            for i, point in enumerate(env.spawn_points)
        }
        with pytest.raises(AssertionError, match="not enough spawn points"):
            env.spawn_point()
