"""The vectorised colour conversion must match the loop it replaced exactly.

`map_to_colors` was the single largest cost inside `step` -- a dict lookup and a
three-element assignment per cell, 225 of each per agent view per step. The
replacement is a lookup table, and the only thing that matters is that it is
indistinguishable from the original.
"""

import numpy as np
import pytest

from commons_game_marp.env.commons_env import HarvestCommonsEnv
from commons_game_marp.env.constants import DIFFERENT_COLORMAP, SAME_COLORMAP
from commons_game_marp.env.maps import MEDIUM_HARVEST_MAP, SMALL_HARVEST_MAP


def reference(grid, color_map):
    """The original nested-loop implementation, kept as the oracle."""
    rgb = np.zeros((grid.shape[0], grid.shape[1], 3), dtype=np.uint8)
    for row in range(grid.shape[0]):
        for col in range(grid.shape[1]):
            rgb[row, col, :] = color_map[grid[row, col]]
    return rgb


@pytest.fixture
def env():
    e = HarvestCommonsEnv(
        ascii_map=SMALL_HARVEST_MAP, num_agents=2, agent_view_range=3, ep_length=30
    )
    e.reset(seed=0)
    return e


class TestMatchesTheReference:
    @pytest.mark.parametrize("color_map", [DIFFERENT_COLORMAP, SAME_COLORMAP])
    def test_on_random_grids(self, env, color_map):
        chars = [c for c in color_map if c != '']
        rng = np.random.default_rng(0)
        for _ in range(30):
            shape = (int(rng.integers(1, 12)), int(rng.integers(1, 12)))
            grid = rng.choice(np.array(chars, dtype='<U1'), size=shape)
            expected = reference(grid.copy(), color_map)
            actual = env.map_to_colors(grid.copy(), color_map, full_map=True)
            np.testing.assert_array_equal(actual, expected)

    def test_on_the_real_world_map(self, env):
        grid = env.get_map_with_agents()
        np.testing.assert_array_equal(
            env.map_to_colors(grid.copy(), env.color_map, full_map=True),
            reference(grid.copy(), env.color_map),
        )

    def test_empty_string_cells_map_to_their_colour(self, env):
        """`''` is a real key in the colour dicts and numpy stores an empty
        character cell as code point 0, not as a space."""
        grid = np.full((3, 3), '', dtype='<U1')
        np.testing.assert_array_equal(
            env.map_to_colors(grid.copy(), DIFFERENT_COLORMAP, full_map=True),
            reference(grid.copy(), DIFFERENT_COLORMAP),
        )

    def test_full_map_false_marks_the_centre(self, env):
        """The 'S' self-marker write must survive, mutation included."""
        grid = np.full((5, 5), ' ', dtype='<U1')
        actual = env.map_to_colors(grid, DIFFERENT_COLORMAP, full_map=False)
        assert grid[2, 2] == 'S'
        np.testing.assert_array_equal(actual[2, 2], DIFFERENT_COLORMAP['S'])

    def test_output_dtype_and_shape(self, env):
        grid = np.full((4, 6), 'A', dtype='<U1')
        out = env.map_to_colors(grid.copy(), DIFFERENT_COLORMAP, full_map=True)
        assert out.shape == (4, 6, 3)
        assert out.dtype == np.uint8


class TestUnknownCharacters:
    def test_raises_like_the_dict_lookup_did(self, env):
        grid = np.full((2, 2), 'Z', dtype='<U1')
        with pytest.raises(KeyError, match="Z"):
            env.map_to_colors(grid, DIFFERENT_COLORMAP, full_map=True)

    def test_raises_for_a_code_point_beyond_the_table(self, env):
        grid = np.full((2, 2), '中', dtype='<U1')
        with pytest.raises(KeyError):
            env.map_to_colors(grid, DIFFERENT_COLORMAP, full_map=True)


class TestCaching:
    def test_table_is_reused_for_the_same_dict(self, env):
        env.map_to_colors(env.get_map_with_agents(), env.color_map, full_map=True)
        first = env._colour_lut_cache
        env.map_to_colors(env.get_map_with_agents(), env.color_map, full_map=True)
        assert env._colour_lut_cache is first

    def test_table_is_rebuilt_for_a_different_dict(self, env):
        env.map_to_colors(env.get_map_with_agents(), DIFFERENT_COLORMAP, full_map=True)
        lut_a = env._colour_lut_cache[1].copy()
        env.map_to_colors(env.get_map_with_agents(), SAME_COLORMAP, full_map=True)
        assert not np.array_equal(lut_a, env._colour_lut_cache[1])


def _legacy_map_to_colors(self, map=None, color_map=None, full_map=False):
    """The exact pre-optimisation implementation, for a like-for-like run."""
    if map is None:
        map = self.get_map_with_agents()
    if color_map is None:
        color_map = self.color_map
    if not full_map:
        map[map.shape[0] // 2, map.shape[1] // 2] = 'S'
    rgb_arr = np.zeros((map.shape[0], map.shape[1], 3), dtype=np.uint8)
    for row_elem in range(map.shape[0]):
        for col_elem in range(map.shape[1]):
            rgb_arr[row_elem, col_elem, :] = color_map[map[row_elem, col_elem]]
    return rgb_arr


def _episode_frames(env, steps=25, seed=4):
    observations, _ = env.reset(seed=seed)
    rng = np.random.default_rng(1)
    frames = [
        {a: o["curr_obs"].copy() for a, o in observations.items()}
    ]
    for _ in range(steps):
        actions = {a: int(x) for a, x in zip(env.agents, rng.integers(0, 8, 3))}
        observations, _, _, _ = env.step(actions)
        frames.append({a: o["curr_obs"].copy() for a, o in observations.items()})
    return frames


def test_every_observation_matches_the_old_implementation(monkeypatch):
    """The real proof: run the same seeded episode both ways, compare frames.

    Not just the conversion in isolation -- `map_to_colors` mutates its input to
    write the 'S' self-marker, and `return_view` can hand it a slice of the
    shared agent grid, so a difference in *when* cells are written would show up
    here and nowhere else.
    """
    def build():
        return HarvestCommonsEnv(
            ascii_map=MEDIUM_HARVEST_MAP, num_agents=3, agent_view_range=5,
            ep_length=40,
        )

    new_frames = _episode_frames(build())

    monkeypatch.setattr(HarvestCommonsEnv, "map_to_colors", _legacy_map_to_colors)
    old_frames = _episode_frames(build())

    assert len(new_frames) == len(old_frames)
    for step, (new, old) in enumerate(zip(new_frames, old_frames)):
        assert new.keys() == old.keys()
        for agent_id in new:
            np.testing.assert_array_equal(
                new[agent_id], old[agent_id],
                err_msg=f"step {step}, {agent_id}",
            )
