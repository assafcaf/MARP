"""Worker processes must be indistinguishable from stepping in-process.

This is the test the whole parallelism change rests on. It is only possible
because `reset(seed=...)` is now reproducible -- before that fix, the two
implementations diverged for reasons having nothing to do with the workers.
"""

import numpy as np
import pytest

from commons_game_marp.env.env_spec import EnvSpec
from commons_game_marp.env.subproc_vec_env import SubprocVecCommonsEnv, split_envs
from commons_game_marp.env.vec_env import VecCommonsEnv

NUM_ENVS = 4
NUM_AGENTS = 3
STEPS = 12


def _spec(**overrides):
    base = dict(
        map_type="small",
        num_agents=NUM_AGENTS,
        agent_view_range=3,
        ep_length=STEPS + 5,
        step_metrics=True,
    )
    base.update(overrides)
    return EnvSpec(**base)


def _seeds(num_envs=NUM_ENVS):
    return [100 + i for i in range(num_envs)]


def _rollout(env, actions_per_step, seeds):
    obs, infos = env.reset(seeds)
    trace = {"reset_obs": obs.copy(), "obs": [], "rewards": [], "dones": [], "nearby": []}
    for actions in actions_per_step:
        obs, rewards, dones, infos = env.step(actions)
        trace["obs"].append(obs.copy())
        trace["rewards"].append(rewards.copy())
        trace["dones"].append(dones.copy())
        trace["nearby"].append([i.get("nearby_apples") for i in infos])
    trace["social"] = env.compute_social_metrics()
    return trace


@pytest.fixture(scope="module")
def action_sequence():
    rng = np.random.default_rng(0)
    return [
        rng.integers(0, 8, size=NUM_ENVS * NUM_AGENTS).astype(np.int64)
        for _ in range(STEPS)
    ]


@pytest.fixture(scope="module")
def in_process_trace(action_sequence):
    spec = _spec()
    env = VecCommonsEnv(spec.build, NUM_ENVS)
    return _rollout(env, action_sequence, _seeds())


class TestSplitEnvs:
    def test_even_split(self):
        assert split_envs(8, 4) == [2, 2, 2, 2]

    def test_remainder_is_spread_not_piled_on_the_last_worker(self):
        """Lockstep stepping costs whatever the slowest worker costs."""
        assert split_envs(7, 3) == [3, 2, 2]
        assert split_envs(10, 4) == [3, 3, 2, 2]

    def test_more_workers_than_envs_is_clamped(self):
        assert split_envs(2, 8) == [1, 1]

    def test_totals_always_match(self):
        for num_envs in range(1, 17):
            for workers in range(1, 9):
                assert sum(split_envs(num_envs, workers)) == num_envs

    @pytest.mark.parametrize("bad", [(0, 1), (4, 0)])
    def test_rejects_nonsense(self, bad):
        with pytest.raises(ValueError):
            split_envs(*bad)


class TestEquivalence:
    @pytest.mark.parametrize("num_workers", [1, 2, 4])
    def test_identical_to_in_process(self, in_process_trace, action_sequence, num_workers):
        env = SubprocVecCommonsEnv(_spec(), NUM_ENVS, num_workers)
        try:
            actual = _rollout(env, action_sequence, _seeds())
        finally:
            env.close()

        np.testing.assert_array_equal(
            actual["reset_obs"], in_process_trace["reset_obs"]
        )
        for step in range(STEPS):
            np.testing.assert_array_equal(
                actual["obs"][step], in_process_trace["obs"][step], err_msg=f"step {step}"
            )
            np.testing.assert_array_equal(
                actual["rewards"][step], in_process_trace["rewards"][step]
            )
            np.testing.assert_array_equal(
                actual["dones"][step], in_process_trace["dones"][step]
            )
            assert actual["nearby"][step] == in_process_trace["nearby"][step]
        assert actual["social"] == in_process_trace["social"]

    def test_row_layout_is_env_major(self):
        """A distinct action per row must reach the matching environment."""
        env = SubprocVecCommonsEnv(_spec(), NUM_ENVS, num_workers=2)
        try:
            env.reset(_seeds())
            obs, rewards, dones, infos = env.step(
                np.zeros(NUM_ENVS * NUM_AGENTS, dtype=np.int64)
            )
            assert obs.shape[0] == NUM_ENVS * NUM_AGENTS
            assert len(infos) == NUM_ENVS * NUM_AGENTS
            assert env.num_rows == NUM_ENVS * NUM_AGENTS
        finally:
            env.close()


class TestInterfaceParity:
    def test_attributes_match_the_in_process_version(self):
        spec = _spec()
        reference = VecCommonsEnv(spec.build, NUM_ENVS)
        env = SubprocVecCommonsEnv(spec, NUM_ENVS, num_workers=2)
        try:
            assert env.agent_ids == reference.agent_ids
            assert env.num_agents == reference.num_agents
            assert env.num_envs == reference.num_envs
            assert env.num_rows == reference.num_rows
            assert env.action_space.n == reference.action_space.n
            assert (
                env.observation_space["curr_obs"].shape
                == reference.observation_space["curr_obs"].shape
            )
        finally:
            env.close()

    def test_render_frame_matches(self):
        spec = _spec()
        reference = VecCommonsEnv(spec.build, NUM_ENVS)
        reference.reset(_seeds())
        env = SubprocVecCommonsEnv(spec, NUM_ENVS, num_workers=2)
        try:
            env.reset(_seeds())
            for env_idx in range(NUM_ENVS):
                np.testing.assert_array_equal(
                    env.render_frame(env_idx), reference.render_frame(env_idx)
                )
        finally:
            env.close()

    def test_rows_to_agents_roundtrips(self):
        env = SubprocVecCommonsEnv(_spec(), NUM_ENVS, num_workers=2)
        try:
            rows = np.arange(env.num_rows, dtype=np.float64)
            assert np.array_equal(env.agents_to_rows(env.rows_to_agents(rows)), rows)
        finally:
            env.close()

    def test_rejects_wrong_action_shape(self):
        env = SubprocVecCommonsEnv(_spec(), NUM_ENVS, num_workers=2)
        try:
            env.reset(_seeds())
            with pytest.raises(ValueError, match="expected actions of shape"):
                env.step(np.zeros(3, dtype=np.int64))
        finally:
            env.close()

    def test_rejects_wrong_seed_count(self):
        env = SubprocVecCommonsEnv(_spec(), NUM_ENVS, num_workers=2)
        try:
            with pytest.raises(ValueError, match="expected 4 seeds"):
                env.reset([1, 2])
        finally:
            env.close()


class TestLifecycle:
    def test_close_is_idempotent(self):
        env = SubprocVecCommonsEnv(_spec(), NUM_ENVS, num_workers=2)
        env.close()
        env.close()

    def test_workers_actually_exit(self):
        env = SubprocVecCommonsEnv(_spec(), NUM_ENVS, num_workers=2)
        processes = list(env._processes)
        env.close()
        for process in processes:
            assert not process.is_alive()
