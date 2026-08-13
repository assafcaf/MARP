"""Sampling, subsampling and residency behaviour of the preference buffer."""

import numpy as np
import pytest

from commons_game_marp.reward_model.preference_buffer import EpisodeRecord, PreferenceBuffer


OBS_SHAPE = (4, 4, 3)


def _episode(steps: int = 10, agents: int = 2, tag: int = 0) -> EpisodeRecord:
    return EpisodeRecord(
        agent_trajs={
            f"agent-{a}": [
                (np.full(OBS_SHAPE, tag, dtype=np.uint8), s) for s in range(steps)
            ]
            for a in range(agents)
        },
        metrics={"efficiency": float(tag)},
    )


class TestPairSampling:
    def test_pairs_are_always_two_distinct_episodes(self):
        buffer = PreferenceBuffer(max_episodes=8)
        for i in range(4):
            buffer.add_episode(_episode(tag=i))

        pairs = buffer.sample_episode_pairs(200)

        assert len(pairs) == 200
        assert all(a is not b for a, b in pairs)

    def test_sampling_covers_the_whole_buffer(self):
        buffer = PreferenceBuffer(max_episodes=8)
        for i in range(4):
            buffer.add_episode(_episode(tag=i))

        seen = {ep.metrics["efficiency"] for pair in buffer.sample_episode_pairs(300) for ep in pair}

        assert seen == {0.0, 1.0, 2.0, 3.0}

    @pytest.mark.parametrize("count, requested", [(0, 4), (1, 4), (4, 0)])
    def test_degenerate_requests_return_nothing(self, count, requested):
        buffer = PreferenceBuffer(max_episodes=8)
        for i in range(count):
            buffer.add_episode(_episode(tag=i))

        assert buffer.sample_episode_pairs(requested) == []

    def test_new_episodes_become_visible_to_sampling(self):
        """The episode snapshot is cached; adding must invalidate it."""
        buffer = PreferenceBuffer(max_episodes=8)
        buffer.add_episode(_episode(tag=0))
        buffer.add_episode(_episode(tag=1))
        buffer.sample_episode_pairs(4)

        buffer.add_episode(_episode(tag=2))
        seen = {ep.metrics["efficiency"] for pair in buffer.sample_episode_pairs(200) for ep in pair}

        assert 2.0 in seen

    def test_evicted_episodes_stop_being_sampled(self):
        buffer = PreferenceBuffer(max_episodes=2)
        for i in range(3):
            buffer.add_episode(_episode(tag=i))

        seen = {ep.metrics["efficiency"] for pair in buffer.sample_episode_pairs(200) for ep in pair}

        assert len(buffer) == 2
        assert seen == {1.0, 2.0}


class TestSubsampling:
    def test_aggregate_merges_agents_in_a_stable_order(self):
        buffer = PreferenceBuffer(max_episodes=4, max_steps_per_sequence=None)
        record = _episode(steps=3, agents=2)

        assert [action for _, action in buffer.aggregate_episode(record)] == [0, 1, 2, 0, 1, 2]

    def test_aggregate_respects_the_sequence_cap(self):
        buffer = PreferenceBuffer(max_episodes=4, max_steps_per_sequence=5)
        assert len(buffer.aggregate_episode(_episode(steps=20, agents=2))) == 5

    def test_short_sequences_are_returned_untouched(self):
        buffer = PreferenceBuffer(max_episodes=4, max_steps_per_sequence=100)
        assert len(buffer.aggregate_episode(_episode(steps=3, agents=2))) == 6

    def test_sampled_agent_trajectory_respects_the_cap(self):
        buffer = PreferenceBuffer(max_episodes=4, max_steps_per_sequence=4)
        assert len(buffer.sample_agent_trajectory(_episode(steps=30))) == 4

    def test_agent_trajectory_of_an_empty_record_is_empty(self):
        buffer = PreferenceBuffer(max_episodes=4)
        assert buffer.sample_agent_trajectory(EpisodeRecord(agent_trajs={}, metrics={})) == []


class TestResidency:
    def test_store_cap_shrinks_episodes_at_insertion(self):
        """Capping at insertion is the only thing that actually bounds memory;
        capping at sample time still pays for the full episode in RAM."""
        buffer = PreferenceBuffer(max_episodes=4, store_max_steps_per_agent=5)
        buffer.add_episode(_episode(steps=100, agents=3))

        stored = buffer._as_list()[0]

        assert all(len(traj) == 5 for traj in stored.agent_trajs.values())

    def test_without_a_store_cap_full_episodes_are_kept(self):
        buffer = PreferenceBuffer(max_episodes=4)
        buffer.add_episode(_episode(steps=100, agents=3))

        assert all(len(t) == 100 for t in buffer._as_list()[0].agent_trajs.values())

    def test_nbytes_tracks_stored_frames(self):
        buffer = PreferenceBuffer(max_episodes=4)
        buffer.add_episode(_episode(steps=10, agents=2))

        # 2 agents x 10 steps x 4*4*3 uint8
        assert buffer.nbytes() == 2 * 10 * 48

    def test_store_cap_reduces_residency(self):
        capped = PreferenceBuffer(max_episodes=4, store_max_steps_per_agent=10)
        uncapped = PreferenceBuffer(max_episodes=4)
        for buffer in (capped, uncapped):
            buffer.add_episode(_episode(steps=100, agents=2))

        assert capped.nbytes() * 10 == uncapped.nbytes()
