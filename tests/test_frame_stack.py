"""Frame stacking mirrors DanfoaTestSOT, which applies it as an env wrapper
(ss.frame_stack_v1 in src/experiment_runner/runners.py:120) so the policy and
the reward predictor both see the stack."""

import gymnasium
import numpy as np
import pytest

from commons_game_marp.env.frame_stack import FrameStackEnv


class _StubEnv:
    """Emits a distinct constant frame per step so stacking order is checkable."""

    def __init__(self, agent_ids=("agent-0", "agent-1"), shape=(3, 3, 3)):
        self.agent_ids = list(agent_ids)
        self.shape = shape
        self.counter = 0
        self.agents = {agent_id: object() for agent_id in self.agent_ids}
        self.action_space = gymnasium.spaces.Discrete(8)
        self.rendered = []

    @property
    def observation_space(self):
        return {
            "curr_obs": gymnasium.spaces.Box(
                low=0, high=255, shape=self.shape, dtype=np.uint8
            )
        }

    def _frame(self):
        return np.full(self.shape, self.counter, dtype=np.uint8)

    def reset(self, seed=None):
        self.counter = 1
        obs = {a: {"curr_obs": self._frame()} for a in self.agent_ids}
        return obs, {a: {} for a in self.agent_ids}

    def step(self, actions):
        self.counter += 1
        obs = {a: {"curr_obs": self._frame()} for a in self.agent_ids}
        rewards = {a: 0.0 for a in self.agent_ids}
        dones = {a: False for a in self.agent_ids}
        return obs, rewards, dones, {a: {} for a in self.agent_ids}

    def render(self, path, mod="human"):
        self.rendered.append(path)

    def get_social_metrics(self):
        return {"efficiency": 1.0}


def test_observation_space_widens_along_the_channel_axis():
    wrapped = FrameStackEnv(_StubEnv(), num_frames=2)
    assert wrapped.observation_space["curr_obs"].shape == (3, 3, 6)
    assert wrapped.observation_space["curr_obs"].dtype == np.uint8


def test_reset_fills_the_stack_by_repeating_the_first_frame():
    """The first step must already have a full stack, not a zero-padded one:
    a zero half-frame is a real observation the policy would have to learn to
    ignore."""
    wrapped = FrameStackEnv(_StubEnv(), num_frames=2)
    obs, _ = wrapped.reset()

    frame = obs["agent-0"]["curr_obs"]
    assert frame.shape == (3, 3, 6)
    assert np.all(frame == 1)


def test_step_appends_the_newest_frame_last():
    wrapped = FrameStackEnv(_StubEnv(), num_frames=2)
    wrapped.reset()
    obs, _, _, _ = wrapped.step({"agent-0": 0, "agent-1": 0})

    frame = obs["agent-0"]["curr_obs"]
    assert np.all(frame[:, :, 0:3] == 1)  # oldest
    assert np.all(frame[:, :, 3:6] == 2)  # newest


def test_oldest_frame_is_evicted_once_the_stack_is_full():
    wrapped = FrameStackEnv(_StubEnv(), num_frames=2)
    wrapped.reset()
    wrapped.step({"agent-0": 0, "agent-1": 0})
    obs, _, _, _ = wrapped.step({"agent-0": 0, "agent-1": 0})

    frame = obs["agent-0"]["curr_obs"]
    assert np.all(frame[:, :, 0:3] == 2)
    assert np.all(frame[:, :, 3:6] == 3)


def test_each_agent_keeps_its_own_stack():
    wrapped = FrameStackEnv(_StubEnv(), num_frames=3)
    wrapped.reset()
    obs, _, _, _ = wrapped.step({"agent-0": 0, "agent-1": 0})

    assert obs["agent-0"]["curr_obs"].shape == (3, 3, 9)
    assert obs["agent-1"]["curr_obs"].shape == (3, 3, 9)


def test_reset_clears_state_from_the_previous_episode():
    """reset() rebuilds the env's agents, so a stack carried across the
    boundary would splice two episodes into one observation."""
    wrapped = FrameStackEnv(_StubEnv(), num_frames=2)
    wrapped.reset()
    wrapped.step({"agent-0": 0, "agent-1": 0})
    obs, _ = wrapped.reset()

    assert np.all(obs["agent-0"]["curr_obs"] == 1)


def test_frames_stay_uint8():
    """PreferenceBuffer sizing assumes uint8; a float32 stack would quadruple
    resident memory."""
    wrapped = FrameStackEnv(_StubEnv(), num_frames=2)
    obs, _ = wrapped.reset()
    assert obs["agent-0"]["curr_obs"].dtype == np.uint8


def test_unknown_attributes_delegate_to_the_wrapped_env():
    """Trainer reaches through for env.agents, compute_social_metrics(), and
    the video recorder's env.render()."""
    inner = _StubEnv()
    wrapped = FrameStackEnv(inner, num_frames=2)

    assert wrapped.agents is inner.agents
    assert wrapped.get_social_metrics() == {"efficiency": 1.0}
    assert wrapped.action_space is inner.action_space
    wrapped.render("/tmp/x.png", mod="human")
    assert inner.rendered == ["/tmp/x.png"]


def test_num_frames_below_two_is_rejected():
    """The trainer must not wrap at all for num_frames == 1; a wrapper that
    silently accepted it would make the default path non-identical."""
    with pytest.raises(ValueError, match="num_frames"):
        FrameStackEnv(_StubEnv(), num_frames=1)


class _MutatingBufferEnv(_StubEnv):
    """Returns the same array object every step, mutated in place.

    A real env is free to do this as an allocation optimisation. A wrapper
    that stored references rather than copies would show N copies of the
    newest frame and nobody would notice, because the shapes and dtypes
    would all still be right.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._buffer = np.zeros(self.shape, dtype=np.uint8)

    def _frame(self):
        self._buffer[:] = self.counter
        return self._buffer


def test_stack_owns_its_frames_when_the_env_reuses_its_buffer():
    wrapped = FrameStackEnv(_MutatingBufferEnv(), num_frames=2)
    wrapped.reset()
    obs, _, _, _ = wrapped.step({"agent-0": 0, "agent-1": 0})

    frame = obs["agent-0"]["curr_obs"]
    assert np.all(frame[:, :, 0:3] == 1), "oldest frame was overwritten by aliasing"
    assert np.all(frame[:, :, 3:6] == 2)
