"""Observation frame stacking, applied as an environment wrapper.

Wrapping the env rather than each algorithm's `_format_obs` means every
consumer picks the stack up from `observation_space` on its own: the four
algorithms, `RewardModel` (built from `env.observation_space["curr_obs"].shape`
in `Trainer.train`), and the preference buffer that stores what
`_format_reward_obs` returns. That mirrors the reference implementation, where
`ss.frame_stack_v1` sits on the env and the reward predictor reads the stack
depth straight off the observation space.

Note the memory consequence: `PreferenceBuffer` holds raw frames, so its
resident size scales linearly with `num_frames`. See `Trainer._warn_if_buffer_large`.
"""

from collections import deque
from typing import Any, Deque, Dict, Optional, Tuple

import gymnasium
import numpy as np


class FrameStackEnv:
    """Stacks the last `num_frames` observations along the channel axis.

    Only constructed when `num_frames > 1`; at 1 the trainer uses the bare env
    so the default code path is unchanged rather than merely equivalent.
    """

    def __init__(self, env: Any, num_frames: int) -> None:
        if num_frames < 2:
            raise ValueError(
                f"num_frames must be >= 2 to stack, got {num_frames}. "
                "Use the unwrapped env for num_frames == 1."
            )
        self.env = env
        self.num_frames = int(num_frames)
        self._stacks: Dict[str, Deque[np.ndarray]] = {}

    @property
    def observation_space(self) -> Dict[str, gymnasium.spaces.Box]:
        inner = self.env.observation_space["curr_obs"]
        height, width, channels = inner.shape
        return {
            "curr_obs": gymnasium.spaces.Box(
                low=0,
                high=255,
                shape=(height, width, channels * self.num_frames),
                dtype=np.uint8,
            )
        }

    def _stack_for(self, agent_id: str) -> np.ndarray:
        # Oldest first, newest last -- the ordering the tests pin and the one a
        # reader of a rendered stack expects.
        return np.concatenate(list(self._stacks[agent_id]), axis=-1)

    def _seed_stacks(self, observations: Dict[str, Any]) -> Dict[str, Any]:
        """Start each agent's stack by repeating its first frame.

        Zero padding would hand the policy a half-blank observation on the
        first steps of every episode -- a real input it would have to learn to
        ignore.
        """
        self._stacks = {}
        stacked = {}
        for agent_id, agent_obs in observations.items():
            frame = np.asarray(agent_obs["curr_obs"], dtype=np.uint8)
            self._stacks[agent_id] = deque(
                [frame] * self.num_frames, maxlen=self.num_frames
            )
            stacked[agent_id] = {**agent_obs, "curr_obs": self._stack_for(agent_id)}
        return stacked

    def _append(self, observations: Dict[str, Any]) -> Dict[str, Any]:
        stacked = {}
        for agent_id, agent_obs in observations.items():
            frame = np.asarray(agent_obs["curr_obs"], dtype=np.uint8)
            if agent_id not in self._stacks:
                self._stacks[agent_id] = deque(
                    [frame] * self.num_frames, maxlen=self.num_frames
                )
            else:
                self._stacks[agent_id].append(frame)
            stacked[agent_id] = {**agent_obs, "curr_obs": self._stack_for(agent_id)}
        return stacked

    def reset(self, seed: Optional[int] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        observations, infos = self.env.reset(seed=seed)
        return self._seed_stacks(observations), infos

    def step(self, actions: Dict[str, int]) -> Tuple[Dict[str, Any], Any, Any, Any]:
        observations, rewards, dones, infos = self.env.step(actions)
        return self._append(observations), rewards, dones, infos

    def __getattr__(self, name: str) -> Any:
        # Only called for attributes this wrapper does not define, so `env`,
        # `num_frames` and the methods above never reach here. Guarded against
        # recursion during unpickling, when `env` may not be set yet.
        if name == "env":
            raise AttributeError(name)
        return getattr(self.env, name)
