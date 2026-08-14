"""A picklable description of one environment copy.

Worker processes cannot receive a factory closure. `Trainer._make_single_env` is
a bound method, and `multiprocessing`'s `spawn` start method pickles whatever it
is given -- so the workers get this instead and build their own environments
from it.

`spawn`, not `fork`: by the time the vector env is constructed the parent has
usually initialised CUDA for the policy networks, and forking a process with a
live CUDA context is undefined behaviour. `spawn` costs a second of startup once
and avoids the whole class of problem.
"""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class EnvSpec:
    """Everything needed to build one `HarvestCommonsEnv`, plus its wrappers."""

    map_type: str = "small"
    num_agents: int = 1
    agent_view_range: int = 7
    ep_length: int = 600
    render: bool = False
    spawn_speed: str = "slow"
    metric: str = "Efficiency"
    penalty: bool = False
    num_frames: int = 1
    include_state_in_info: bool = False
    step_metrics: bool = False
    nearby_apple_radius: int = 2

    @classmethod
    def from_config(cls, env_cfg: Any, step_metrics: bool, nearby_apple_radius: int) -> "EnvSpec":
        return cls(
            map_type=env_cfg.map_type,
            num_agents=int(env_cfg.num_agents),
            agent_view_range=int(env_cfg.agent_view_range),
            ep_length=int(env_cfg.ep_length),
            render=bool(env_cfg.render),
            spawn_speed=env_cfg.spawn_speed,
            metric=env_cfg.metric,
            penalty=bool(env_cfg.penalty),
            num_frames=int(env_cfg.num_frames),
            include_state_in_info=bool(env_cfg.include_state_in_info),
            step_metrics=bool(step_metrics),
            nearby_apple_radius=int(nearby_apple_radius),
        )

    def build(self):
        """Construct the environment this spec describes.

        Imported inside the method so a worker pays the import cost once, in the
        worker, rather than dragging the env package into the pickle.
        """
        from .commons_env import MAP, HarvestCommonsEnv
        from .frame_stack import FrameStackEnv

        env = HarvestCommonsEnv(
            ascii_map=MAP[self.map_type],
            num_agents=self.num_agents,
            render=self.render,
            agent_view_range=self.agent_view_range,
            ep_length=self.ep_length,
            spawn_speed=self.spawn_speed,
            metric=self.metric,
            penalty=self.penalty,
            include_state_in_info=self.include_state_in_info,
            step_metrics=self.step_metrics,
            nearby_apple_radius=self.nearby_apple_radius,
        )
        if self.num_frames < 1:
            raise ValueError(f"env.num_frames must be >= 1, got {self.num_frames}")
        if self.num_frames > 1:
            return FrameStackEnv(env, self.num_frames)
        return env
