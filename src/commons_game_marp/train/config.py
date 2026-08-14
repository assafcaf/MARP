from dataclasses import dataclass, field
from typing import Any, Optional

from omegaconf import MISSING


@dataclass
class EnvConfig:
    map_type: str = "small"
    num_agents: int = 1
    agent_view_range: int = 7
    # Observations stacked along the channel axis. 1 leaves the env unwrapped;
    # above 1 the trainer applies FrameStackEnv and every consumer -- policies,
    # reward model, preference buffer -- widens with it. Note that the
    # preference buffer's resident size scales linearly with this.
    num_frames: int = 1
    # Independent environment copies stepped in lockstep. Above 1 the trainer
    # runs `episodes // num_envs` iterations, each completing num_envs
    # episodes, so `episodes` stays a total budget rather than an iteration
    # count. Stepping is serial -- this buys decorrelated samples per update,
    # not wall-clock speed.
    num_envs: int = 1
    ep_length: int = 600
    render: bool = False
    spawn_speed: str = "slow"
    metric: str = "Efficiency"
    penalty: bool = False
    # Put the full-map RGB render on every agent's info dict as `state`.
    # Off by default: it is a whole-map render per agent per step -- 1938 bytes
    # against the observation's 675 on the medium map -- that nothing in this
    # repo reads. Enabling it measured 2.2x slower (3064 -> 1398 agent-steps/s)
    # and, with `num_workers` above 0, adds ~15 MB per iteration of pipe
    # traffic. Turn it on only for an external consumer that needs global state.
    include_state_in_info: bool = False
    # Worker processes the environment copies are spread across. 0 steps them
    # in-process, which is what `num_envs` used to do exclusively -- and why it
    # bought sample decorrelation but no wall-clock speed. Above 0 the copies
    # run in parallel; results are bit-identical either way.
    #
    # More workers than `num_envs` is clamped to `num_envs`. Each worker is a
    # process, so keep the total under the core count.
    num_workers: int = 0


@dataclass
class DQNConfig:
    name: str = "dqn"
    learning_rate: float = 1e-3
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.1
    epsilon_decay: float = 0.995
    batch_size: int = 32
    replay_buffer_size: int = 5000
    train_after: int = 100
    train_every: int = 1
    target_update_freq: int = 200
    max_grad_norm: float = 10.0
    normalize_obs: bool = True
    device: str = "auto"


@dataclass
class IPPOConfig:
    name: str = "ippo"
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef_mode: str = "adaptive"  # "fixed" | "anneal" | "adaptive"
    ent_coef: float = 0.1  # initial value (adaptive) / schedule start (anneal)
    ent_coef_end: float = 0.01  # anneal only
    target_entropy_frac: float = 0.6  # adaptive: target = frac * ln(num_actions)
    ent_coef_lr: float = 3e-3
    ent_coef_min: float = 1e-3
    ent_coef_max: float = 0.5
    vf_coef: float = 0.5
    vf_clip: Optional[float] = 10.0
    # Timesteps collected per environment before an update. Null aligns the
    # update with the episode boundary by resolving to `env.ep_length`, so
    # every episode ends in exactly one update over `ep_length * num_envs`
    # transitions. Setting a value that does not divide `ep_length` leaves a
    # short final batch each episode -- 512 against ep_length 600 updates on
    # 512 transitions and then on 88, below the 128 minibatch size.
    n_steps: Optional[int] = None
    batch_size: int = 128
    update_epochs: int = 2
    hidden_size: int = 256
    max_grad_norm: float = 0.5
    normalize_obs: bool = True
    flatten_obs: bool = False
    device: str = "auto"


@dataclass
class MAPPOConfig:
    name: str = "mappo"
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef_mode: str = "adaptive"  # "fixed" | "anneal" | "adaptive"
    ent_coef: float = 0.1  # initial value (adaptive) / schedule start (anneal)
    ent_coef_end: float = 0.01  # anneal only
    target_entropy_frac: float = 0.6  # adaptive: target = frac * ln(num_actions)
    ent_coef_lr: float = 3e-3
    ent_coef_min: float = 1e-3
    ent_coef_max: float = 0.5
    vf_coef: float = 0.5
    # Null aligns the update with the episode boundary; see IPPOConfig.n_steps.
    n_steps: Optional[int] = None
    batch_size: int = 256
    update_epochs: int = 4
    hidden_size: int = 256
    max_grad_norm: float = 0.5
    normalize_obs: bool = False
    flatten_obs: bool = False
    device: str = "auto"


@dataclass
class RandomConfig:
    """The random policy has no hyperparameters, but `normalize_obs` must be
    declared: `_format_reward_obs` reads it off the selected algorithm node with
    no fallback. True preserves the previous behavior, where `random` borrowed
    the dqn section's scale."""

    name: str = "random"
    normalize_obs: bool = True
    device: str = "auto"


@dataclass
class LoggingConfig:
    log_dir: str = "logs"
    # Groups the runs of one configuration: `log_dir/<run_name>/<timestamp>-seed=N`.
    # Left null, the group name is derived from the configuration itself.
    run_name: Optional[str] = None
    # The run's own directory. Set by the Hydra entry point to
    # `hydra.runtime.output_dir`, so Hydra's `.hydra/` snapshot and job log sit
    # beside the metrics, videos and checkpoints. Null outside Hydra: a
    # programmatic `Trainer(...)` derives the same layout from `log_dir`.
    run_dir: Optional[str] = None
    log_interval: int = 1
    # Console output: "auto" shows a progress bar on a terminal and periodic
    # status lines when the stream is redirected, "bar"/"plain" force one of the
    # two, "quiet" silences everything.
    console: str = "auto"
    # Episodes between status lines when no progress bar is shown.
    status_every: int = 10
    video_enabled: bool = True
    video_every_n_episodes: int = 100
    video_max_steps: int = 600
    video_fps: int = 10
    video_keep_frames: bool = False
    log_agent_episode_details: bool = True
    # Action distribution, harvest behaviour and the step-level reward-model
    # breakdowns (`action/*`, `harvest/*`, `rm_*`). The cost is one stencil
    # gather per agent per environment per step; turning this off restores the
    # previous tag set.
    detailed_metrics: bool = True
    # Radius for `nearby_apples` and the `rm_by_nearby_apples` buckets. 2 is
    # APPLE_RADIUS -- the neighbourhood that actually drives regrowth.
    nearby_apple_radius: int = 2
    # Episodes between distribution histograms (`rm_pred/hist`,
    # `reward/agent_hist`). 0 disables them; they are far larger on disk than a
    # scalar, so this is off by default.
    histogram_every_n_episodes: int = 0


@dataclass
class RewardModelConfig:
    enabled: bool = False
    mode: str = "narrow_view"
    phi: str = "efficiency_x_peace"
    lr: float = 1e-4
    batch_pairs: int = 64
    train_steps_per_update: int = 50
    # Env steps between reward-model updates. Null trains once at every episode
    # boundary, which is the same cadence the policies update on when
    # `n_steps` is null. An integer keeps the step-denominated period, which
    # the trainer catches up on a period at a time so the rate does not depend
    # on num_envs.
    update_every_env_steps: Optional[int] = None
    warmup_episodes: int = 50
    max_episodes_in_buffer: int = 5000
    device: str = "auto"
    save_every_episodes: int = 200
    # Optimisation
    weight_decay: float = 1e-4  # Matches the reference MARP predictor's Adam
    max_grad_norm: Optional[float] = 1.0  # None or 0 disables clipping
    delta_temperature: float = 1.0  # softmax(delta / (std + tau)) pair weighting
    tie_tolerance: float = 0.0  # |phi_i - phi_j| <= this counts as a tie (mu=0.5)
    # Performance optimization options
    use_amp: bool = True  # Use mixed precision (FP16) for faster training
    chunk_size: int = 512  # Max steps per forward pass chunk (memory control)
    grad_checkpoint: bool = False  # Recompute chunk activations in backward
    max_steps_per_sequence: Optional[int] = 256  # Temporal subsampling limit (None = no limit)
    # Cap steps kept per agent at insertion time (None = keep the full episode).
    # The buffer's resident size is the binding constraint at the default
    # buffer length; see PreferenceBuffer's docstring for the arithmetic.
    store_max_steps_per_agent: Optional[int] = None


@dataclass
class TrainerConfig:
    episodes: int = 100
    seed: Optional[int] = 0
    env: EnvConfig = field(default_factory=EnvConfig)
    # `Any` rather than a union type: Hydra selects one of the four algorithm
    # nodes into this slot via the `algorithm` config group, and each node has a
    # different shape. The selected node is validated against its own schema.
    algorithm: Any = MISSING
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    reward_model: RewardModelConfig = field(default_factory=RewardModelConfig)


def resolve_iterations(episodes: int, num_envs: int) -> int:
    """Iterations needed to spend an `episodes` budget `num_envs` at a time.

    Refuses a non-divisible pair rather than truncating: a run that quietly
    completes 996 of a requested 1000 episodes is not comparable with one that
    completed 1000, and nothing downstream would show the difference.
    """
    if num_envs < 1:
        raise ValueError(f"env.num_envs must be >= 1, got {num_envs}")
    remainder = episodes % num_envs
    if remainder:
        raise ValueError(
            f"episodes ({episodes}) must be divisible by env.num_envs ({num_envs}); "
            f"otherwise the episode budget is silently truncated. "
            f"Use {episodes - remainder} or {episodes + num_envs - remainder} episodes."
        )
    return episodes // num_envs


def register_configs() -> None:
    """Register the schema for every config group with Hydra.

    Registering the dataclasses makes the YAML type-checked at composition
    time: an unknown or mistyped key fails at startup instead of silently
    falling back to a default. Also registers the resolvers `hydra.run.dir`
    refers to, which must exist before composition. Safe to call more than once.
    """
    from hydra.core.config_store import ConfigStore

    from .run_paths import register_resolvers

    register_resolvers()

    cs = ConfigStore.instance()
    cs.store(name="base_config", node=TrainerConfig)
    cs.store(group="env", name="base_env", node=EnvConfig)
    cs.store(group="algorithm", name="base_dqn", node=DQNConfig)
    cs.store(group="algorithm", name="base_ippo", node=IPPOConfig)
    cs.store(group="algorithm", name="base_mappo", node=MAPPOConfig)
    cs.store(group="algorithm", name="base_random", node=RandomConfig)
    cs.store(group="logging", name="base_logging", node=LoggingConfig)
    cs.store(group="reward_model", name="base_reward_model", node=RewardModelConfig)
