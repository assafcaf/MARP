from dataclasses import dataclass, field
from typing import Any, Optional

from omegaconf import MISSING


@dataclass
class EnvConfig:
    map_type: str = "small"
    num_agents: int = 1
    agent_view_range: int = 7
    ep_length: int = 600
    render: bool = False
    spawn_speed: str = "slow"
    metric: str = "Efficiency"
    penalty: bool = False


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
    ent_coef_lr: float = 3e-4
    ent_coef_min: float = 1e-3
    ent_coef_max: float = 0.5
    vf_coef: float = 0.5
    vf_clip: Optional[float] = 10.0
    n_steps: int = 512
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
    ent_coef_lr: float = 3e-4
    ent_coef_min: float = 1e-3
    ent_coef_max: float = 0.5
    vf_coef: float = 0.5
    n_steps: int = 1024
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


@dataclass
class RewardModelConfig:
    enabled: bool = False
    mode: str = "narrow_view"
    phi: str = "efficiency_x_peace"
    lr: float = 1e-4
    batch_pairs: int = 64
    train_steps_per_update: int = 50
    update_every_env_steps: int = 1000
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
