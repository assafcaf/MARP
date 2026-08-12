import json
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class EnvConfig:
    map_type: str = "small"
    num_agents: int = 1
    agent_view_range: int = 5
    ep_length: int = 600
    render: bool = False
    spawn_speed: str = "slow"
    metric: str = "Efficiency"
    penalty: bool = False

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "EnvConfig":
        return EnvConfig(**data)


@dataclass
class DQNConfig:
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
    normalize_obs: bool = True
    device: str = "auto"

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "DQNConfig":
        return DQNConfig(**data)


@dataclass
class IPPOConfig:
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.1
    ent_coef_end: float = 0.01
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

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "IPPOConfig":
        return IPPOConfig(**data)


@dataclass
class MAPPOConfig:
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    n_steps: int = 1024
    batch_size: int = 256
    update_epochs: int = 4
    hidden_size: int = 256
    max_grad_norm: float = 0.5
    normalize_obs: bool = False
    flatten_obs: bool = False
    device: str = "auto"

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "MAPPOConfig":
        return MAPPOConfig(**data)


@dataclass
class AlgorithmConfig:
    name: str = "dqn"
    dqn: DQNConfig = field(default_factory=DQNConfig)
    ippo: IPPOConfig = field(default_factory=IPPOConfig)
    mappo: MAPPOConfig = field(default_factory=MAPPOConfig)

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "AlgorithmConfig":
        dqn = DQNConfig.from_dict(data.get("dqn", {}))
        ippo = IPPOConfig.from_dict(data.get("ippo", {}))
        mappo = MAPPOConfig.from_dict(data.get("mappo", {}))
        return AlgorithmConfig(
            name=data.get("name", "dqn"),
            dqn=dqn,
            ippo=ippo,
            mappo=mappo,
        )


@dataclass
class LoggingConfig:
    log_dir: str = "logs"
    run_name: Optional[str] = None
    log_interval: int = 1
    video_enabled: bool = True
    video_every_n_episodes: int = 100
    video_max_steps: int = 600
    video_fps: int = 10
    video_keep_frames: bool = False
    log_agent_episode_details: bool = True

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "LoggingConfig":
        return LoggingConfig(**data)


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
    # Performance optimization options
    use_amp: bool = True  # Use mixed precision (FP16) for faster training
    chunk_size: int = 512  # Max steps per forward pass chunk (memory control)
    max_steps_per_sequence: Optional[int] = 256  # Temporal subsampling limit (None = no limit)

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "RewardModelConfig":
        return RewardModelConfig(**data)


@dataclass
class TrainerConfig:
    episodes: int = 100
    seed: Optional[int] = 0
    env: EnvConfig = field(default_factory=EnvConfig)
    algorithm: AlgorithmConfig = field(default_factory=AlgorithmConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    reward_model: RewardModelConfig = field(default_factory=RewardModelConfig)

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "TrainerConfig":
        return TrainerConfig(
            episodes=data.get("episodes", 100),
            seed=data.get("seed", 0) if "seed" in data else None,
            env=EnvConfig.from_dict(data.get("env", {})),
            algorithm=AlgorithmConfig.from_dict(data.get("algorithm", {})),
            logging=LoggingConfig.from_dict(data.get("logging", {})),
            reward_model=RewardModelConfig.from_dict(data.get("reward_model", {})),
        )


def load_config(path: str) -> TrainerConfig:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return TrainerConfig.from_dict(data)


def save_config(path: str, config: TrainerConfig) -> None:
    payload = {
        "episodes": config.episodes,
        "seed": config.seed if config.seed is not None else None,
        "env": config.env.__dict__,
        "algorithm": {
            "name": config.algorithm.name,
            "dqn": config.algorithm.dqn.__dict__,
            "ippo": config.algorithm.ippo.__dict__,
            "mappo": config.algorithm.mappo.__dict__,
        },
        "logging": config.logging.__dict__,
        "reward_model": config.reward_model.__dict__,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
