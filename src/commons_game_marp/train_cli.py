"""Hydra entry point for training.

Run with `commons-game-train` (installed console script) or
`python main.py` from the repo root. Any config value can be overridden on the
command line; see README for examples.
"""

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from .train.config import TrainerConfig, register_configs
from .train.trainer import Trainer

register_configs()


@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    # to_object resolves interpolations and instantiates the registered
    # dataclasses, so Trainer receives a real TrainerConfig rather than a
    # DictConfig -- no changes needed inside Trainer or the algorithms.
    config: TrainerConfig = OmegaConf.to_object(cfg)
    # Hydra already created this directory and put `.hydra/` and the job log in
    # it. Pointing the Trainer at it is what keeps a run's Hydra output and its
    # artifacts in one place, for `--multirun` jobs as much as single runs.
    config.logging.run_dir = HydraConfig.get().runtime.output_dir
    Trainer(config).train()


if __name__ == "__main__":
    main()
