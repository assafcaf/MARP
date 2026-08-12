from commons_game_marp.train import Trainer, load_config


def run_training(config_path: str) -> None:
    config = load_config(config_path)
    trainer = Trainer(config)
    trainer.train()


if __name__ == "__main__":
    # Template runs (uncomment one). Toggle reward modeling via the
    # reward_model section in the config.
    # run_training("configs/train_dqn.json")
    # run_training("configs/train_ippo.json")
    run_training("configs/train_mappo.json")
