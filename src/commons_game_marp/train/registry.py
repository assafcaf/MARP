def build_algorithm(config):
    """Build the algorithm named by the selected config node.

    `config` is one of DQNConfig / IPPOConfig / MAPPOConfig / RandomConfig --
    the single node Hydra selected into `TrainerConfig.algorithm`.
    """
    name = config.name
    if name == "dqn":
        from .algorithms.dqn import DQNAlgorithm

        return DQNAlgorithm(config)
    if name == "ippo":
        from .algorithms.ippo import IPPOAlgorithm

        return IPPOAlgorithm(config)
    if name == "mappo":
        from .algorithms.mappo import MAPPOAlgorithm

        return MAPPOAlgorithm(config)
    if name == "random":
        from .algorithms.random_policy import RandomAlgorithm

        return RandomAlgorithm(config)
    raise ValueError(f"Unknown algorithm '{name}'. Available: ['dqn', 'ippo', 'mappo', 'random']")
