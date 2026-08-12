from .config import AlgorithmConfig


def build_algorithm(config: AlgorithmConfig):
    name = config.name
    if name == "dqn":
        from .algorithms.dqn import DQNAlgorithm

        return DQNAlgorithm(config.dqn)
    if name == "ippo":
        from .algorithms.ippo import IPPOAlgorithm

        return IPPOAlgorithm(config.ippo)
    if name == "mappo":
        from .algorithms.mappo import MAPPOAlgorithm

        return MAPPOAlgorithm(config.mappo)
    if name == "random":
        from .algorithms.random_policy import RandomAlgorithm

        return RandomAlgorithm(config)
    raise ValueError(f"Unknown algorithm '{name}'. Available: ['dqn', 'ippo', 'mappo', 'random']")
