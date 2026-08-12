import argparse
import json
from pathlib import Path
from typing import List

from commons_game_marp.train import Trainer, load_config

DEFAULT_CONFIGS = {
    "dqn": "configs/train_dqn.json",
    "ippo": "configs/train_ippo.json",
    "mappo": "configs/train_mappo.json",
    "random": "configs/train_dqn.json",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MARP training with CLI overrides.")
    parser.add_argument(
        "--algo",
        default=["dqn"],
        choices=sorted(DEFAULT_CONFIGS.keys()),
        nargs="+",
        help="Algorithm(s) to run. Can specify multiple to run sequentially (e.g., --algo dqn ippo mappo).",
    )
    parser.add_argument("--episodes", type=int, help="Number of training episodes.")
    parser.add_argument(
        "--seed",
        nargs="+",
        help="Random seed(s) (integer or 'random'). Can specify multiple to run sequentially (e.g., --seed 0 1 random 3). Use 'random' to generate a random seed for that game.",
    )
    parser.add_argument(
        "--random-seed",
        action="store_true",
        help="Use a random seed for all games (overrides --seed). Can also use 'random' in --seed list or specify random_seed in sequence files.",
    )
    parser.add_argument("--map", dest="map_type", nargs="+", help="Map type(s) (e.g., small). Can specify multiple to run sequentially.")
    parser.add_argument("--agents", type=int, nargs="+", help="Number of agents. Can specify multiple to run sequentially.")
    rm_group = parser.add_mutually_exclusive_group()
    rm_group.add_argument(
        "--reward-model",
        action="store_true",
        dest="reward_model",
        help="Enable reward modeling.",
    )
    rm_group.add_argument(
        "--no-reward-model",
        action="store_true",
        dest="no_reward_model",
        help="Disable reward modeling.",
    )
    parser.add_argument("--mode", help="Reward model mode (e.g., narrow_view).")
    parser.add_argument("--phi", help="Reward model objective (e.g., efficiency_x_peace).")
    parser.add_argument(
        "--penalty",
        action="store_true",
        help="Enable penalty for FIRE action (agents get -1 reward when using FIRE).",
    )
    parser.add_argument(
        "--sequence-file",
        type=str,
        help="Path to JSON file containing a list of game configurations to run sequentially. Each entry should be a dict with CLI argument keys (e.g., {'algo': 'dqn', 'episodes': 100}).",
    )
    return parser.parse_args()


def apply_args_to_config(config, args_dict: dict) -> None:
    """Apply CLI arguments to a config object."""
    if "algo" in args_dict and args_dict["algo"] is not None:
        algo = args_dict["algo"]
        if isinstance(algo, list):
            algo = algo[0]  # Use first algo for config loading
        config.algorithm.name = algo
    
    if "episodes" in args_dict and args_dict["episodes"] is not None:
        config.episodes = int(args_dict["episodes"])
    
    # Handle seed: None means random seed, integer means specific seed
    if "seed" in args_dict:
        if args_dict["seed"] is None:
            config.seed = None  # Will be randomly generated in Trainer
        else:
            config.seed = int(args_dict["seed"])
    elif "random_seed" in args_dict and args_dict["random_seed"]:
        config.seed = None  # Will be randomly generated in Trainer
    if "map_type" in args_dict and args_dict["map_type"] is not None:
        map_val = args_dict["map_type"]
        config.env.map_type = map_val if not isinstance(map_val, list) else map_val[0]
    if "agents" in args_dict and args_dict["agents"] is not None:
        agents_val = args_dict["agents"]
        config.env.num_agents = int(agents_val if not isinstance(agents_val, list) else agents_val[0])

    reward_model_enabled = None
    if "reward_model" in args_dict and args_dict["reward_model"]:
        reward_model_enabled = True
    elif "no_reward_model" in args_dict and args_dict["no_reward_model"]:
        reward_model_enabled = False
    if reward_model_enabled is not None:
        config.reward_model.enabled = reward_model_enabled

    if "mode" in args_dict and args_dict["mode"] is not None:
        config.reward_model.mode = args_dict["mode"]
    if "phi" in args_dict and args_dict["phi"] is not None:
        config.reward_model.phi = args_dict["phi"]
    
    if "penalty" in args_dict and args_dict["penalty"] is not None:
        config.env.penalty = bool(args_dict["penalty"])


def run_single_game(args_dict: dict) -> None:
    """Run a single game with the given arguments."""
    algo = args_dict.get("algo", "dqn")
    if isinstance(algo, list):
        algo = algo[0]
    
    config_path = DEFAULT_CONFIGS[algo]
    if not Path(config_path).exists():
        raise FileNotFoundError(f"Missing config file: {config_path}")
    config = load_config(config_path)
    
    apply_args_to_config(config, args_dict)
    
    print(f"\n{'='*60}")
    print(f"Running game: algo={config.algorithm.name}, map={config.env.map_type}, agents={config.env.num_agents}, episodes={config.episodes}")
    print(f"{'='*60}\n")
    
    trainer = Trainer(config)
    trainer.train()


def generate_combinations(args: argparse.Namespace) -> List[dict]:
    """Generate all combinations of games to run from CLI arguments."""
    combinations = []
    
    # nargs="+" always returns a list, but handle edge cases
    algos = args.algo if isinstance(args.algo, list) else [args.algo]
    maps = args.map_type if args.map_type is not None else [None]
    if maps and not isinstance(maps, list):
        maps = [maps]
    agents_list = args.agents if args.agents is not None else [None]
    if agents_list and not isinstance(agents_list, list):
        agents_list = [agents_list]
    # Parse seeds: convert "random" strings to None, integers to int
    if args.seed is not None:
        seeds = []
        for s in args.seed:
            if isinstance(s, str) and s.lower() == "random":
                seeds.append(None)
            else:
                try:
                    seeds.append(int(s))
                except (ValueError, TypeError):
                    seeds.append(None)
    else:
        seeds = [None]
    
    if not isinstance(seeds, list):
        seeds = [seeds]
    
    # If random_seed is specified, override seeds
    if args.random_seed:
        seeds = [None]  # None means random seed will be generated
    
    # Generate all combinations
    for algo in algos:
        for map_type in maps:
            for agents in agents_list:
                for seed in seeds:
                    combo = {
                        "algo": algo,
                        "map_type": map_type,
                        "agents": agents,
                        "episodes": args.episodes,
                        "seed": seed,
                        "random_seed": args.random_seed,
                        "reward_model": args.reward_model,
                        "no_reward_model": args.no_reward_model,
                        "mode": args.mode,
                        "phi": args.phi,
                        "penalty": args.penalty,
                    }
                    combinations.append(combo)
    
    return combinations


def load_sequence_file(file_path: str) -> List[dict]:
    """Load a sequence of game configurations from a JSON file."""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    if not isinstance(data, list):
        raise ValueError("Sequence file must contain a JSON array of configurations.")
    
    # Normalize seed values: convert "random" strings or null to None
    for config in data:
        if "seed" in config:
            if config["seed"] is None or (isinstance(config["seed"], str) and config["seed"].lower() == "random"):
                config["seed"] = None
            elif isinstance(config["seed"], str):
                try:
                    config["seed"] = int(config["seed"])
                except ValueError:
                    config["seed"] = None
        # Also handle random_seed flag
        if config.get("random_seed", False):
            config["seed"] = None
    
    return data


def main() -> None:
    args = parse_args()
    
    # Check if sequence file is provided
    if args.sequence_file:
        if not Path(args.sequence_file).exists():
            raise FileNotFoundError(f"Sequence file not found: {args.sequence_file}")
        sequences = load_sequence_file(args.sequence_file)
        print(f"Loaded {len(sequences)} game configurations from sequence file.")
        for i, seq_args in enumerate(sequences, 1):
            print(f"\n[{i}/{len(sequences)}] Running game from sequence file...")
            run_single_game(seq_args)
    else:
        # Generate combinations from CLI arguments
        combinations = generate_combinations(args)
        
        if len(combinations) > 1:
            print(f"Running {len(combinations)} games sequentially...")
        
        for i, combo in enumerate(combinations, 1):
            if len(combinations) > 1:
                print(f"\n[{i}/{len(combinations)}] Running game...")
            run_single_game(combo)


if __name__ == "__main__":
    main()
