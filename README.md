# MARP Environment

This repo contains a grid-based multi-agent environment built on top of `gymnasium`.
Agents move on an ASCII map, collect apples for reward, and can optionally use a
`FIRE` action to penalize other agents. The environment tracks social metrics like
efficiency, equality, sustainability, and peace.

Key pieces:
- `src/commons_game_marp/env/commons_env.py`: Harvest commons environment (`HarvestCommonsEnv`).
- `src/commons_game_marp/env/map_env.py`: Core map simulation, movement, and rendering.
- `src/commons_game_marp/env/commons_agent.py`: Agent behavior and action space.
- `src/commons_game_marp/env/maps.py`: ASCII map layouts used for spawning walls/apples/agents.
- `src/commons_game_marp/reward_model/`: MARP-style preference-based reward model and training utilities.
- `src/commons_game_marp/train/metrics.py`: Agent-specific metrics calculation (nearby apples, cluster detection).
- `scripts/plot_phi_comparisons.py`: Phi comparison plots and NV vs IA social metrics galleries.

## Installation

This project is a Python package managed with [uv](https://docs.astral.sh/uv/).

```bash
# Install uv (once, if you don't have it)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create the environment and install the project with all dependencies
uv sync
```

`uv sync` reads `pyproject.toml` and `uv.lock`, creates `.venv/`, and installs
`commons-game-marp` in editable mode — edits to `src/commons_game_marp/` take
effect without reinstalling.

Prefix commands with `uv run` to use the project environment:

```bash
uv run pytest              # run the test suite
uv run python main.py      # run training
```

### PyTorch and CUDA

The pinned build is `torch==2.5.1+cu124`, resolved from PyTorch's CUDA 12.4
index (configured in `pyproject.toml`). To run on CPU or a different CUDA
version, change the `torch` pin and the `[[tool.uv.index]]` URL, then re-run
`uv sync`.

## Training (configurable trainer)

The training entrypoint is `main.py`. Uncomment one of the template lines or use the
inline script below.

Template (edit `main.py`):

```bash
uv run python main.py
```

Inline run (example: `configs/train_dqn.json`):

```bash
uv run python - << 'PY'
from commons_game_marp.train import Trainer, load_config

config = load_config('configs/train_dqn.json')
trainer = Trainer(config)
trainer.train()
print('done')
PY
```

Logs are written to `logs/<run-name>/metrics.jsonl` and `logs/<run-name>/config.json`.
Videos are written to `logs/<run-name>/videos/episode=XXXX.mp4`.
TensorBoard logs are written to `logs/<run-name>/tensorboard/`.
Extended agent episode information is written to `logs/<run-name>/extended_info/agent_X_episodes.csv`.
Run folders include a reward-model suffix, e.g. `...-rm=off` or `...-rm=narrow_view`.

**Detailed agent episode logs:** When `logging.log_agent_episode_details` is enabled (default: `true`), separate CSV files are created for each agent in the `extended_info/` subdirectory (e.g., `extended_info/agent_0_episodes.csv`, `extended_info/agent_1_episodes.csv`). Each row in the CSV represents one step within an episode, with the following columns:
- `episode`: Episode number
- `step`: Step number within the episode
- `action`: Action taken by the agent
- `reward`: Reward received for this step
- `predicted_reward`: Predicted reward for this step (only if reward model enabled)
- `apple_eaten`: Boolean indicating whether an apple was consumed in the current step (True if reward > 0)
- `nearby_apples`: Integer count of apples within 2 steps (Euclidean distance) from the agent's position. This metric only counts apples that are actually nearby, not all apples in the agent's full view range.
- `ate_last_apple_in_cluster`: Boolean indicating whether the agent consumed the last remaining apple in a cluster (a resource that will not reproduce). This is True when an apple is eaten and no other apples remain within the spawn radius (APPLE_RADIUS=2) of the nearest apple spawn point.

View TensorBoard (live during training):

```bash
tensorboard --logdir logs
```

Config tips:
- `logging.video_every_n_episodes` defaults to 100; reduce it to record more frequently.
- `logging.video_max_steps` caps episode length in videos.
- `logging.video_enabled=false` disables video capture for faster training.
- `logging.log_agent_episode_details=true` (default) enables detailed per-agent episode logging to separate files.
- The last episode is always recorded (if video is enabled), regardless of `video_every_n_episodes`.
- Example configs: `configs/train_dqn.json`, `configs/train_ippo.json`, `configs/train_mappo.json`.

## Run the environment script

Use `scripts/run_env.py` to launch training with CLI overrides:

```bash
python scripts/run_env.py --algo mappo --episodes 200 --reward-model --mode narrow_view --phi efficiency_x_peace
```

Windows example:

```bash
python scripts\run_env.py --algo mappo --episodes 200 --agents 5 --seed 0 --reward-model --mode narrow_view --phi efficiency_x_peace
```

Random seed example:

```bash
python scripts/run_env.py --algo dqn --episodes 100 --random-seed
```

Penalty example (with penalty for FIRE action):

```bash
python scripts/run_env.py --algo mappo --episodes 250 --random-seed --no-reward-model --penalty
```

### Running sequences of games

You can run multiple games sequentially in several ways:

**Multiple algorithms:**
```bash
python scripts/run_env.py --algo dqn ippo mappo --episodes 100
```

**Multiple maps:**
```bash
python scripts/run_env.py --algo dqn --map small large --episodes 100
```

**Multiple agent counts:**
```bash
python scripts/run_env.py --algo dqn --agents 3 5 7 --episodes 100
```

**Multiple seeds (including random):**
```bash
python scripts/run_env.py --algo dqn --seed 0 1 random 3 --episodes 100
```
This runs 4 games: seeds 0, 1, random, and 3.

**All random seeds:**
```bash
python scripts/run_env.py --algo dqn ippo --random-seed --episodes 100
```
This runs 2 games, each with a different randomly generated seed.

**All combinations:**
```bash
python scripts/run_env.py --algo dqn ippo --map small large --agents 3 5 --seed 0 1 --episodes 100
```
This will run all combinations: 2 algorithms × 2 maps × 2 agent counts × 2 seeds = 16 games total.

**Using a sequence file:**
Create a JSON file (e.g., `sequence.json`) with a list of game configurations:

```json
[
  {"algo": "dqn", "episodes": 100, "map_type": "small", "agents": 3, "seed": 0},
  {"algo": "ippo", "episodes": 200, "map_type": "large", "agents": 5, "reward_model": true, "seed": 1},
  {"algo": "mappo", "episodes": 150, "map_type": "small", "agents": 7, "seed": null},
  {"algo": "dqn", "episodes": 100, "map_type": "small", "agents": 3, "random_seed": true}
]
```
Note: Use `"seed": null` or `"random_seed": true` in sequence files to use random seeds for that game.

Then run:
```bash
python scripts/run_env.py --sequence-file sequence.json
```

Arguments:
- `--algo {dqn,ippo,mappo,random}` selects the algorithm (default: dqn). Can specify multiple values to run sequentially.
- `--episodes N` sets the number of training episodes.
- `--seed N` sets the random seed(s) (integer or 'random'). Can specify multiple values to run sequentially (e.g., `--seed 0 1 random 3`). Use 'random' to generate a random seed for that specific game.
- `--random-seed` uses a randomly generated seed for all games (overrides any `--seed` values). Each game will get a different random seed.
- `--map NAME` sets `env.map_type`. Can specify multiple values to run sequentially.
- `--agents N` sets `env.num_agents`. Can specify multiple values to run sequentially.
- `--penalty` enables penalty for FIRE action (agents get -1 reward when using FIRE action). When disabled (default), FIRE action has no direct reward penalty.
- `--reward-model` / `--no-reward-model` toggles reward modeling.
- `--mode MODE` sets `reward_model.mode`.
- `--phi PHI` sets `reward_model.phi`.
- `--sequence-file PATH` path to JSON file containing a list of game configurations to run sequentially.

Switch algorithms by changing `algorithm.name` in the config. Supported values:
`dqn`, `random`, `ippo`, `mappo`.

- **DQN**: Deep Q-Network for single or multi-agent (independent learners).
- **IPPO**: Independent PPO - true decentralized training where all agents train simultaneously with their own policies. Each agent has independent actor and critic networks.
- **MAPPO**: Multi-Agent PPO with centralized training and decentralized execution (CTDE). Uses a shared actor and centralized critic over concatenated observations.

## Preference-based reward modeling (MARP)

The trainer supports learning a reward model `r_hat(o, a)` from preferences derived
from social metrics (e.g., efficiency, peace). When enabled, the trainer uses the
learned reward instead of the environment reward for training, while still logging
environment rewards for analysis.

Key mechanics:
- `compute_social_metrics()` and `get_social_metrics()` provide metrics per episode.
- Preference pairs are generated from episode metrics (e.g., efficiency x peace).
- Reward model is trained via Bradley-Terry on trajectory pairs.
- DQN/IPPO/MAPPO use the external loop and swap `rewards` with `r_hat` in `Trainer.train()`.

Enable in config:

```json
{
  "reward_model": {
    "enabled": true,
    "mode": "narrow_view",
    "phi": "efficiency_x_peace",
    "lr": 0.0001,
    "batch_pairs": 64,
    "train_steps_per_update": 50,
    "update_every_env_steps": 1000,
    "warmup_episodes": 50,
    "max_episodes_in_buffer": 5000,
    "device": "auto",
    "save_every_episodes": 200,
    "use_amp": true,
    "chunk_size": 512,
    "max_steps_per_sequence": 256
  }
}
```

### Available phi functions

The `phi` parameter determines how social metrics are combined into a single preference score:

| Phi Key | Formula | Description |
|---------|---------|-------------|
| `efficiency` | efficiency | Maximize average reward per agent |
| `efficiency_x_peace` | efficiency × peace | Balance resource collection with non-aggressive behavior |
| `efficiency_x_equality` | efficiency × equality | Balance resource collection with fair distribution |
| `efficiency_x_sustainability` | efficiency × sustainability | Balance resource collection with long-term resource availability |
| `efficiency_x_peace_x_equality` | efficiency × peace × equality | Balance all three: collection, fairness, and non-aggression |
| `equality_x_peace` | equality × peace | Promote fair distribution and non-aggressive behavior |
| `efficiency_x_peace_x_equality_x_sustainability` | efficiency × peace × equality × sustainability | Balance all four metrics: collection, fairness, non-aggression, and long-term resource availability |

### Performance optimization options

The reward model training supports several performance optimizations for memory-constrained GPUs:

| Option | Default | Description |
|--------|---------|-------------|
| `use_amp` | `true` | Use mixed precision (FP16) training. Halves GPU memory usage and speeds up training on compatible GPUs. Automatically disabled on CPU. |
| `chunk_size` | `512` | Maximum number of steps per forward pass chunk. Larger values are faster but use more memory. Reduce if encountering OOM errors. |
| `max_steps_per_sequence` | `256` | Temporal subsampling limit. Limits the number of steps per trajectory using uniform spacing. Set to `null` to disable subsampling (process all steps). |

**Memory usage tips:**
- For 8GB GPU: Use defaults (`max_steps_per_sequence: 256`, `chunk_size: 512`, `use_amp: true`)
- For 4GB GPU: Try `max_steps_per_sequence: 128`, `chunk_size: 256`
- For larger GPUs: Increase `max_steps_per_sequence` to `512` or `null` for full precision

**Mode comparison:**
- `input_aggregation`: Aggregates trajectories from ALL agents (higher memory, captures global patterns)
- `narrow_view`: Samples single agent trajectory per episode (lower memory, faster)

Logging:
- `reward_pred_*` tracks predicted rewards when RM is enabled.
- `reward_env_*` tracks environment rewards.
- `reward_model/*` in TensorBoard shows RM loss/accuracy/correlation.

Checkpoints:
- All algorithms: `logs/<run>/model_last.pt`, `logs/<run>/reward_model_last.pt`

## Plotting run metrics

Generate reward and social-metric plots from a run folder (expects `metrics.jsonl`):

```bash
python scripts/plot_run_metrics.py logs/<run-name>
```

Outputs:
- `logs/<run-name>/plots/rewards.png`
- `logs/<run-name>/plots/social_metrics.png`

Options:
- `--smooth N`: moving average window (episodes); also adds a faded ±1 std band.
- `--normalize`: normalize each series to [0, 1] and plot social metrics on one graph.
- `--no-title`: hide titles from all plots (useful for publication figures where titles are added in captions).
- `--single-plots`: generate individual plot files for each social metric (`social_efficiency.png`, `social_equality.png`, etc.) instead of a combined 2x2 subplot.

### Plotting multiple runs (averaged)

Generate averaged plots with standard deviation across multiple runs:

```bash
python scripts/plot_multiple_runs.py logs/<run1> logs/<run2> logs/<run3> ...
```

Example:
```bash
python scripts/plot_multiple_runs.py logs/20251231-224618-mappo-map=small-agents=5-rm=narrow_view-seed=1814091097 logs/20251231-234319-mappo-map=small-agents=5-rm=narrow_view-seed=1942310406 logs/20260101-004021-mappo-map=small-agents=5-rm=narrow_view-seed=1364072973
```

Outputs:
- `plots_averaged/rewards_averaged.png` (or custom output directory)
- `plots_averaged/social_metrics_averaged.png`
- `plots_averaged/agent_predicted_rewards_normalized.png` (normalized per-agent predicted rewards)
- `plots_averaged/with_se/` - Same plots using Standard Error (SE) instead of Standard Deviation (STD)

Options:
- `--output-dir DIR` or `-o DIR`: output directory for plots (default: `plots_averaged`).
- `--smooth N`: moving average window (episodes); use 1 to disable smoothing.
- `--normalize`: normalize each metric series to [0, 1] and plot social metrics on one graph.
- `--no-title`: hide titles from all plots (useful for publication figures where titles are added in captions).

The script computes mean, standard deviation (STD), and standard error (SE) across all runs for each episode:
- **Standard Deviation (STD)**: Measures variability in the data: `STD = sqrt(Σ(x-μ)²/(n-1))`
- **Standard Error (SE)**: Measures uncertainty in the mean: `SE = STD / sqrt(n)`

STD is shown as shaded regions in the main output folder, SE versions are saved in the `with_se/` subfolder. SE is typically preferred for publication as it reflects confidence in the mean and shrinks with more runs.

**Publication Quality:** Plots are generated with publication-quality settings:
- 300 DPI resolution (suitable for high-quality printing)
- Serif fonts (Times New Roman) for professional appearance
- Colorblind-friendly color palette
- Clean styling with optimized spacing and grid

### Processing all sessions with cross-session comparisons

Generate per-session averaged plots and cross-session comparison plots:

```bash
python scripts/process_all_sessions.py
```

This script processes all defined experiment sessions and generates:
1. **Per-session plots**: Averaged metrics for each session, including:
   - Social metrics (efficiency, equality, sustainability, peace)
   - Predicted rewards by condition (No apple eaten, Eat with 0/+4 apples nearby)
   - Predicted rewards by granular condition (detailed breakdown by nearby apple count: 0, 1, 2, 3, +4)
   - Predicted rewards by action (movement directions)
2. **Cross-session comparisons**: Overlay plots, grid comparisons, and bar charts

**CLI options:**

```bash
# Run everything (default)
python scripts/process_all_sessions.py

# Run ONLY cross-session comparisons (skip per-session plots)
python scripts/process_all_sessions.py --comparisons-only

# Run ONLY per-session plots (skip comparisons)
python scripts/process_all_sessions.py --skip-comparisons

# Generate plots without titles (useful for publication figures)
python scripts/process_all_sessions.py --no-title
```

**Comparison outputs** (`logs/comparisons/`):
- `by_approach/`: Compare approaches (narrow view vs input aggregation) for same social target
  - `with_std/`: Same plots with standard deviation (STD) shading
  - `with_se/`: Same plots with standard error (SE) shading
- `by_target/`: Compare social targets for same approach
  - `with_std/`: Same plots with standard deviation (STD) shading
  - `with_se/`: Same plots with standard error (SE) shading
- `all_sessions/`: All sessions overlaid + grid comparisons
  - `with_std/`: Same plots with standard deviation (STD) shading
  - `with_se/`: Same plots with standard error (SE) shading
- `summary_bars/`: Bar charts with final and average values
  - `normalized/`: Same bar charts with values normalized to [0, 1] per metric for better cross-metric comparison

**Error bars explanation:**
- **Standard Deviation (STD)**: Shows the variability/spread of individual runs around the mean
- **Standard Error (SE)**: Shows the uncertainty in the estimated mean (SE = STD / √n, shrinks with more runs)

**Session naming convention:**
Sessions are automatically parsed from the format `"{approach} - {social_target}"`. New social targets are automatically integrated into comparisons by adding sessions following this naming convention.

Example session names:
- `"narrow view - efficiency"`
- `"input aggregation - efficiency x peace"`

### Plotting phi comparisons and NV vs IA galleries

Generate comparison plots for different phi values and social metrics galleries:

```bash
python scripts/plot_phi_comparisons.py --algorithm ippo
```

This script produces two types of plots:

**A. Phi Comparison Plots** - Compare efficiency metric across 4 configurations:
- Narrow View with φ=efficiency vs φ=efficiency×\<social\>
- Input Aggregation with φ=efficiency vs φ=efficiency×\<social\>

Generated for peace, equality, and sustainability social targets.

**B. Social Metrics Gallery (2x2)** - Compare all 4 social metrics (efficiency, equality, sustainability, peace) between Narrow View and Input Aggregation:
- Overall comparison (averaged across all phi values)
- Per-phi galleries (efficiency, efficiency×peace, efficiency×equality, efficiency×sustainability)

All plots use Standard Error (SE) for shading.

**CLI options:**

```bash
# Use IPPO sessions (default)
python scripts/plot_phi_comparisons.py --algorithm ippo

# Use MAPPO sessions
python scripts/plot_phi_comparisons.py --algorithm mappo

# Custom output directory
python scripts/plot_phi_comparisons.py -o custom/output/dir

# Hide titles (for publication figures)
python scripts/plot_phi_comparisons.py --no-title

# Custom smoothing window (default: 10)
python scripts/plot_phi_comparisons.py --smooth 20
```

The script generates both unsmoothed and smoothed versions of all plots. Smoothing applies a moving average to the mean line while keeping SE (standard error) bands unchanged.

**Outputs** (`logs/<algorithm>/comparisons/phi_comparisons/`):
- `phi_comparison_efficiency_vs_peace.png` - Peace metric comparison: φ=efficiency vs φ=efficiency×peace
- `phi_comparison_efficiency_vs_equality.png` - Equality metric comparison: φ=efficiency vs φ=efficiency×equality
- `phi_comparison_efficiency_vs_sustainability.png` - Sustainability metric comparison: φ=efficiency vs φ=efficiency×sustainability
- `social_metrics_gallery_nv_vs_ia_overall.png` - Overall NV vs IA comparison
- `social_metrics_gallery_<phi>.png` - Per-phi social metrics galleries
- `smoothed_<N>/` - Subdirectory with smoothed versions (same files, moving average applied to means)

## Running IPPO

IPPO (Independent PPO) uses true decentralized training where all agents train
simultaneously with their own learning policies. Each agent has independent actor
and critic networks that use local observations only.

```json
{
  "env": {"num_agents": 5},
  "algorithm": {
    "name": "ippo",
    "ippo": {
      "learning_rate": 0.0003,
      "n_steps": 1024,
      "batch_size": 256,
      "update_epochs": 4,
      "hidden_size": 256,
      "flatten_obs": false,
      "normalize_obs": false
    }
  }
}
```

Key differences from MAPPO:
- **Decentralized critics**: Each agent's critic uses only its local observation (not global state).
- **Independent networks**: Each agent has its own actor and critic networks.
- **No parameter sharing**: Agents do not share any network weights.

## Running MAPPO

MAPPO uses a shared actor and centralized critic over concatenated observations.
Enable it by switching the algorithm name and configuring the `mappo` block:

```json
{
  "env": {"num_agents": 5},
  "algorithm": {
    "name": "mappo",
    "mappo": {
      "n_steps": 1024,
      "batch_size": 256,
      "update_epochs": 4,
      "flatten_obs": false,
      "normalize_obs": true
    }
  }
}
```
