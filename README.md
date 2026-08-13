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
- `src/commons_game_marp/analysis/`: Plotting and cross-session analysis commands.
- `src/commons_game_marp/cli.py`: The `commons-game` command.

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
uv run pytest                    # run the test suite
uv run commons-game --help       # see all commands
```

## The `commons-game` command

Everything runs through one command:

| Command | Does |
|---|---|
| `commons-game train` | Train agents (Hydra — see [Training](#training)) |
| `commons-game plot run <run-dir>` | Plot reward and social metrics for one run |
| `commons-game plot runs <run-dirs>` | Plot averaged metrics with std dev across runs |
| `commons-game plot phi` | Phi comparison plots and NV vs IA galleries |
| `commons-game compare-modes` | Compare narrow-view against input-aggregation runs |
| `commons-game sessions` | Process sessions and generate cross-session plots |
| `commons-game tensorboard` | Launch TensorBoard on a log directory |

Run `commons-game <command> --help` for a command's options.

`commons-game-train` and `uv run python main.py` remain as aliases for the
training path.

### PyTorch and CUDA

The pinned build is `torch==2.5.1+cu124`, resolved from PyTorch's CUDA 12.4
index (configured in `pyproject.toml`). To run on CPU or a different CUDA
version, change the `torch` pin and the `[[tool.uv.index]]` URL, then re-run
`uv sync`.

## Training

Training is configured with [Hydra](https://hydra.cc/). Run with defaults:

```bash
uv run commons-game train
```

Override any value from the command line:

```bash
uv run commons-game train algorithm=ippo env=medium episodes=300 seed=7
uv run commons-game train reward_model=off env.penalty=true
uv run commons-game train algorithm=mappo algorithm.learning_rate=1e-4
```

Print the composed config without training:

```bash
uv run commons-game train --cfg job
```

`uv run python main.py` is equivalent — `main.py` is a two-line shim over the
same entry point.

### Config groups

Configs live in `src/commons_game_marp/configs/`, split into groups:

| Group | Values | Selects |
|---|---|---|
| `env` | `small`, `medium` | Map size and agent count |
| `algorithm` | `dqn`, `ippo`, `mappo`, `random` | Learner and its hyperparameters |
| `reward_model` | `off`, `narrow_view`, `input_aggregation` | Preference-based reward modeling |
| `logging` | `default` | Log directory, video capture |

Top-level `episodes` and `seed` are set in `config.yaml` and overridable
directly.

### Experiment presets

Named combinations live in `configs/experiment/` and are selected with a `+`:

```bash
uv run commons-game train +experiment=mappo
uv run commons-game train +experiment=ippo episodes=500
```

### Sweeps

`--multirun` (`-m`) runs the cross-product of comma-separated values
sequentially:

```bash
# Three algorithms x three seeds = 9 runs
uv run commons-game train -m algorithm=dqn,ippo,mappo seed=0,1,2

# Sweep a hyperparameter
uv run commons-game train -m algorithm=ippo algorithm.learning_rate=1e-3,3e-4,1e-4

# Compare reward-model modes across five seeds
uv run commons-game train -m +experiment=sequence_narrow_vs_input_agg \
    reward_model=narrow_view,input_aggregation seed=0,1,2,3,4
```

Multirun output is grouped under `logs/hydra/multirun/<timestamp>/`.

### Outputs

Logs are written to `logs/<run-name>/metrics.jsonl` and `logs/<run-name>/config.yaml`.
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

Config tips (each is a command-line override, e.g. `logging.video_enabled=false`):
- `logging.video_every_n_episodes` defaults to 100; reduce it to record more frequently.
- `logging.video_max_steps` caps episode length in videos.
- `logging.video_enabled=false` disables video capture for faster training.
- `logging.log_agent_episode_details=true` (default) enables detailed per-agent episode logging to separate files.
- The last episode is always recorded (if video is enabled), regardless of `video_every_n_episodes`.
- Ready-made combinations: `+experiment=dqn`, `+experiment=ippo`, `+experiment=mappo`.

## Algorithms

Select the learner with the `algorithm` group. Supported values: `dqn`,
`random`, `ippo`, `mappo`.

```bash
uv run commons-game train algorithm=ippo
```

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

Enable it by selecting a `reward_model` group value (`off` is the default):

```bash
uv run commons-game train reward_model=narrow_view
uv run commons-game train reward_model=input_aggregation reward_model.phi=efficiency_x_equality
```

The `narrow_view` and `input_aggregation` presets set:

```yaml
enabled: true
mode: narrow_view          # or input_aggregation
phi: efficiency_x_peace
lr: 0.0001
batch_pairs: 64
train_steps_per_update: 50
update_every_env_steps: 1000
warmup_episodes: 50
max_episodes_in_buffer: 5000
device: auto
save_every_episodes: 200
```

`use_amp`, `chunk_size`, and `max_steps_per_sequence` take their schema defaults
(`true`, `512`, `256`) and are overridable the same way, e.g.
`reward_model.chunk_size=256`.

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
uv run commons-game plot run logs/<run-name>
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
uv run commons-game plot runs logs/<run1> logs/<run2> logs/<run3> ...
```

Example:
```bash
uv run commons-game plot runs logs/20251231-224618-mappo-map=small-agents=5-rm=narrow_view-seed=1814091097 logs/20251231-234319-mappo-map=small-agents=5-rm=narrow_view-seed=1942310406 logs/20260101-004021-mappo-map=small-agents=5-rm=narrow_view-seed=1364072973
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
uv run commons-game sessions
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
uv run commons-game sessions

# Run ONLY cross-session comparisons (skip per-session plots)
uv run commons-game sessions --comparisons-only

# Run ONLY per-session plots (skip comparisons)
uv run commons-game sessions --skip-comparisons

# Generate plots without titles (useful for publication figures)
uv run commons-game sessions --no-title
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
uv run commons-game plot phi --algorithm ippo
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
uv run commons-game plot phi --algorithm ippo

# Use MAPPO sessions
uv run commons-game plot phi --algorithm mappo

# Custom output directory
uv run commons-game plot phi -o custom/output/dir

# Hide titles (for publication figures)
uv run commons-game plot phi --no-title

# Custom smoothing window (default: 10)
uv run commons-game plot phi --smooth 20
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

```bash
uv run commons-game train algorithm=ippo env.num_agents=5 \
    algorithm.learning_rate=0.0003 algorithm.n_steps=1024 \
    algorithm.batch_size=256 algorithm.update_epochs=4 \
    algorithm.hidden_size=256 algorithm.flatten_obs=false \
    algorithm.normalize_obs=false
```

Persistent changes go in `src/commons_game_marp/configs/algorithm/ippo.yaml`.

Key differences from MAPPO:
- **Decentralized critics**: Each agent's critic uses only its local observation (not global state).
- **Independent networks**: Each agent has its own actor and critic networks.
- **No parameter sharing**: Agents do not share any network weights.

## Running MAPPO

MAPPO uses a shared actor and centralized critic over concatenated observations.
Enable it by selecting the `mappo` algorithm group:

```bash
uv run commons-game train algorithm=mappo env.num_agents=5 \
    algorithm.n_steps=1024 algorithm.batch_size=256 \
    algorithm.update_epochs=4 algorithm.flatten_obs=false \
    algorithm.normalize_obs=true
```

Persistent changes go in `src/commons_game_marp/configs/algorithm/mappo.yaml`.
