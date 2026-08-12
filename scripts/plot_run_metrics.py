import argparse
import csv
import json
import os
from typing import Any, Dict, Iterable, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import math

# Publication-quality settings
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'axes.linewidth': 1.0,
    'grid.linewidth': 0.5,
    'lines.linewidth': 2.0,
    'patch.linewidth': 1.0,
    'xtick.major.width': 1.0,
    'ytick.major.width': 1.0,
    'xtick.minor.width': 0.5,
    'ytick.minor.width': 0.5,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    'text.usetex': False,  # Use matplotlib's built-in math rendering instead of LaTeX
    'mathtext.fontset': 'stix',  # Use STIX fonts for math symbols
})

# Colorblind-friendly color palette
PUBLICATION_COLORS = [
    '#1f77b4',  # blue
    '#ff7f0e',  # orange
    '#2ca02c',  # green
    '#d62728',  # red
    '#9467bd',  # purple
    '#8c564b',  # brown
    '#e377c2',  # pink
    '#7f7f7f',  # gray
    '#bcbd22',  # olive
    '#17becf',  # cyan
]


REWARD_KEYS = (
    "reward_sum",
    "reward_mean",
    "reward_pred_sum",
    "reward_pred_mean",
    "reward_env_sum",
    "reward_env_mean",
)
SOCIAL_ORDER = ("efficiency", "equality", "sustainability", "peace")


def _load_metrics(metrics_path: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with open(metrics_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _iter_records_with_episode(
    records: Iterable[Dict[str, Any]],
) -> Iterable[Tuple[int, Dict[str, Any]]]:
    for record in records:
        episode = record.get("episode")
        if isinstance(episode, int):
            yield episode, record
        elif isinstance(episode, float):
            yield int(episode), record


def _normalize_social_metrics(value: Any) -> Dict[str, float]:
    if isinstance(value, dict):
        return {k: float(v) for k, v in value.items() if isinstance(v, (int, float))}
    if isinstance(value, (list, tuple)) and len(value) == 4:
        return {
            name: float(metric)
            for name, metric in zip(SOCIAL_ORDER, value)
            if isinstance(metric, (int, float))
        }
    return {}


def _smooth_points_with_std(
    points: List[Tuple[int, float]], window: int
) -> Tuple[List[Tuple[int, float]], List[float]]:
    if window <= 1 or len(points) < 2:
        return points, []
    points = sorted(points, key=lambda item: item[0])
    values = [value for _, value in points]
    smoothed: List[Tuple[int, float]] = []
    stds: List[float] = []
    for idx in range(len(values)):
        start = max(0, idx - window + 1)
        window_values = values[start : idx + 1]
        mean = sum(window_values) / len(window_values)
        variance = sum((val - mean) ** 2 for val in window_values) / len(window_values)
        std = math.sqrt(variance)
        smoothed.append((points[idx][0], mean))
        stds.append(std)
    return smoothed, stds


def _plot_series(
    series: Dict[str, List[Tuple[int, float]]],
    title: str,
    ylabel: str,
    output_path: str,
    smooth_window: int,
    show_title: bool = True,
) -> bool:
    if not series:
        return False
    fig, ax = plt.subplots(figsize=(6, 4))
    plotted = False
    color_cycle = iter(PUBLICATION_COLORS)
    
    for name, points in series.items():
        if not points:
            continue
        points, stds = _smooth_points_with_std(points, smooth_window)
        xs = [episode for episode, _ in points]
        ys = [value for _, value in points]
        color = next(color_cycle)
        ax.plot(xs, ys, label=name, color=color, linewidth=2.0, zorder=2)
        if smooth_window > 1 and stds:
            lower = [y - s for y, s in zip(ys, stds)]
            upper = [y + s for y, s in zip(ys, stds)]
            ax.fill_between(xs, lower, upper, alpha=0.25, color=color, zorder=1)
        plotted = True
    if not plotted:
        plt.close()
        return False
    if show_title:
        ax.set_title(title, fontweight='bold', pad=10)
    ax.set_xlabel("Episode", fontweight='normal')
    ax.set_ylabel(ylabel, fontweight='normal')
    ax.grid(True, linestyle='--', alpha=0.3, linewidth=0.5, zorder=0)
    ax.legend(loc='best', frameon=True, fancybox=True, shadow=False, framealpha=0.9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    return True


def _normalize_series(series: Dict[str, List[Tuple[int, float]]]) -> Dict[str, List[Tuple[int, float]]]:
    normalized: Dict[str, List[Tuple[int, float]]] = {}
    for name, points in series.items():
        if not points:
            continue
        points = sorted(points, key=lambda item: item[0])
        values = [value for _, value in points]
        min_val = min(values)
        max_val = max(values)
        denom = max_val - min_val
        if denom == 0:
            normalized[name] = [(episode, 0.0) for episode, _ in points]
            continue
        normalized[name] = [(episode, (value - min_val) / denom) for episode, value in points]
    return normalized


def _load_run_context(run_dir: str) -> Tuple[str, str]:
    config_path = os.path.join(run_dir, "config.json")
    if not os.path.isfile(config_path):
        return "unknown", ""
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except json.JSONDecodeError:
        return "unknown", ""
    algo_name = payload.get("algorithm", {}).get("name", "unknown")
    rm_phi = ""
    reward_model_cfg = payload.get("reward_model")
    if isinstance(reward_model_cfg, dict) and reward_model_cfg.get("enabled"):
        phi = reward_model_cfg.get("phi")
        if isinstance(phi, str) and phi:
            rm_phi = phi
    return algo_name, rm_phi


def _plot_single_social_metric(
    name: str,
    points: List[Tuple[int, float]],
    output_path: str,
    smooth_window: int,
    show_title: bool = True,
) -> bool:
    """Plot a single social metric as an individual graph."""
    if not points:
        return False
    
    smoothed_points, stds = _smooth_points_with_std(points, smooth_window)
    xs = [episode for episode, _ in smoothed_points]
    ys = [value for _, value in smoothed_points]
    
    fig, ax = plt.subplots(figsize=(6, 4))
    color = PUBLICATION_COLORS[0]
    
    ax.plot(xs, ys, label=name.capitalize(), color=color, linewidth=2.0, zorder=2)
    if smooth_window > 1 and stds:
        lower = [y - s for y, s in zip(ys, stds)]
        upper = [y + s for y, s in zip(ys, stds)]
        ax.fill_between(xs, lower, upper, alpha=0.25, color=color, zorder=1)
    
    ax.set_xlabel("Episode", fontweight='normal')
    ax.set_ylabel(name.capitalize(), fontweight='normal')
    if show_title:
        ax.set_title(name.capitalize(), fontweight='bold', pad=10)
    ax.grid(True, linestyle='--', alpha=0.3, linewidth=0.5, zorder=0)
    ax.legend(loc='best', frameon=True, fancybox=True, shadow=False, framealpha=0.9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    return True


def _plot_social_subplots(
    series: Dict[str, List[Tuple[int, float]]],
    title: str,
    output_path: str,
    smooth_window: int,
    show_title: bool = True,
) -> bool:
    ordered_names = [name for name in SOCIAL_ORDER if series.get(name)]
    if not ordered_names:
        ordered_names = [name for name, points in series.items() if points]
    if not ordered_names:
        return False

    # Capitalize metric names for display
    def capitalize_metric(name: str) -> str:
        return name.capitalize()

    if len(ordered_names) == 4:
        fig, axes = plt.subplots(2, 2, figsize=(7, 5), sharex=True)
        axes_list = axes.flatten()
    else:
        fig, axes = plt.subplots(
            len(ordered_names), 1, figsize=(6, 2.5 * len(ordered_names)), sharex=True
        )
        axes_list = [axes] if len(ordered_names) == 1 else list(axes)

    color = PUBLICATION_COLORS[0]  # Use consistent color for all subplots

    for ax, name in zip(axes_list, ordered_names):
        points, stds = _smooth_points_with_std(series[name], smooth_window)
        xs = [episode for episode, _ in points]
        ys = [value for _, value in points]
        ax.plot(xs, ys, label=capitalize_metric(name), color=color, linewidth=2.0, zorder=2)
        if smooth_window > 1 and stds:
            lower = [y - s for y, s in zip(ys, stds)]
            upper = [y + s for y, s in zip(ys, stds)]
            ax.fill_between(xs, lower, upper, alpha=0.25, color=color, zorder=1)
        ax.set_ylabel(capitalize_metric(name), fontweight='normal')
        ax.grid(True, linestyle='--', alpha=0.3, linewidth=0.5, zorder=0)
        ax.legend(loc='best', frameon=True, fancybox=True, shadow=False, framealpha=0.9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    if len(ordered_names) == 4:
        for ax in axes_list[-2:]:
            ax.set_xlabel("Episode", fontweight='normal')
    else:
        axes_list[-1].set_xlabel("Episode", fontweight='normal')
    if show_title:
        fig.suptitle(title, fontweight='bold', y=0.995)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
    else:
        fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    return True


def _load_agent_csv(agent_csv_path: str) -> List[Dict[str, Any]]:
    """Load agent episode CSV file and return list of row dictionaries."""
    rows = []
    if not os.path.isfile(agent_csv_path):
        return rows
    try:
        with open(agent_csv_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
    except (csv.Error, IOError):
        return rows
    return rows


def _parse_csv_value(value: str, field: str) -> Any:
    """Parse CSV value based on field type."""
    if not value or value == "":
        return None
    if field in ("episode", "step", "action", "nearby_apples"):
        try:
            return int(value)
        except ValueError:
            return None
    if field in ("reward", "predicted_reward"):
        try:
            return float(value)
        except ValueError:
            return None
    if field in ("apple_eaten", "ate_last_apple_in_cluster"):
        # Handle boolean values (True/False, 1/0, etc.)
        value_lower = value.lower().strip()
        if value_lower in ("true", "1", "yes"):
            return True
        if value_lower in ("false", "0", "no", ""):
            return False
        return None
    return value


def _extract_agent_predicted_reward_series(
    agent_csv_path: str,
) -> Dict[str, List[Tuple[int, Tuple[float, float]]]]:
    """
    Extract predicted reward series for three conditions:
    - no_apple_eaten: average when apple_eaten == False
    - zero_apples_nearby: average when nearby_apples == 0
    - four_plus_apples_nearby: average when nearby_apples >= 4
    
    Returns dict with keys as condition names, values as list of (episode, (mean, std)) tuples.
    """
    rows = _load_agent_csv(agent_csv_path)
    if not rows:
        return {}
    
    # Group by episode
    episodes_data: Dict[int, Dict[str, List[float]]] = {}
    
    for row in rows:
        episode = _parse_csv_value(row.get("episode", ""), "episode")
        if episode is None:
            continue
        
        predicted_reward = _parse_csv_value(row.get("predicted_reward", ""), "predicted_reward")
        if predicted_reward is None:
            continue
        
        apple_eaten = _parse_csv_value(row.get("apple_eaten", ""), "apple_eaten")
        nearby_apples = _parse_csv_value(row.get("nearby_apples", ""), "nearby_apples")
        
        if episode not in episodes_data:
            episodes_data[episode] = {
                "no_apple_eaten": [],
                "zero_apples_nearby": [],
                "four_plus_apples_nearby": [],
            }
        
        # Condition 1: no apple eaten
        if apple_eaten is False:
            episodes_data[episode]["no_apple_eaten"].append(predicted_reward)
        
        # Condition 2: 0 apples nearby
        if nearby_apples == 0:
            episodes_data[episode]["zero_apples_nearby"].append(predicted_reward)
        
        # Condition 3: 4+ apples nearby
        if nearby_apples is not None and nearby_apples >= 4:
            episodes_data[episode]["four_plus_apples_nearby"].append(predicted_reward)
    
    # Calculate mean and std for each episode and condition
    series: Dict[str, List[Tuple[int, Tuple[float, float]]]] = {
        "no_apple_eaten": [],
        "zero_apples_nearby": [],
        "four_plus_apples_nearby": [],
    }
    
    for episode in sorted(episodes_data.keys()):
        for condition in series.keys():
            values = episodes_data[episode][condition]
            if values:
                mean = sum(values) / len(values)
                variance = sum((v - mean) ** 2 for v in values) / len(values)
                std = math.sqrt(variance)
                series[condition].append((episode, (mean, std)))
    
    return series


def _plot_agent_predicted_rewards(
    series: Dict[str, List[Tuple[int, Tuple[float, float]]]],
    agent_id: str,
    title: str,
    output_path: str,
    smooth_window: int,
    show_title: bool = True,
) -> bool:
    """Plot predicted reward series for an agent with three conditions."""
    if not series or not any(series.values()):
        return False
    
    fig, ax = plt.subplots(figsize=(6, 4))
    plotted = False
    
    condition_labels = {
        "no_apple_eaten": "No apple eaten",
        "zero_apples_nearby": "Eat, 0 apples nearby",
        "four_plus_apples_nearby": "Eat, +4 apples nearby",
    }
    
    color_cycle = iter(PUBLICATION_COLORS)
    
    for condition, points in series.items():
        if not points:
            continue
        
        # Extract episodes, means, and stds
        episodes = [ep for ep, _ in points]
        means = [mean for _, (mean, _) in points]
        stds = [std for _, (_, std) in points]
        
        # Smooth the means and average the stds in the window
        if smooth_window > 1 and len(means) > 1:
            smoothed_means = []
            smoothed_stds = []
            for idx in range(len(means)):
                start = max(0, idx - smooth_window + 1)
                window_means = means[start : idx + 1]
                window_stds = stds[start : idx + 1]
                smoothed_mean = sum(window_means) / len(window_means)
                # Average the stds in the window to show average within-episode variability
                smoothed_std = sum(window_stds) / len(window_stds) if window_stds else 0.0
                smoothed_means.append(smoothed_mean)
                smoothed_stds.append(smoothed_std)
            means = smoothed_means
            stds = smoothed_stds
        
        label = condition_labels.get(condition, condition)
        color = next(color_cycle)
        ax.plot(episodes, means, label=label, color=color, linewidth=2.0, zorder=2)
        
        # Add std as background fill
        if stds:
            lower = [m - s for m, s in zip(means, stds)]
            upper = [m + s for m, s in zip(means, stds)]
            ax.fill_between(episodes, lower, upper, alpha=0.25, color=color, zorder=1)
        
        plotted = True
    
    if not plotted:
        plt.close()
        return False
    
    if show_title:
        ax.set_title(title, fontweight='bold', pad=10)
    ax.set_xlabel("Episode", fontweight='normal')
    ax.set_ylabel("Average Predicted Reward", fontweight='normal')
    ax.grid(True, linestyle='--', alpha=0.3, linewidth=0.5, zorder=0)
    ax.legend(loc='best', frameon=True, fancybox=True, shadow=False, framealpha=0.9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    return True


def _extract_agent_predicted_reward_by_action_series(
    agent_csv_path: str,
) -> Dict[str, List[Tuple[int, Tuple[float, float]]]]:
    """
    Extract predicted reward series for movement actions:
    - move_left: average when action == 0
    - move_right: average when action == 1
    - move_up: average when action == 2
    - move_down: average when action == 3
    
    Returns dict with keys as action names, values as list of (episode, (mean, std)) tuples.
    """
    rows = _load_agent_csv(agent_csv_path)
    if not rows:
        return {}
    
    # Group by episode and action
    episodes_data: Dict[int, Dict[int, List[float]]] = {}
    
    for row in rows:
        episode = _parse_csv_value(row.get("episode", ""), "episode")
        if episode is None:
            continue
        
        predicted_reward = _parse_csv_value(row.get("predicted_reward", ""), "predicted_reward")
        if predicted_reward is None:
            continue
        
        action = _parse_csv_value(row.get("action", ""), "action")
        if action is None or action not in (0, 1, 2, 3):  # Only movement actions
            continue
        
        if episode not in episodes_data:
            episodes_data[episode] = {0: [], 1: [], 2: [], 3: []}
        
        episodes_data[episode][action].append(predicted_reward)
    
    # Calculate mean and std for each episode and action
    action_names = {0: "move_left", 1: "move_right", 2: "move_up", 3: "move_down"}
    series: Dict[str, List[Tuple[int, Tuple[float, float]]]] = {
        name: [] for name in action_names.values()
    }
    
    for episode in sorted(episodes_data.keys()):
        for action, name in action_names.items():
            values = episodes_data[episode][action]
            if values:
                mean = sum(values) / len(values)
                variance = sum((v - mean) ** 2 for v in values) / len(values)
                std = math.sqrt(variance)
                series[name].append((episode, (mean, std)))
    
    return series


def _plot_agent_predicted_rewards_by_action(
    series: Dict[str, List[Tuple[int, Tuple[float, float]]]],
    agent_id: str,
    title: str,
    output_path: str,
    smooth_window: int,
    show_title: bool = True,
) -> bool:
    """Plot predicted reward series for an agent by movement actions."""
    if not series or not any(series.values()):
        return False
    
    fig, ax = plt.subplots(figsize=(6, 4))
    plotted = False
    
    action_labels = {
        "move_left": "Move Left",
        "move_right": "Move Right",
        "move_up": "Move Up",
        "move_down": "Move Down",
    }
    
    # Order actions consistently
    action_order = ["move_left", "move_right", "move_up", "move_down"]
    color_cycle = iter(PUBLICATION_COLORS)
    
    for action_name in action_order:
        if action_name not in series:
            continue
        points = series[action_name]
        if not points:
            continue
        
        # Extract episodes, means, and stds
        episodes = [ep for ep, _ in points]
        means = [mean for _, (mean, _) in points]
        stds = [std for _, (_, std) in points]
        
        # Smooth the means and average the stds in the window
        if smooth_window > 1 and len(means) > 1:
            smoothed_means = []
            smoothed_stds = []
            for idx in range(len(means)):
                start = max(0, idx - smooth_window + 1)
                window_means = means[start : idx + 1]
                window_stds = stds[start : idx + 1]
                smoothed_mean = sum(window_means) / len(window_means)
                # Average the stds in the window to show average within-episode variability
                smoothed_std = sum(window_stds) / len(window_stds) if window_stds else 0.0
                smoothed_means.append(smoothed_mean)
                smoothed_stds.append(smoothed_std)
            means = smoothed_means
            stds = smoothed_stds
        
        label = action_labels.get(action_name, action_name)
        color = next(color_cycle)
        ax.plot(episodes, means, label=label, color=color, linewidth=2.0, zorder=2)
        
        # Add std as background fill
        if stds:
            lower = [m - s for m, s in zip(means, stds)]
            upper = [m + s for m, s in zip(means, stds)]
            ax.fill_between(episodes, lower, upper, alpha=0.25, color=color, zorder=1)
        
        plotted = True
    
    if not plotted:
        plt.close()
        return False
    
    if show_title:
        ax.set_title(title, fontweight='bold', pad=10)
    ax.set_xlabel("Episode", fontweight='normal')
    ax.set_ylabel("Average Predicted Reward", fontweight='normal')
    ax.grid(True, linestyle='--', alpha=0.3, linewidth=0.5, zorder=0)
    ax.legend(loc='best', frameon=True, fancybox=True, shadow=False, framealpha=0.9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    return True


def generate_agent_predicted_reward_plots(
    run_dir: str, smooth_window: int, show_title: bool = True
) -> Dict[str, bool]:
    """Generate predicted reward plots for each agent."""
    extended_info_dir = os.path.join(run_dir, "extended_info")
    if not os.path.isdir(extended_info_dir):
        return {}
    
    plots_dir = os.path.join(run_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    algo_name, rm_phi = _load_run_context(run_dir)
    
    # Find all agent CSV files
    agent_files = {}
    for filename in os.listdir(extended_info_dir):
        if filename.startswith("agent_") and filename.endswith("_episodes.csv"):
            agent_id = filename[len("agent_") : -len("_episodes.csv")]
            agent_csv_path = os.path.join(extended_info_dir, filename)
            agent_files[agent_id] = agent_csv_path
    
    results = {}
    
    for agent_id, agent_csv_path in agent_files.items():
        # Plot predicted rewards by condition
        series = _extract_agent_predicted_reward_series(agent_csv_path)
        if series and any(series.values()):
            title = f"Predicted Reward per Condition - Agent {agent_id} (algo={algo_name})"
            if rm_phi:
                title = f"{title}, reward_model_phi={rm_phi}"
            
            output_path = os.path.join(plots_dir, f"agent_{agent_id}_predicted_rewards.png")
            plotted = _plot_agent_predicted_rewards(
                series, agent_id, title, output_path, smooth_window, show_title
            )
            results[f"{agent_id}_condition"] = plotted
        
        # Plot predicted rewards by action
        action_series = _extract_agent_predicted_reward_by_action_series(agent_csv_path)
        if action_series and any(action_series.values()):
            title = f"Predicted Reward per Action - Agent {agent_id} (algo={algo_name})"
            if rm_phi:
                title = f"{title}, reward_model_phi={rm_phi}"
            
            output_path = os.path.join(plots_dir, f"agent_{agent_id}_predicted_rewards_by_action.png")
            plotted = _plot_agent_predicted_rewards_by_action(
                action_series, agent_id, title, output_path, smooth_window, show_title
            )
            results[f"{agent_id}_action"] = plotted
    
    return results


def generate_run_plots(
    run_dir: str, smooth_window: int, normalize: bool, show_title: bool = True,
    single_plots: bool = False
) -> Tuple[bool, bool]:
    metrics_path = os.path.join(run_dir, "metrics.jsonl")
    if not os.path.isfile(metrics_path):
        # Try metrics.json as fallback (some runs use .json extension for JSONL format)
        metrics_path = os.path.join(run_dir, "metrics.json")
        if not os.path.isfile(metrics_path):
            raise FileNotFoundError(f"metrics.jsonl or metrics.json not found in {run_dir}")

    algo_name, rm_phi = _load_run_context(run_dir)
    reward_title = f"Rewards (algo={algo_name})"
    if rm_phi:
        reward_title = f"{reward_title}, reward_model_phi={rm_phi}"
    social_title = f"Social Metrics (algo={algo_name})"

    records = _load_metrics(metrics_path)
    rewards: Dict[str, List[Tuple[int, float]]] = {key: [] for key in REWARD_KEYS}
    social: Dict[str, List[Tuple[int, float]]] = {}

    plots_dir = os.path.join(run_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    for episode, record in _iter_records_with_episode(records):
        for key in REWARD_KEYS:
            value = record.get(key)
            if isinstance(value, (int, float)):
                rewards[key].append((episode, float(value)))
        social_metrics = _normalize_social_metrics(record.get("social_metrics"))
        for name, value in social_metrics.items():
            social.setdefault(name, []).append((episode, value))

    rewards_path = os.path.join(plots_dir, "rewards.png")
    social_path = os.path.join(plots_dir, "social_metrics.png")
    rewards_series = {k: v for k, v in rewards.items() if v}
    social_series = {k: social.get(k, []) for k in SOCIAL_ORDER if social.get(k)}
    if not social_series:
        social_series = {k: v for k, v in social.items() if v}
    if normalize:
        rewards_series = _normalize_series(rewards_series)
        social_series = _normalize_series(social_series)
        reward_title = f"{reward_title} (normalized)"
        social_title = f"{social_title} (normalized)"
        reward_ylabel = "Normalized value"
        social_ylabel = "Normalized value"
    else:
        reward_ylabel = "Reward"
        social_ylabel = "Metric"
    rewards_plotted = _plot_series(
        rewards_series,
        title=reward_title,
        ylabel=reward_ylabel,
        output_path=rewards_path,
        smooth_window=smooth_window,
        show_title=show_title,
    )
    
    # Generate single plots for each social metric if requested
    if single_plots:
        social_plotted = False
        for name, points in social_series.items():
            if points:
                single_path = os.path.join(plots_dir, f"social_{name}.png")
                if _plot_single_social_metric(name, points, single_path, smooth_window, show_title):
                    social_plotted = True
    elif normalize:
        social_plotted = _plot_series(
            social_series,
            title=social_title,
            ylabel=social_ylabel,
            output_path=social_path,
            smooth_window=smooth_window,
            show_title=show_title,
        )
    else:
        social_plotted = _plot_social_subplots(
            social_series,
            title=social_title,
            output_path=social_path,
            smooth_window=smooth_window,
            show_title=show_title,
        )
    return rewards_plotted, social_plotted


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate reward and social metric plots from a run folder."
    )
    parser.add_argument(
        "run_dir",
        help="Path to a run folder containing metrics.jsonl or a metrics.jsonl file.",
    )
    parser.add_argument(
        "--smooth",
        type=int,
        default=1,
        help="Moving average window (episodes). Use 1 to disable smoothing.",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Normalize each metric series to [0, 1] and plot social metrics on one graph.",
    )
    parser.add_argument(
        "--no-title",
        action="store_true",
        help="Hide titles from all plots.",
    )
    parser.add_argument(
        "--single-plots",
        action="store_true",
        help="Generate individual plot files for each social metric instead of a combined subplot.",
    )
    args = parser.parse_args()

    path = args.run_dir
    if os.path.isfile(path):
        run_dir = os.path.dirname(path)
    else:
        run_dir = path

    show_title = not args.no_title

    rewards_plotted, social_plotted = generate_run_plots(
        run_dir,
        smooth_window=args.smooth,
        normalize=args.normalize,
        show_title=show_title,
        single_plots=args.single_plots,
    )
    if rewards_plotted:
        print(f"Saved rewards plot to {os.path.join(run_dir, 'plots', 'rewards.png')}")
    else:
        print("No reward metrics found to plot.")
    if social_plotted:
        if args.single_plots:
            print(f"Saved individual social metric plots to {os.path.join(run_dir, 'plots', 'social_<metric>.png')}")
        else:
            print(f"Saved social metrics plot to {os.path.join(run_dir, 'plots', 'social_metrics.png')}")
    else:
        print("No social metrics found to plot.")
    
    # Generate per-agent predicted reward plots
    agent_results = generate_agent_predicted_reward_plots(
        run_dir, smooth_window=args.smooth, show_title=show_title
    )
    if agent_results:
        for key, plotted in agent_results.items():
            if plotted:
                if key.endswith("_condition"):
                    agent_id = key[:-10]  # Remove "_condition" suffix
                    print(f"Saved agent {agent_id} predicted reward plot (by condition) to {os.path.join(run_dir, 'plots', f'agent_{agent_id}_predicted_rewards.png')}")
                elif key.endswith("_action"):
                    agent_id = key[:-7]  # Remove "_action" suffix
                    print(f"Saved agent {agent_id} predicted reward plot (by action) to {os.path.join(run_dir, 'plots', f'agent_{agent_id}_predicted_rewards_by_action.png')}")
            else:
                if key.endswith("_condition"):
                    agent_id = key[:-10]
                    print(f"No predicted reward data found for agent {agent_id} (by condition).")
                elif key.endswith("_action"):
                    agent_id = key[:-7]
                    print(f"No predicted reward data found for agent {agent_id} (by action).")
    else:
        print("No agent episode CSV files found to plot.")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
