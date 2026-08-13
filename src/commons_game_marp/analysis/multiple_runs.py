import argparse
import json
import os
from typing import Any, Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

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


def _format_label(label: str) -> str:
    """
    Format label for display:
    - Convert 'efficiency_x_peace' to 'Efficiency $\\times$ Peace' (LaTeX math with capitalization)
    - Replace underscores with spaces and capitalize words in other cases
    """
    if '_x_' in label:
        # Split by _x_, capitalize each part, and join with LaTeX \times
        parts = label.split('_x_')
        formatted_parts = [p.replace('_', ' ').title() for p in parts]
        return r' $\times$ '.join(formatted_parts)
    else:
        # Just replace underscores with spaces and capitalize
        return label.replace('_', ' ').title()


def _load_metrics(metrics_path: str) -> List[Dict[str, Any]]:
    """Load metrics from a JSONL file."""
    records: List[Dict[str, Any]] = []
    with open(metrics_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _extract_metric_series(
    records: List[Dict[str, Any]], metric_path: str
) -> List[Tuple[int, float]]:
    """
    Extract a metric series from records.
    metric_path can be a simple key like "reward_mean" or a nested path like "social_metrics.efficiency".
    """
    series: List[Tuple[int, float]] = []
    for record in records:
        episode = record.get("episode")
        if episode is None:
            continue
        if isinstance(episode, float):
            episode = int(episode)
        
        # Handle nested paths
        value = record
        for key in metric_path.split("."):
            if isinstance(value, dict):
                value = value.get(key)
            else:
                value = None
                break
        
        if isinstance(value, (int, float)):
            series.append((episode, float(value)))
    return sorted(series, key=lambda x: x[0])


def _align_series(
    all_series: List[List[Tuple[int, float]]]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Align multiple series to the same episode indices and compute mean, std, and SE.
    Returns: (episodes, means, stds, ses) where SE = std / sqrt(n)
    """
    if not all_series:
        return np.array([]), np.array([]), np.array([]), np.array([])
    
    # Get all unique episode numbers
    all_episodes = set()
    for series in all_series:
        all_episodes.update(ep for ep, _ in series)
    episodes = sorted(all_episodes)
    
    if not episodes:
        return np.array([]), np.array([]), np.ndarray([]), np.array([])
    
    # Interpolate/extract values for each series at each episode
    values_matrix = []
    for series in all_series:
        series_dict = dict(series)
        values = []
        for ep in episodes:
            if ep in series_dict:
                values.append(series_dict[ep])
            else:
                # Use linear interpolation if episode is missing
                # Find nearest episodes
                series_eps = [e for e, _ in series]
                if not series_eps:
                    values.append(np.nan)
                elif ep < series_eps[0]:
                    values.append(series_dict[series_eps[0]])
                elif ep > series_eps[-1]:
                    values.append(series_dict[series_eps[-1]])
                else:
                    # Interpolate - find the two closest episodes
                    for i in range(len(series) - 1):
                        ep1, val1 = series[i]
                        ep2, val2 = series[i + 1]
                        if ep1 <= ep <= ep2:
                            if ep2 == ep1:
                                values.append(val1)
                            else:
                                t = (ep - ep1) / (ep2 - ep1)
                                values.append(val1 + t * (val2 - val1))
                            break
                    else:
                        values.append(np.nan)
        values_matrix.append(values)
    
    # Compute mean, std, and SE across runs
    values_array = np.array(values_matrix)
    means = np.nanmean(values_array, axis=0)
    stds = np.nanstd(values_array, axis=0, ddof=1)  # Sample std
    
    # Compute SE = std / sqrt(n) where n is number of non-NaN values per episode
    n_valid = np.sum(~np.isnan(values_array), axis=0)
    ses = stds / np.sqrt(np.maximum(n_valid, 1))  # Avoid division by zero
    
    return np.array(episodes), means, stds, ses


def _plot_averaged_series(
    metric_name: str,
    episodes: np.ndarray,
    means: np.ndarray,
    stds: np.ndarray,
    title: str,
    ylabel: str,
    output_path: str,
    smooth_window: int = 1,
    show_title: bool = True,
) -> bool:
    """Plot a single averaged metric series with standard deviation."""
    if len(episodes) == 0:
        return False
    
    # Apply smoothing if requested
    if smooth_window > 1 and len(episodes) > 1:
        smoothed_means = []
        smoothed_stds = []
        for i in range(len(episodes)):
            start = max(0, i - smooth_window + 1)
            end = min(len(episodes), i + 1)
            window_means = means[start:end]
            window_stds = stds[start:end]
            smoothed_means.append(np.nanmean(window_means))
            # For smoothed std, we compute std of the window
            smoothed_stds.append(np.nanstd(window_means) if len(window_means) > 1 else 0.0)
        means = np.array(smoothed_means)
        stds = np.array(smoothed_stds)
    
    plt.figure(figsize=(10, 5))
    plt.errorbar(
        episodes,
        means,
        yerr=stds,
        label=metric_name,
        linewidth=2,
        capsize=3,
        capthick=1.5,
        elinewidth=1.5,
        alpha=0.7,
    )
    if show_title:
        plt.title(title)
    plt.xlabel("Episode")
    plt.ylabel(ylabel)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
    return True


def _thin_error_bars(episodes: np.ndarray, means: np.ndarray, stds: np.ndarray, step: int = 10) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Thin out error bars to show every Nth point."""
    if len(episodes) <= step:
        return episodes, means, stds
    indices = np.arange(0, len(episodes), step)
    return episodes[indices], means[indices], stds[indices]


def _plot_multiple_averaged_series(
    series_dict: Dict[str, Tuple],
    title: str,
    ylabel: str,
    output_path: str,
    smooth_window: int = 1,
    error_bar_step: int = 10,
    line_color: str = None,
    error_bar_color: str = None,
    show_title: bool = True,
    use_se: bool = False,
) -> bool:
    """Plot multiple averaged metric series on the same plot.
    
    Args:
        series_dict: Dict mapping metric name to tuple of (episodes, means, stds, ses) or (episodes, means, stds)
        use_se: If True, use standard error instead of std for shading
    """
    if not series_dict:
        return False
    
    # Use column width for single-column figures (3.5 inches) or double-column (7 inches)
    fig, ax = plt.subplots(figsize=(6, 4))
    
    # Get color cycle
    color_cycle = iter(PUBLICATION_COLORS)
    
    for idx, (metric_name, data_tuple) in enumerate(series_dict.items()):
        # Handle both 3-tuple and 4-tuple formats
        if len(data_tuple) == 4:
            episodes, means, stds, ses = data_tuple
            errors = ses if use_se else stds
        else:
            episodes, means, stds = data_tuple
            errors = stds
        
        if len(episodes) == 0:
            continue
        
        # Apply smoothing if requested
        if smooth_window > 1 and len(episodes) > 1:
            smoothed_means = []
            smoothed_errors = []
            for i in range(len(episodes)):
                start = max(0, i - smooth_window + 1)
                end = min(len(episodes), i + 1)
                window_means = means[start:end]
                smoothed_means.append(np.nanmean(window_means))
                smoothed_errors.append(np.nanstd(window_means) if len(window_means) > 1 else 0.0)
            means = np.array(smoothed_means)
            errors = np.array(smoothed_errors)
        
        # Get color for this series
        if line_color:
            color = line_color
        else:
            color = next(color_cycle)
        
        # Format metric name for display
        formatted_label = _format_label(metric_name)
        # Plot the line
        ax.plot(episodes, means, label=formatted_label, color=color, linewidth=2.0, zorder=2)
        
        # Use shaded area for all metrics (same as rewards)
        ax.fill_between(
            episodes,
            means - errors,
            means + errors,
            alpha=0.25,
            color=color,
            zorder=1,
        )
    
    ax.set_xlabel("Episode", fontweight='normal')
    ax.set_ylabel(ylabel, fontweight='normal')
    if show_title:
        ax.set_title(title, fontweight='bold', pad=10)
    ax.grid(True, linestyle='--', alpha=0.3, linewidth=0.5, zorder=0)
    ax.legend(loc='best', frameon=True, fancybox=True, shadow=False, framealpha=0.9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    return True


def _plot_social_subplots_averaged(
    series_dict: Dict[str, Tuple],
    title: str,
    output_path: str,
    smooth_window: int = 1,
    error_bar_step: int = 10,
    show_title: bool = True,
    use_se: bool = False,
) -> bool:
    """Plot social metrics as subplots with averaged values (shaded area).
    
    Args:
        series_dict: Dict mapping metric name to tuple of (episodes, means, stds, ses) or (episodes, means, stds)
        use_se: If True, use standard error instead of std for shading
    """
    ordered_names = [name for name in SOCIAL_ORDER if name in series_dict]
    if not ordered_names:
        ordered_names = [name for name in series_dict.keys() if series_dict[name][0].size > 0]
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
        data_tuple = series_dict[name]
        # Handle both 3-tuple and 4-tuple formats
        if len(data_tuple) == 4:
            episodes, means, stds, ses = data_tuple
            errors = ses if use_se else stds
        else:
            episodes, means, stds = data_tuple
            errors = stds
        
        if len(episodes) == 0:
            continue
        
        # Apply smoothing if requested
        if smooth_window > 1 and len(episodes) > 1:
            smoothed_means = []
            smoothed_errors = []
            for i in range(len(episodes)):
                start = max(0, i - smooth_window + 1)
                end = min(len(episodes), i + 1)
                window_means = means[start:end]
                smoothed_means.append(np.nanmean(window_means))
                smoothed_errors.append(np.nanstd(window_means) if len(window_means) > 1 else 0.0)
            means = np.array(smoothed_means)
            errors = np.array(smoothed_errors)
        
        # Plot line
        ax.plot(episodes, means, label=capitalize_metric(name), color=color, linewidth=2.0, zorder=2)
        
        # Plot shaded area (same as rewards)
        ax.fill_between(
            episodes,
            means - errors,
            means + errors,
            alpha=0.25,
            color=color,
            zorder=1,
        )
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


def _extract_per_agent_predicted_rewards(
    records: List[Dict[str, Any]]
) -> Dict[str, List[Tuple[int, float]]]:
    """
    Extract per-agent predicted reward series from records.
    Returns a dictionary mapping agent_id to list of (episode, reward) tuples.
    """
    agent_series: Dict[str, List[Tuple[int, float]]] = {}
    
    for record in records:
        episode = record.get("episode")
        if episode is None:
            continue
        if isinstance(episode, float):
            episode = int(episode)
        
        reward_pred_per_agent = record.get("reward_pred_per_agent")
        if isinstance(reward_pred_per_agent, dict):
            for agent_id, reward in reward_pred_per_agent.items():
                if isinstance(reward, (int, float)):
                    if agent_id not in agent_series:
                        agent_series[agent_id] = []
                    agent_series[agent_id].append((episode, float(reward)))
    
    # Sort each series by episode
    for agent_id in agent_series:
        agent_series[agent_id] = sorted(agent_series[agent_id], key=lambda x: x[0])
    
    return agent_series


def _plot_normalized_per_agent_predicted_rewards(
    all_agent_series: Dict[str, List[List[Tuple[int, float]]]],
    title: str,
    output_path: str,
    smooth_window: int = 1,
    show_title: bool = True,
) -> bool:
    """
    Plot normalized per-agent predicted rewards on the same graph.
    Each agent's series is normalized independently to [0, 1].
    """
    if not all_agent_series:
        return False
    
    # Align and compute averages for each agent
    agent_averaged: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for agent_id, series_list in all_agent_series.items():
        if series_list:
            episodes, means, stds, _ = _align_series(series_list)
            if len(episodes) > 0:
                agent_averaged[agent_id] = (episodes, means, stds)
    
    if not agent_averaged:
        return False
    
    # Normalize each agent's series independently to [0, 1]
    agent_averaged_normalized: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for agent_id, (episodes, means, stds) in agent_averaged.items():
        min_val = np.nanmin(means)
        max_val = np.nanmax(means)
        if max_val > min_val:
            means_norm = (means - min_val) / (max_val - min_val)
            stds_norm = stds / (max_val - min_val)
        else:
            means_norm = means
            stds_norm = stds
        agent_averaged_normalized[agent_id] = (episodes, means_norm, stds_norm)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(6, 4))
    
    # Sort agent IDs for consistent ordering
    sorted_agent_ids = sorted(agent_averaged_normalized.keys())
    color_cycle = iter(PUBLICATION_COLORS)
    
    for agent_id in sorted_agent_ids:
        episodes, means, stds = agent_averaged_normalized[agent_id]
        
        # Apply smoothing if requested
        if smooth_window > 1 and len(episodes) > 1:
            smoothed_means = []
            smoothed_stds = []
            for i in range(len(episodes)):
                start = max(0, i - smooth_window + 1)
                end = min(len(episodes), i + 1)
                window_means = means[start:end]
                smoothed_means.append(np.nanmean(window_means))
                smoothed_stds.append(np.nanstd(window_means) if len(window_means) > 1 else 0.0)
            means = np.array(smoothed_means)
            stds = np.array(smoothed_stds)
        
        color = next(color_cycle)
        label = agent_id.replace('agent-', 'Agent ')
        
        # Plot line
        ax.plot(episodes, means, label=label, color=color, linewidth=2.0, zorder=2)
        
        # Plot shaded area
        ax.fill_between(
            episodes,
            means - stds,
            means + stds,
            alpha=0.25,
            color=color,
            zorder=1,
        )
    
    ax.set_xlabel("Episode", fontweight='normal')
    ax.set_ylabel("Normalized Predicted Reward", fontweight='normal')
    if show_title:
        ax.set_title(title, fontweight='bold', pad=10)
    ax.grid(True, linestyle='--', alpha=0.3, linewidth=0.5, zorder=0)
    ax.legend(loc='best', frameon=True, fancybox=True, shadow=False, framealpha=0.9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    return True


def _load_run_context(run_dir: str) -> Tuple[str, str]:
    """Load algorithm name and reward model phi from config."""
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


def plot_multiple_runs(
    run_dirs: List[str],
    output_dir: str,
    smooth_window: int = 1,
    normalize: bool = False,
    show_title: bool = True,
) -> Tuple[bool, bool, bool]:
    """
    Plot averaged metrics across multiple runs with standard deviation.
    
    Args:
        run_dirs: List of run directory paths
        output_dir: Directory to save plots
        smooth_window: Moving average window size
        normalize: Whether to normalize metrics to [0, 1]
        show_title: Whether to show titles on plots
    
    Returns:
        Tuple of (rewards_plotted, social_plotted, agent_pred_plotted)
    """
    if not run_dirs:
        raise ValueError("No run directories provided")
    
    # Load metrics from all runs
    all_reward_series: Dict[str, List[List[Tuple[int, float]]]] = {
        key: [] for key in REWARD_KEYS
    }
    all_social_series: Dict[str, List[List[Tuple[int, float]]]] = {}
    all_agent_pred_rewards: Dict[str, List[List[Tuple[int, float]]]] = {}
    
    algo_name = "unknown"
    rm_phi = ""
    
    for run_dir in run_dirs:
        metrics_path = os.path.join(run_dir, "metrics.jsonl")
        if not os.path.isfile(metrics_path):
            print(f"Warning: metrics.jsonl not found in {run_dir}, skipping")
            continue
        
        # Load context from first valid run
        if algo_name == "unknown":
            algo_name, rm_phi = _load_run_context(run_dir)
        
        records = _load_metrics(metrics_path)
        
        # Extract reward metrics
        for key in REWARD_KEYS:
            series = _extract_metric_series(records, key)
            if series:
                all_reward_series[key].append(series)
        
        # Extract social metrics
        for name in SOCIAL_ORDER:
            series = _extract_metric_series(records, f"social_metrics.{name}")
            if series:
                if name not in all_social_series:
                    all_social_series[name] = []
                all_social_series[name].append(series)
        
        # Extract per-agent predicted rewards
        agent_series = _extract_per_agent_predicted_rewards(records)
        for agent_id, series in agent_series.items():
            if agent_id not in all_agent_pred_rewards:
                all_agent_pred_rewards[agent_id] = []
            all_agent_pred_rewards[agent_id].append(series)
    
    # Compute averages, stds, and SEs
    rewards_averaged: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    for key, series_list in all_reward_series.items():
        if series_list:
            episodes, means, stds, ses = _align_series(series_list)
            if len(episodes) > 0:
                rewards_averaged[key] = (episodes, means, stds, ses)
    
    social_averaged: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    for name, series_list in all_social_series.items():
        if series_list:
            episodes, means, stds, ses = _align_series(series_list)
            if len(episodes) > 0:
                social_averaged[name] = (episodes, means, stds, ses)
    
    # Normalize if requested
    if normalize:
        # Normalize each metric to [0, 1]
        for key in rewards_averaged:
            episodes, means, stds, ses = rewards_averaged[key]
            min_val = np.nanmin(means)
            max_val = np.nanmax(means)
            if max_val > min_val:
                means = (means - min_val) / (max_val - min_val)
                stds = stds / (max_val - min_val)
                ses = ses / (max_val - min_val)
            rewards_averaged[key] = (episodes, means, stds, ses)
        
        for name in social_averaged:
            episodes, means, stds, ses = social_averaged[name]
            min_val = np.nanmin(means)
            max_val = np.nanmax(means)
            if max_val > min_val:
                means = (means - min_val) / (max_val - min_val)
                stds = stds / (max_val - min_val)
                ses = ses / (max_val - min_val)
            social_averaged[name] = (episodes, means, stds, ses)
    
    # Create output directory and SE subdirectory
    os.makedirs(output_dir, exist_ok=True)
    se_output_dir = os.path.join(output_dir, "with_se")
    os.makedirs(se_output_dir, exist_ok=True)
    
    # Generate titles (more concise for publication)
    algo_display = algo_name.upper() if algo_name != "unknown" else "Unknown"
    if rm_phi:
        # Format phi value for display (convert _x_ to \times, remove other underscores)
        phi_display = _format_label(rm_phi)
        reward_title = f"Average Rewards ({algo_display}, {len(run_dirs)} runs, φ={phi_display})"
    else:
        reward_title = f"Average Rewards ({algo_display}, {len(run_dirs)} runs)"
    social_title = f"Average Social Metrics ({algo_display}, {len(run_dirs)} runs)"
    
    if normalize:
        reward_title = f"{reward_title} (Normalized)"
        social_title = f"{social_title} (Normalized)"
        reward_ylabel = "Normalized Value"
        social_ylabel = "Normalized Value"
    else:
        reward_ylabel = "Reward"
        social_ylabel = "Metric Value"
    
    # Plot rewards (with STD)
    rewards_path = os.path.join(output_dir, "rewards_averaged.png")
    rewards_plotted = _plot_multiple_averaged_series(
        rewards_averaged,
        title=reward_title,
        ylabel=reward_ylabel,
        output_path=rewards_path,
        smooth_window=smooth_window,
        error_bar_step=10,
        show_title=show_title,
        use_se=False,
    )
    
    # Plot rewards (with SE)
    rewards_se_path = os.path.join(se_output_dir, "rewards_averaged.png")
    _plot_multiple_averaged_series(
        rewards_averaged,
        title=reward_title,
        ylabel=reward_ylabel,
        output_path=rewards_se_path,
        smooth_window=smooth_window,
        error_bar_step=10,
        show_title=show_title,
        use_se=True,
    )
    
    # Plot social metrics (with STD)
    social_path = os.path.join(output_dir, "social_metrics_averaged.png")
    if normalize:
        social_plotted = _plot_multiple_averaged_series(
            social_averaged,
            title=social_title,
            ylabel=social_ylabel,
            output_path=social_path,
            smooth_window=smooth_window,
            error_bar_step=10,
            line_color=None,  # Use default blue color
            error_bar_color='black',
            show_title=show_title,
            use_se=False,
        )
        # Also plot with SE
        social_se_path = os.path.join(se_output_dir, "social_metrics_averaged.png")
        _plot_multiple_averaged_series(
            social_averaged,
            title=social_title,
            ylabel=social_ylabel,
            output_path=social_se_path,
            smooth_window=smooth_window,
            error_bar_step=10,
            line_color=None,
            error_bar_color='black',
            show_title=show_title,
            use_se=True,
        )
    else:
        # Filter to only ordered social metrics
        ordered_social = {
            name: social_averaged[name]
            for name in SOCIAL_ORDER
            if name in social_averaged
        }
        if not ordered_social:
            ordered_social = social_averaged
        social_plotted = _plot_social_subplots_averaged(
            ordered_social,
            title=social_title,
            output_path=social_path,
            smooth_window=smooth_window,
            error_bar_step=10,
            show_title=show_title,
            use_se=False,
        )
        # Also plot with SE
        social_se_path = os.path.join(se_output_dir, "social_metrics_averaged.png")
        _plot_social_subplots_averaged(
            ordered_social,
            title=social_title,
            output_path=social_se_path,
            smooth_window=smooth_window,
            error_bar_step=10,
            show_title=show_title,
            use_se=True,
        )
    
    # Plot normalized per-agent predicted rewards
    agent_pred_title = f"Normalized Predicted Rewards per Agent ({algo_display}, {len(run_dirs)} runs)"
    if rm_phi:
        # Format phi value for display (convert _x_ to \times, remove other underscores)
        phi_display = _format_label(rm_phi)
        agent_pred_title = f"Normalized Predicted Rewards per Agent ({algo_display}, {len(run_dirs)} runs, φ={phi_display})"
    agent_pred_path = os.path.join(output_dir, "agent_predicted_rewards_normalized.png")
    agent_pred_plotted = _plot_normalized_per_agent_predicted_rewards(
        all_agent_pred_rewards,
        title=agent_pred_title,
        output_path=agent_pred_path,
        smooth_window=smooth_window,
        show_title=show_title,
    )
    
    return rewards_plotted, social_plotted, agent_pred_plotted


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="commons-game plot runs",
        description="Generate averaged plots with standard deviation across multiple runs.",
    )
    parser.add_argument(
        "run_dirs",
        nargs="+",
        help="Paths to run directories containing metrics.jsonl files.",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="plots_averaged",
        help="Output directory for plots (default: plots_averaged)",
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
    return parser


def run(args: argparse.Namespace) -> int:
    show_title = not args.no_title

    rewards_plotted, social_plotted, agent_pred_plotted = plot_multiple_runs(
        args.run_dirs,
        output_dir=args.output_dir,
        smooth_window=args.smooth,
        normalize=args.normalize,
        show_title=show_title,
    )
    
    if rewards_plotted:
        print(f"Saved averaged rewards plot to {os.path.join(args.output_dir, 'rewards_averaged.png')}")
    else:
        print("No reward metrics found to plot.")
    
    if social_plotted:
        print(f"Saved averaged social metrics plot to {os.path.join(args.output_dir, 'social_metrics_averaged.png')}")
    else:
        print("No social metrics found to plot.")
    
    if agent_pred_plotted:
        print(f"Saved normalized per-agent predicted rewards plot to {os.path.join(args.output_dir, 'agent_predicted_rewards_normalized.png')}")
    else:
        print("No per-agent predicted reward metrics found to plot.")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(run(build_parser().parse_args()))

