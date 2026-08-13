"""
Compare metrics between different reward model modes (narrow_view vs input_aggregation).
Generates side-by-side and overlaid comparison plots.
"""

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
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    'mathtext.fontset': 'stix',
})

# Colorblind-friendly colors for the two modes
MODE_COLORS = {
    'narrow_view': '#1f77b4',       # blue
    'input_aggregation': '#ff7f0e',  # orange
}

MODE_LABELS = {
    'narrow_view': 'Narrow View',
    'input_aggregation': 'Input Aggregation',
}

SOCIAL_ORDER = ("efficiency", "equality", "sustainability", "peace")


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
    """Extract a metric series from records."""
    series: List[Tuple[int, float]] = []
    for record in records:
        episode = record.get("episode")
        if episode is None:
            continue
        if isinstance(episode, float):
            episode = int(episode)
        
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
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Align multiple series and compute mean and std."""
    if not all_series:
        return np.array([]), np.array([]), np.array([])
    
    all_episodes = set()
    for series in all_series:
        all_episodes.update(ep for ep, _ in series)
    episodes = sorted(all_episodes)
    
    if not episodes:
        return np.array([]), np.array([]), np.array([])
    
    values_matrix = []
    for series in all_series:
        series_dict = dict(series)
        values = []
        for ep in episodes:
            if ep in series_dict:
                values.append(series_dict[ep])
            else:
                series_eps = [e for e, _ in series]
                if not series_eps:
                    values.append(np.nan)
                elif ep < series_eps[0]:
                    values.append(series_dict[series_eps[0]])
                elif ep > series_eps[-1]:
                    values.append(series_dict[series_eps[-1]])
                else:
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
    
    values_array = np.array(values_matrix)
    means = np.nanmean(values_array, axis=0)
    stds = np.nanstd(values_array, axis=0, ddof=1)
    
    return np.array(episodes), means, stds


def _load_runs(run_dirs: List[str], metric_paths: List[str]) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Load and average metrics from multiple runs."""
    all_series: Dict[str, List[List[Tuple[int, float]]]] = {path: [] for path in metric_paths}
    
    for run_dir in run_dirs:
        metrics_path = os.path.join(run_dir, "metrics.jsonl")
        if not os.path.isfile(metrics_path):
            print(f"Warning: metrics.jsonl not found in {run_dir}, skipping")
            continue
        
        records = _load_metrics(metrics_path)
        for path in metric_paths:
            series = _extract_metric_series(records, path)
            if series:
                all_series[path].append(series)
    
    result = {}
    for path, series_list in all_series.items():
        if series_list:
            episodes, means, stds = _align_series(series_list)
            if len(episodes) > 0:
                result[path] = (episodes, means, stds)
    
    return result


def plot_comparison(
    narrow_view_dirs: List[str],
    input_agg_dirs: List[str],
    output_dir: str,
) -> None:
    """Generate comparison plots between two reward model modes."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Define metrics to compare
    reward_metrics = ["reward_mean", "reward_pred_mean"]
    social_metrics = [f"social_metrics.{name}" for name in SOCIAL_ORDER]
    
    # Load data for both modes
    narrow_data = _load_runs(narrow_view_dirs, reward_metrics + social_metrics)
    input_agg_data = _load_runs(input_agg_dirs, reward_metrics + social_metrics)
    
    # 1. Plot reward comparison (overlaid)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    for idx, metric in enumerate(reward_metrics):
        ax = axes[idx]
        metric_label = "Mean Environment Reward" if metric == "reward_mean" else "Mean Predicted Reward"
        
        for mode, data, color in [
            ('narrow_view', narrow_data, MODE_COLORS['narrow_view']),
            ('input_aggregation', input_agg_data, MODE_COLORS['input_aggregation']),
        ]:
            if metric in data:
                episodes, means, stds = data[metric]
                ax.plot(episodes, means, label=MODE_LABELS[mode], color=color, linewidth=2.0)
                ax.fill_between(episodes, means - stds, means + stds, alpha=0.25, color=color)
        
        ax.set_xlabel("Episode")
        ax.set_ylabel(metric_label)
        ax.set_title(metric_label)
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.legend(loc='best', frameon=True, framealpha=0.9)
    
    fig.suptitle(f"Reward Comparison: Narrow View ({len(narrow_view_dirs)} runs) vs Input Aggregation ({len(input_agg_dirs)} runs)", fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "reward_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {os.path.join(output_dir, 'reward_comparison.png')}")
    
    # 2. Plot social metrics comparison (2x2 grid)
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes_flat = axes.flatten()
    
    for idx, name in enumerate(SOCIAL_ORDER):
        ax = axes_flat[idx]
        metric_path = f"social_metrics.{name}"
        
        for mode, data, color in [
            ('narrow_view', narrow_data, MODE_COLORS['narrow_view']),
            ('input_aggregation', input_agg_data, MODE_COLORS['input_aggregation']),
        ]:
            if metric_path in data:
                episodes, means, stds = data[metric_path]
                ax.plot(episodes, means, label=MODE_LABELS[mode], color=color, linewidth=2.0)
                ax.fill_between(episodes, means - stds, means + stds, alpha=0.25, color=color)
        
        ax.set_xlabel("Episode")
        ax.set_ylabel(name.capitalize())
        ax.set_title(name.capitalize())
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.legend(loc='best', frameon=True, framealpha=0.9)
    
    fig.suptitle(f"Social Metrics Comparison: Narrow View ({len(narrow_view_dirs)} runs) vs Input Aggregation ({len(input_agg_dirs)} runs)", fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "social_metrics_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {os.path.join(output_dir, 'social_metrics_comparison.png')}")
    
    # 3. Summary bar chart comparing final performance
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Get final values (last 10 episodes average)
    def get_final_avg(data: Dict, metric: str, n_episodes: int = 10) -> Tuple[float, float]:
        if metric not in data:
            return 0, 0
        episodes, means, stds = data[metric]
        if len(means) < n_episodes:
            return np.mean(means), np.mean(stds)
        return np.mean(means[-n_episodes:]), np.mean(stds[-n_episodes:])
    
    # Rewards bar chart
    ax = axes[0]
    x = np.arange(2)
    width = 0.35
    
    narrow_reward_mean, narrow_reward_std = get_final_avg(narrow_data, "reward_mean")
    input_reward_mean, input_reward_std = get_final_avg(input_agg_data, "reward_mean")
    narrow_pred_mean, narrow_pred_std = get_final_avg(narrow_data, "reward_pred_mean")
    input_pred_mean, input_pred_std = get_final_avg(input_agg_data, "reward_pred_mean")
    
    bars1 = ax.bar(x - width/2, [narrow_reward_mean, narrow_pred_mean], width, 
                   yerr=[narrow_reward_std, narrow_pred_std], label='Narrow View',
                   color=MODE_COLORS['narrow_view'], capsize=5)
    bars2 = ax.bar(x + width/2, [input_reward_mean, input_pred_mean], width,
                   yerr=[input_reward_std, input_pred_std], label='Input Aggregation',
                   color=MODE_COLORS['input_aggregation'], capsize=5)
    
    ax.set_ylabel('Mean Reward (Final 10 Episodes)')
    ax.set_title('Final Reward Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(['Environment Reward', 'Predicted Reward'])
    ax.legend()
    ax.grid(True, axis='y', linestyle='--', alpha=0.3)
    
    # Social metrics bar chart
    ax = axes[1]
    x = np.arange(4)
    width = 0.35
    
    narrow_social = [get_final_avg(narrow_data, f"social_metrics.{name}") for name in SOCIAL_ORDER]
    input_social = [get_final_avg(input_agg_data, f"social_metrics.{name}") for name in SOCIAL_ORDER]
    
    bars1 = ax.bar(x - width/2, [m for m, s in narrow_social], width,
                   yerr=[s for m, s in narrow_social], label='Narrow View',
                   color=MODE_COLORS['narrow_view'], capsize=5)
    bars2 = ax.bar(x + width/2, [m for m, s in input_social], width,
                   yerr=[s for m, s in input_social], label='Input Aggregation',
                   color=MODE_COLORS['input_aggregation'], capsize=5)
    
    ax.set_ylabel('Metric Value (Final 10 Episodes)')
    ax.set_title('Final Social Metrics Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels([name.capitalize() for name in SOCIAL_ORDER])
    ax.legend()
    ax.grid(True, axis='y', linestyle='--', alpha=0.3)
    
    fig.suptitle('Final Performance Summary', fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "final_performance_summary.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {os.path.join(output_dir, 'final_performance_summary.png')}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="commons-game compare-modes",
        description="Compare reward model modes",
    )
    parser.add_argument("--narrow-view", nargs="+", required=True, help="Narrow view run directories")
    parser.add_argument("--input-aggregation", nargs="+", required=True, help="Input aggregation run directories")
    parser.add_argument("--output-dir", "-o", default="logs/comparisons/mode_comparison", help="Output directory")
    return parser


def run(args: argparse.Namespace) -> int:
    plot_comparison(args.narrow_view, args.input_aggregation, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(run(build_parser().parse_args()))






