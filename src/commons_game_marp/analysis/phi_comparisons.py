"""
Plot phi comparisons and social metrics comparisons between narrow view and input aggregation.

A. Phi comparisons (efficiency vs efficiency_x_<social>):
   1. Efficiency metric: Compare NV/IA with efficiency vs efficiency_x_peace
   2. Efficiency metric: Compare NV/IA with efficiency vs efficiency_x_equality  
   3. Efficiency metric: Compare NV/IA with efficiency vs efficiency_x_sustainability

B. Social metrics gallery:
   1. 2x2 gallery of all social metrics comparing NV to IA

All graphs use Standard Error (SE) for shading.
"""

import argparse
import json
import os
from typing import Any, Dict, List, Tuple, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .sessions import (
    IPPO_SESSIONS,
    MAPPO_SESSIONS,
    get_sessions_for_algorithm,
    parse_session_name,
    get_session_by_approach_and_target,
)

# Publication-quality settings
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
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
    'text.usetex': False,
    'mathtext.fontset': 'stix',
})

SOCIAL_ORDER = ("efficiency", "equality", "sustainability", "peace")


def _smooth_series(values: np.ndarray, window: int) -> np.ndarray:
    """Apply moving average smoothing to a series."""
    if window <= 1 or len(values) <= 1:
        return values
    
    smoothed = np.zeros_like(values)
    for i in range(len(values)):
        start = max(0, i - window + 1)
        end = i + 1
        smoothed[i] = np.nanmean(values[start:end])
    return smoothed

# Colorblind-friendly colors for 4 lines - distinct colors for each line
PHI_COMPARISON_COLORS = {
    'nv_base': '#1f77b4',        # blue - NV efficiency
    'nv_combined': '#2ca02c',    # green - NV efficiency_x_<social>
    'ia_base': '#ff7f0e',        # orange - IA efficiency
    'ia_combined': '#d62728',    # red - IA efficiency_x_<social>
}

# Colors for NV vs IA comparison (2 lines)
MODE_COLORS = {
    'narrow_view': '#1f77b4',       # blue
    'input_aggregation': '#ff7f0e',  # orange
}

MODE_LABELS = {
    'narrow_view': 'Narrow View',
    'input_aggregation': 'Input Aggregation',
}


def _format_label(label: str) -> str:
    """
    Format label for display:
    - Convert 'efficiency_x_peace' to 'Efficiency × Peace'
    - Replace underscores with spaces and capitalize words
    """
    if '_x_' in label:
        parts = label.split('_x_')
        formatted_parts = [p.replace('_', ' ').title() for p in parts]
        return r' $\times$ '.join(formatted_parts)
    else:
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
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Align multiple series to the same episode indices and compute mean, std, and SE.
    Returns: (episodes, means, stds, ses) where SE = std / sqrt(n)
    """
    if not all_series:
        return np.array([]), np.array([]), np.array([]), np.array([])
    
    all_episodes = set()
    for series in all_series:
        all_episodes.update(ep for ep, _ in series)
    episodes = sorted(all_episodes)
    
    if not episodes:
        return np.array([]), np.array([]), np.array([]), np.array([])
    
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
    
    n_valid = np.sum(~np.isnan(values_array), axis=0)
    ses = stds / np.sqrt(np.maximum(n_valid, 1))
    
    return np.array(episodes), means, stds, ses


def load_session_metric(
    sessions: Dict[str, List[str]],
    session_name: str,
    metric_path: str,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Load and average a metric across all runs in a session.
    
    Returns: (episodes, means, stds, ses) or None if session not found.
    """
    if session_name not in sessions:
        return None
    
    run_dirs = sessions[session_name]
    all_series: List[List[Tuple[int, float]]] = []
    
    for run_dir in run_dirs:
        metrics_path = os.path.join(run_dir, "metrics.jsonl")
        if not os.path.isfile(metrics_path):
            print(f"Warning: metrics.jsonl not found in {run_dir}, skipping")
            continue
        
        records = _load_metrics(metrics_path)
        series = _extract_metric_series(records, metric_path)
        if series:
            all_series.append(series)
    
    if not all_series:
        return None
    
    return _align_series(all_series)


def plot_phi_comparison(
    sessions: Dict[str, List[str]],
    social_target: str,
    output_path: str,
    show_title: bool = True,
    algorithm: str = "ippo",
    smooth_window: int = 1,
) -> bool:
    """
    Plot the social_target metric comparing NV and IA with efficiency vs efficiency_x_<social_target>.
    
    Args:
        sessions: Sessions dictionary
        social_target: The social metric to compare (e.g., 'peace', 'equality', 'sustainability')
        output_path: Path to save the plot
        show_title: Whether to show plot title
        algorithm: Algorithm name for title
        smooth_window: Moving average window for smoothing means (SE is not affected)
    
    Returns:
        True if plot was created successfully
    """
    # Define the 4 session configurations to compare (all solid lines, distinct colors)
    configs = [
        ("narrow_view", "efficiency", "NV - Efficiency", PHI_COMPARISON_COLORS['nv_base']),
        ("narrow_view", f"efficiency_x_{social_target}", f"NV - Efficiency × {social_target.title()}", PHI_COMPARISON_COLORS['nv_combined']),
        ("input_aggregation", "efficiency", "IA - Efficiency", PHI_COMPARISON_COLORS['ia_base']),
        ("input_aggregation", f"efficiency_x_{social_target}", f"IA - Efficiency × {social_target.title()}", PHI_COMPARISON_COLORS['ia_combined']),
    ]
    
    # Metric to plot is the social_target (not efficiency)
    metric_path = f"social_metrics.{social_target}"
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Collect all data first to find the minimum episode range
    all_data = []
    for approach, phi, label, color in configs:
        session_name = get_session_by_approach_and_target(sessions, approach, phi)
        if session_name is None:
            print(f"Warning: Session not found for approach={approach}, target={phi}")
            all_data.append(None)
            continue
        
        data = load_session_metric(sessions, session_name, metric_path)
        if data is None:
            print(f"Warning: No data for session {session_name}")
            all_data.append(None)
            continue
        
        all_data.append(data)
    
    # Find the minimum max episode across all series (to crop x-axis)
    max_episodes = []
    for data in all_data:
        if data is not None:
            episodes, means, stds, ses = data
            if len(episodes) > 0:
                max_episodes.append(episodes[-1])
    
    if not max_episodes:
        plt.close()
        return False
    
    min_max_episode = min(max_episodes)
    
    # Plot each series, cropping to the minimum max episode
    has_data = False
    for (approach, phi, label, color), data in zip(configs, all_data):
        if data is None:
            continue
        
        episodes, means, stds, ses = data
        if len(episodes) == 0:
            continue
        
        # Crop to min_max_episode
        mask = episodes <= min_max_episode
        episodes = episodes[mask]
        means = means[mask]
        ses = ses[mask]
        
        if len(episodes) == 0:
            continue
        
        # Apply smoothing to means only (SE stays unsmoothed)
        smoothed_means = _smooth_series(means, smooth_window)
        
        has_data = True
        ax.plot(episodes, smoothed_means, label=label, color=color, linestyle='-', linewidth=2.0, zorder=2)
        ax.fill_between(episodes, smoothed_means - ses, smoothed_means + ses, alpha=0.25, color=color, zorder=1)
    
    if not has_data:
        plt.close()
        return False
    
    ax.set_xlabel("Episode")
    ax.set_ylabel(social_target.capitalize())
    if show_title:
        ax.set_title(f"{social_target.capitalize()} Comparison: φ=Efficiency vs φ=Efficiency×{social_target.title()} ({algorithm.upper()})", 
                     fontweight='bold', pad=10)
    ax.grid(True, linestyle='--', alpha=0.3, linewidth=0.5, zorder=0)
    ax.legend(loc='best', frameon=True, fancybox=True, shadow=False, framealpha=0.9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    return True


def plot_social_metrics_gallery(
    sessions: Dict[str, List[str]],
    phi: str,
    output_path: str,
    show_title: bool = True,
    algorithm: str = "ippo",
    smooth_window: int = 1,
) -> bool:
    """
    Plot a 2x2 gallery of all social metrics comparing NV to IA for a given phi.
    
    Args:
        sessions: Sessions dictionary
        phi: The phi value to use (e.g., 'efficiency', 'efficiency_x_peace')
        output_path: Path to save the plot
        show_title: Whether to show plot title
        algorithm: Algorithm name for title
        smooth_window: Moving average window for smoothing means (SE is not affected)
    
    Returns:
        True if plot was created successfully
    """
    # Get session names for both approaches
    nv_session = get_session_by_approach_and_target(sessions, "narrow_view", phi)
    ia_session = get_session_by_approach_and_target(sessions, "input_aggregation", phi)
    
    if nv_session is None and ia_session is None:
        print(f"Warning: No sessions found for phi={phi}")
        return False
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes_flat = axes.flatten()
    
    has_data = False
    for idx, metric_name in enumerate(SOCIAL_ORDER):
        ax = axes_flat[idx]
        metric_path = f"social_metrics.{metric_name}"
        
        # Plot NV
        if nv_session:
            nv_data = load_session_metric(sessions, nv_session, metric_path)
            if nv_data is not None:
                episodes, means, stds, ses = nv_data
                if len(episodes) > 0:
                    has_data = True
                    smoothed_means = _smooth_series(means, smooth_window)
                    ax.plot(episodes, smoothed_means, label=MODE_LABELS['narrow_view'], 
                           color=MODE_COLORS['narrow_view'], linewidth=2.0, zorder=2)
                    ax.fill_between(episodes, smoothed_means - ses, smoothed_means + ses, 
                                   alpha=0.25, color=MODE_COLORS['narrow_view'], zorder=1)
        
        # Plot IA
        if ia_session:
            ia_data = load_session_metric(sessions, ia_session, metric_path)
            if ia_data is not None:
                episodes, means, stds, ses = ia_data
                if len(episodes) > 0:
                    has_data = True
                    smoothed_means = _smooth_series(means, smooth_window)
                    ax.plot(episodes, smoothed_means, label=MODE_LABELS['input_aggregation'], 
                           color=MODE_COLORS['input_aggregation'], linewidth=2.0, zorder=2)
                    ax.fill_between(episodes, smoothed_means - ses, smoothed_means + ses, 
                                   alpha=0.25, color=MODE_COLORS['input_aggregation'], zorder=1)
        
        ax.set_xlabel("Episode")
        ax.set_ylabel(metric_name.capitalize())
        if show_title:
            ax.set_title(metric_name.capitalize(), fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.3, linewidth=0.5, zorder=0)
        ax.legend(loc='best', frameon=True, fancybox=True, shadow=False, framealpha=0.9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    if not has_data:
        plt.close()
        return False
    
    if show_title:
        phi_display = _format_label(phi)
        fig.suptitle(f"Social Metrics: NV vs IA (φ={phi_display}, {algorithm.upper()})", 
                     fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    return True


def plot_all_social_metrics_gallery_nv_vs_ia(
    sessions: Dict[str, List[str]],
    output_path: str,
    show_title: bool = True,
    algorithm: str = "ippo",
    smooth_window: int = 1,
) -> bool:
    """
    Plot a 2x2 gallery of all social metrics comparing NV to IA, 
    averaged across all phi values for each approach.
    
    Args:
        sessions: Sessions dictionary
        output_path: Path to save the plot
        show_title: Whether to show plot title
        algorithm: Algorithm name for title
        smooth_window: Moving average window for smoothing means (SE is not affected)
    
    Returns:
        True if plot was created successfully
    """
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes_flat = axes.flatten()
    
    has_data = False
    
    for idx, metric_name in enumerate(SOCIAL_ORDER):
        ax = axes_flat[idx]
        metric_path = f"social_metrics.{metric_name}"
        
        # Collect all series for each approach
        for approach, color in [("narrow_view", MODE_COLORS['narrow_view']), 
                                 ("input_aggregation", MODE_COLORS['input_aggregation'])]:
            all_series = []
            
            # Find all sessions for this approach
            for session_name in sessions.keys():
                sess_approach, _ = parse_session_name(session_name)
                if sess_approach != approach:
                    continue
                
                run_dirs = sessions[session_name]
                for run_dir in run_dirs:
                    metrics_path = os.path.join(run_dir, "metrics.jsonl")
                    if not os.path.isfile(metrics_path):
                        continue
                    records = _load_metrics(metrics_path)
                    series = _extract_metric_series(records, metric_path)
                    if series:
                        all_series.append(series)
            
            if not all_series:
                continue
            
            episodes, means, stds, ses = _align_series(all_series)
            if len(episodes) == 0:
                continue
            
            has_data = True
            smoothed_means = _smooth_series(means, smooth_window)
            label = MODE_LABELS[approach]
            ax.plot(episodes, smoothed_means, label=label, color=color, linewidth=2.0, zorder=2)
            ax.fill_between(episodes, smoothed_means - ses, smoothed_means + ses, alpha=0.25, color=color, zorder=1)
        
        ax.set_xlabel("Episode")
        ax.set_ylabel(metric_name.capitalize())
        if show_title:
            ax.set_title(metric_name.capitalize(), fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.3, linewidth=0.5, zorder=0)
        ax.legend(loc='best', frameon=True, fancybox=True, shadow=False, framealpha=0.9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    if not has_data:
        plt.close()
        return False
    
    if show_title:
        fig.suptitle(f"Social Metrics: Narrow View vs Input Aggregation ({algorithm.upper()})", 
                     fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    return True


def _generate_all_plots(
    sessions: Dict[str, List[str]],
    output_dir: str,
    show_title: bool,
    algorithm: str,
    smooth_window: int = 1,
) -> None:
    """Generate all plots with the given smoothing window."""
    os.makedirs(output_dir, exist_ok=True)
    
    # A. Phi comparison plots (efficiency vs efficiency_x_<social>)
    print("=" * 60)
    print("A. Phi Comparison Plots")
    print("=" * 60)
    
    social_targets = ["peace", "equality", "sustainability"]
    for target in social_targets:
        output_path = os.path.join(output_dir, f"phi_comparison_efficiency_vs_{target}.png")
        success = plot_phi_comparison(
            sessions, 
            target, 
            output_path, 
            show_title=show_title,
            algorithm=algorithm,
            smooth_window=smooth_window,
        )
        if success:
            print(f"  [OK] Saved: {output_path}")
        else:
            print(f"  [FAIL] Failed: efficiency vs efficiency_x_{target}")
    
    print()
    
    # B. Social metrics gallery (NV vs IA)
    print("=" * 60)
    print("B. Social Metrics Gallery (2x2, NV vs IA)")
    print("=" * 60)
    
    # B.1 Overall comparison (averaged across all phi values)
    output_path = os.path.join(output_dir, "social_metrics_gallery_nv_vs_ia_overall.png")
    success = plot_all_social_metrics_gallery_nv_vs_ia(
        sessions,
        output_path,
        show_title=show_title,
        algorithm=algorithm,
        smooth_window=smooth_window,
    )
    if success:
        print(f"  [OK] Saved: {output_path}")
    else:
        print(f"  [FAIL] Failed: Overall NV vs IA gallery")
    
    # B.2 Per-phi galleries
    phi_values = ["efficiency", "efficiency_x_peace", "efficiency_x_equality", "efficiency_x_sustainability"]
    for phi in phi_values:
        output_path = os.path.join(output_dir, f"social_metrics_gallery_{phi}.png")
        success = plot_social_metrics_gallery(
            sessions,
            phi,
            output_path,
            show_title=show_title,
            algorithm=algorithm,
            smooth_window=smooth_window,
        )
        if success:
            print(f"  [OK] Saved: {output_path}")
        else:
            print(f"  [FAIL] Failed: {phi} gallery")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="commons-game plot phi",
        description="Generate phi comparison and social metrics gallery plots.",
    )
    parser.add_argument(
        "--algorithm",
        "-a",
        type=str,
        default="ippo",
        choices=["ippo", "mappo"],
        help="Algorithm to use (default: ippo)",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default=None,
        help="Output directory for plots (default: logs/<algorithm>/comparisons/phi_comparisons)",
    )
    parser.add_argument(
        "--no-title",
        action="store_true",
        help="Hide titles from all plots (useful for publication figures).",
    )
    parser.add_argument(
        "--smooth",
        type=int,
        default=10,
        help="Smoothing window size for the smoothed version (default: 10). SE is not affected by smoothing.",
    )
    return parser


def run(args: argparse.Namespace) -> int:
    # Get sessions for the specified algorithm
    sessions = get_sessions_for_algorithm(args.algorithm)
    
    # Set default output directory
    if args.output_dir is None:
        output_dir = f"logs/{args.algorithm}/comparisons/phi_comparisons"
    else:
        output_dir = args.output_dir
    
    show_title = not args.no_title
    
    print(f"Generating plots for algorithm: {args.algorithm.upper()}")
    print(f"Output directory: {output_dir}")
    print()
    
    # Generate unsmoothed plots
    print("#" * 60)
    print("# UNSMOOTHED PLOTS")
    print("#" * 60)
    print()
    _generate_all_plots(
        sessions,
        output_dir,
        show_title,
        args.algorithm,
        smooth_window=1,
    )
    
    print()
    
    # Generate smoothed plots in subdirectory
    smoothed_dir = os.path.join(output_dir, f"smoothed_{args.smooth}")
    print("#" * 60)
    print(f"# SMOOTHED PLOTS (window={args.smooth})")
    print("#" * 60)
    print()
    _generate_all_plots(
        sessions,
        smoothed_dir,
        show_title,
        args.algorithm,
        smooth_window=args.smooth,
    )
    
    print()
    print("Done!")
    return 0


if __name__ == "__main__":
    raise SystemExit(run(build_parser().parse_args()))

