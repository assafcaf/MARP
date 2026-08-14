"""Metrics calculation module for agent episode tracking.

The per-step measurements now live in `env/step_metrics.py`, so the environment
can compute them itself and hand them back on the info dict -- which is what
lets the environments run in worker processes. They are re-exported here
unchanged, so every existing import path still resolves.
"""

import numpy as np
from typing import Tuple, Dict, Any

from ..env.step_metrics import (
    APPLE_RADIUS,
    check_ate_last_apple_in_cluster,
    count_apples_around,
    disc_offsets,
)

__all__ = [
    "APPLE_RADIUS",
    "check_ate_last_apple_in_cluster",
    "count_apples_around",
    "count_nearby_apples",
    "compute_agent_step_metrics",
    "disc_offsets",
]


def count_nearby_apples(agent_view: np.ndarray, view_center_row: int, view_center_col: int, radius: int = 2) -> int:
    """
    Count apples within a specified radius from the agent's position in the view.
    
    Parameters
    ----------
    agent_view : np.ndarray
        The agent's view of the environment (2D array with characters)
    view_center_row : int
        Row index of the agent's position in the view (center of view)
    view_center_col : int
        Column index of the agent's position in the view (center of view)
    radius : int, optional
        Maximum distance (in steps) to count apples, by default 2
        
    Returns
    -------
    int
        Number of apples within the specified radius
    """
    nearby_apples = 0
    for i in range(agent_view.shape[0]):
        for j in range(agent_view.shape[1]):
            if agent_view[i, j] == 'A':
                # Calculate distance from agent center in view coordinates
                dist = np.sqrt((i - view_center_row)**2 + (j - view_center_col)**2)
                if dist <= radius:
                    nearby_apples += 1
    return nearby_apples


def compute_agent_step_metrics(
    agent,
    env,
    reward: float,
    apple_eaten: bool,
    nearby_radius: int = 2
) -> Dict[str, Any]:
    """
    Compute all agent-specific metrics for a single step.
    
    Parameters
    ----------
    agent
        Agent object with get_state(), get_pos(), row_size, col_size attributes
    env
        Environment object with apple_points and world_map attributes
    reward : float
        Reward received by the agent in this step
    apple_eaten : bool
        Whether an apple was eaten (reward > 0)
    nearby_radius : int, optional
        Radius for counting nearby apples, by default 2
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - nearby_apples: int
        - ate_last_apple_in_cluster: bool
    """
    # Get agent's view and position
    agent_view = agent.get_state()
    view_center_row = agent.row_size
    view_center_col = agent.col_size
    
    # Count nearby apples
    nearby_apples = count_nearby_apples(agent_view, view_center_row, view_center_col, nearby_radius)
    
    # Check if agent ate the last apple in a cluster
    ate_last_apple_in_cluster = False
    if apple_eaten:
        agent_pos = agent.get_pos()
        ate_last_apple_in_cluster = check_ate_last_apple_in_cluster(
            agent_pos,
            env.apple_points,
            env.world_map
        )
    
    return {
        "nearby_apples": nearby_apples,
        "ate_last_apple_in_cluster": ate_last_apple_in_cluster,
    }

