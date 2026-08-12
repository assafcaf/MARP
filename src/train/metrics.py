"""
Metrics calculation module for agent episode tracking.

This module provides functions to calculate agent-specific metrics such as:
- nearby_apples: Count of apples within a small radius of the agent
- ate_last_apple_in_cluster: Whether the agent consumed the last apple in a cluster
"""

import numpy as np
from typing import Tuple, Dict, Any
from src.env.commons_env import APPLE_RADIUS


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


def check_ate_last_apple_in_cluster(
    agent_pos: np.ndarray,
    apple_points: list,
    world_map: np.ndarray,
    apple_radius: int = APPLE_RADIUS
) -> bool:
    """
    Check if the agent ate the last apple in a cluster.
    
    An apple is considered the "last in cluster" if:
    1. It was eaten (agent is at a position that was an apple spawn point)
    2. No other apples remain within APPLE_RADIUS of the nearest spawn point
    
    Parameters
    ----------
    agent_pos : np.ndarray
        Agent's position in world coordinates [row, col]
    apple_points : list
        List of apple spawn points [[row, col], ...]
    world_map : np.ndarray
        Current state of the world map (after the apple was eaten)
    apple_radius : int, optional
        Radius for apple cluster detection, by default APPLE_RADIUS
        
    Returns
    -------
    bool
        True if the agent ate the last apple in a cluster, False otherwise
    """
    # Find the nearest apple spawn point to determine which cluster this apple belonged to
    nearest_spawn_point = None
    min_dist = float('inf')
    
    for spawn_point in apple_points:
        dist = np.sqrt((agent_pos[0] - spawn_point[0])**2 + (agent_pos[1] - spawn_point[1])**2)
        if dist < min_dist:
            min_dist = dist
            nearest_spawn_point = spawn_point
    
    # If we found a nearby spawn point, check if any apples remain in its cluster
    if nearest_spawn_point is None or min_dist > apple_radius:
        return False
    
    # Count how many apples are within APPLE_RADIUS of this spawn point
    # (the eaten apple is already gone from world_map)
    apples_in_cluster = 0
    for j in range(-apple_radius, apple_radius + 1):
        for k in range(-apple_radius, apple_radius + 1):
            if j ** 2 + k ** 2 <= apple_radius:
                x, y = nearest_spawn_point[0] + j, nearest_spawn_point[1] + k
                if 0 <= x < world_map.shape[0] and 0 <= y < world_map.shape[1]:
                    if world_map[x, y] == 'A':
                        apples_in_cluster += 1
    
    # If no apples remain in the cluster, this was the last one
    return apples_in_cluster == 0


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

