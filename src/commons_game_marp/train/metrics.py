"""
Metrics calculation module for agent episode tracking.

This module provides functions to calculate agent-specific metrics such as:
- nearby_apples: Count of apples within a small radius of the agent
- ate_last_apple_in_cluster: Whether the agent consumed the last apple in a cluster
"""

import functools
import numpy as np
from typing import Tuple, Dict, Any
from ..env.commons_env import APPLE_RADIUS


@functools.lru_cache(maxsize=16)
def disc_offsets(radius: int) -> np.ndarray:
    """Row/col offsets of every cell within Euclidean `radius` of the origin.

    Cached per radius: the offset set is a property of the radius alone, and
    this is rebuilt once per agent per step otherwise.

    Returns
    -------
    np.ndarray
        Shape `(K, 2)`, integer offsets with `sqrt(dr^2 + dc^2) <= radius`.
        The origin itself is included, matching `count_nearby_apples`, which
        counts an apple standing on the agent's own cell.
    """
    span = np.arange(-radius, radius + 1)
    rows, cols = np.meshgrid(span, span, indexing="ij")
    inside = (rows**2 + cols**2) <= radius**2
    return np.stack([rows[inside], cols[inside]], axis=1).astype(np.int64)


def count_apples_around(grid: np.ndarray, pos, offsets: np.ndarray) -> int:
    """Count apples within a precomputed offset stencil of `pos` on `grid`.

    The vectorised equivalent of `count_nearby_apples(agent.get_state(), ...)`,
    reading the same array the agent's view is cut from (`agent.grid`, which
    `MapEnv.step` refreshes every step) instead of materialising the view. That
    matters because this now runs for every agent of every environment on every
    step, not just environment 0 -- the Python double loop it replaces walked
    all 225 cells of a 15x15 view per call.

    Cells outside the map are treated as empty, which is what `return_view`'s
    zero padding amounts to. An agent in timeout sits at `OUTCAST_POSITION`, so
    every offset falls outside the map and the count is 0.

    Parameters
    ----------
    grid : np.ndarray
        2D character map, e.g. `agent.grid` or `env.world_map`.
    pos : sequence of int
        Agent's `[row, col]` position in `grid` coordinates.
    offsets : np.ndarray
        Stencil from `disc_offsets`.

    Returns
    -------
    int
        Number of `'A'` cells within the stencil.
    """
    rows = offsets[:, 0] + int(pos[0])
    cols = offsets[:, 1] + int(pos[1])
    inside = (
        (rows >= 0) & (rows < grid.shape[0]) & (cols >= 0) & (cols < grid.shape[1])
    )
    if not inside.any():
        return 0
    return int(np.count_nonzero(grid[rows[inside], cols[inside]] == 'A'))


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

