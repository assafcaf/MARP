"""Per-step, per-agent measurements taken inside the environment.

These used to be computed by the trainer, which reached into live environment
objects -- `agent.grid`, `agent.get_pos()`, `env.world_map`, `env.apple_points`.
That works only while the environment lives in the same process. Computing them
here instead means a worker can put two scalars on the info dict and the main
process never needs the objects at all, which is what makes
`SubprocVecCommonsEnv` possible.

The module also owns `APPLE_RADIUS`, so `commons_env` can import from here
without a cycle. `commons_env` and `train.metrics` both re-export it, so every
pre-existing import path still resolves.
"""

import functools

import numpy as np

# Radius of the neighbourhood that drives apple regrowth: `spawn_apples` scales
# spawn probability with how many apples remain within it.
APPLE_RADIUS = 2


@functools.lru_cache(maxsize=16)
def disc_offsets(radius: int) -> np.ndarray:
    """Row/col offsets of every cell within Euclidean `radius` of the origin.

    Cached per radius: the offset set depends on nothing else, and this would
    otherwise be rebuilt once per agent per step.

    Returns
    -------
    np.ndarray
        Shape `(K, 2)`, integer offsets with `sqrt(dr^2 + dc^2) <= radius`. The
        origin is included, matching `count_nearby_apples`, which counts an
        apple standing on the agent's own cell.
    """
    span = np.arange(-radius, radius + 1)
    rows, cols = np.meshgrid(span, span, indexing="ij")
    inside = (rows**2 + cols**2) <= radius**2
    return np.stack([rows[inside], cols[inside]], axis=1).astype(np.int64)


def count_apples_around(grid: np.ndarray, pos, offsets: np.ndarray) -> int:
    """Count apples within a precomputed offset stencil of `pos` on `grid`.

    Cells outside the map are treated as empty, which is what `return_view`'s
    zero padding amounts to. An agent in timeout sits at `OUTCAST_POSITION`, so
    every offset falls outside the map and the count is 0.
    """
    rows = offsets[:, 0] + int(pos[0])
    cols = offsets[:, 1] + int(pos[1])
    inside = (
        (rows >= 0) & (rows < grid.shape[0]) & (cols >= 0) & (cols < grid.shape[1])
    )
    if not inside.any():
        return 0
    return int(np.count_nonzero(grid[rows[inside], cols[inside]] == 'A'))


def check_ate_last_apple_in_cluster(
    agent_pos: np.ndarray,
    apple_points: list,
    world_map: np.ndarray,
    apple_radius: int = APPLE_RADIUS,
) -> bool:
    """Whether the apple just eaten was the last one standing in its cluster.

    An apple with no surviving neighbours cannot seed regrowth, so a high rate
    of these is a policy eating its own future.

    Note the deliberate `j ** 2 + k ** 2 <= apple_radius` comparison: it matches
    `spawn_apples`, which compares a squared distance against an unsquared
    radius and so treats a cluster as a 9-cell block rather than the 13-cell
    radius-2 neighbourhood. The two must agree, or this would report clusters
    empty that the spawner still considers seeded. See the xfail in
    tests/test_metrics.py.
    """
    nearest_spawn_point = None
    min_dist = float('inf')

    for spawn_point in apple_points:
        dist = np.sqrt(
            (agent_pos[0] - spawn_point[0]) ** 2 + (agent_pos[1] - spawn_point[1]) ** 2
        )
        if dist < min_dist:
            min_dist = dist
            nearest_spawn_point = spawn_point

    if nearest_spawn_point is None or min_dist > apple_radius:
        return False

    apples_in_cluster = 0
    for j in range(-apple_radius, apple_radius + 1):
        for k in range(-apple_radius, apple_radius + 1):
            if j ** 2 + k ** 2 <= apple_radius:
                x, y = nearest_spawn_point[0] + j, nearest_spawn_point[1] + k
                if 0 <= x < world_map.shape[0] and 0 <= y < world_map.shape[1]:
                    if world_map[x, y] == 'A':
                        apples_in_cluster += 1

    return apples_in_cluster == 0
