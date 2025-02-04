# Helpers for agent-centric gridworld mapping
from jax import lax
import jax.numpy as jnp
import numpy as np


def convert_to_agent_centric_map(grid: jnp.ndarray, pos: jnp.ndarray):
    """
    Given a rows x cols grid, returns an agent-centric (2 * rows - 1) x (2 * cols - 1) map of
    the same grid.
    :param grid:
    :param pos:
    :return:
    """
    expanded_rows = grid.shape[0] + grid.shape[0] - 1
    expanded_cols = grid.shape[1] + grid.shape[1] - 1
    expanded_agent_pos = jnp.array([expanded_rows // 2, expanded_cols // 2])

    expanded_map = jnp.zeros((expanded_rows, expanded_cols)) - 1
    y_start, x_start = expanded_agent_pos[0] - pos[0], expanded_agent_pos[1] - pos[1]
    expanded_map = lax.dynamic_update_slice(expanded_map, grid, (y_start, x_start))
    return expanded_map


def position_to_map(pos: jnp.ndarray, center: jnp.ndarray, rows: int, cols: int):
    pass


def positions_to_map(poss: jnp.ndarray, center: jnp.ndarray, rows: int, cols: int):
    # Now we get our expanded map, with our agent at the center.
    expanded_rows = rows + rows - 1
    expanded_agent_pos = expanded_map_size // 2

    expanded_map = jnp.zeros((expanded_map_size, expanded_map_size, unexpanded_map_no_pos.shape[-1]))
    y_start, x_start = expanded_agent_pos - state.position[0], expanded_agent_pos - state.position[1]
    expanded_map = lax.dynamic_update_slice(expanded_map, unexpanded_map_no_pos, (y_start, x_start, 0))


def bresenham_line(r0, c0, r1, c1):
    """
    Return the list of grid cells (as (row, col) tuples) from (r0,c0) to (r1,c1)
    using Bresenham’s algorithm.
    """
    points = []
    # Compute differences
    dr = abs(r1 - r0)
    dc = abs(c1 - c0)
    # For a horizontal or vertical line, we want a zero step in that direction.
    step_r = 0 if r0 == r1 else (1 if r0 < r1 else -1)
    step_c = 0 if c0 == c1 else (1 if c0 < c1 else -1)
    err = dc - dr
    r, c = r0, c0
    while True:
        points.append((r, c))
        if r == r1 and c == c1:
            break
        e2 = 2 * err
        if e2 > -dr:
            err -= dr
            c += step_c
        if e2 < dc:
            err += dc
            r += step_r
    return points


def precompute_rays(GRID_ROWS: int, GRID_COLS: int, AGENT_R: int, AGENT_C: int):
    """
    For every cell in the GRID_ROWS x GRID_COLS grid,
    compute the list (ray) of cells from the agent’s cell (AGENT_R, AGENT_C)
    to that cell (excluding the agent’s own cell).

    Returns:
      - RAYS: an array of shape (GRID_ROWS, GRID_COLS, max_ray_length, 2)
              where for each target cell we pad with (-1, -1) if the ray is shorter.
      - RAY_LENGTHS: an integer array (GRID_ROWS, GRID_COLS) giving the actual ray lengths.
    """
    max_length = 0
    ray_dict = {}  # temporary dict: (r,c) -> list of (row,col) along the ray
    ray_lengths = np.zeros((GRID_ROWS, GRID_COLS), dtype=np.int32)
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            pts = bresenham_line(AGENT_R, AGENT_C, r, c)
            # We drop the agent’s own cell (the first element) because we start looking “ahead.”
            ray = pts[1:]
            ray_dict[(r, c)] = ray
            ray_lengths[r, c] = len(ray)
            if len(ray) > max_length:
                max_length = len(ray)
    # Create an array for the rays.
    RAYS = np.full((GRID_ROWS, GRID_COLS, max_length, 2), -1, dtype=np.int32)
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            for i, (rr, cc) in enumerate(ray_dict[(r, c)]):
                RAYS[r, c, i, 0] = rr
                RAYS[r, c, i, 1] = cc
    return jnp.array(RAYS), jnp.array(ray_lengths)


if __name__ == "__main__":
    grid = jnp.array([
        [0, 1, 1, 0],
        [0, 1, 1, 0],
        [0, 1, 1, 0],
        [0, 1, 1, 0],
    ], dtype=float)
    pos = jnp.array([3, 3])
    ac_map = convert_to_agent_centric_map(grid, pos)
    ac_map_np = np.array(ac_map)
    print()
