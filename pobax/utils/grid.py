# Helpers for agent-centric gridworld mapping
from jax import lax, Array
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


def agent_centric_map(
        grid: Array, origin: Array, direction: Array, padding_value: int = 0
) -> Array:
    """Puts a grid around a given origin, facing a given direction, with a given radius.

    Args:
        grid (Array): A 2D grid of shape `(height, width)`.
        origin (Array): The origin of the crop.
        direction (Array): The direction the crop is facing.
        padding_value (int, optional): The padding value. Defaults to 0.

    Returns:
        Array: A cropped grid."""
    input_shape = grid.shape
    largest_dim = max(input_shape[0], input_shape[1])
    radius = largest_dim - 1

    add_padding_axis_0 = largest_dim - input_shape[0]

    add_padding_axis_1 = largest_dim - input_shape[1]

    # pad with radius
    padding = [(radius, 0), (radius, 0)]
    for _ in range(len(input_shape) - 2):
        padding.append((0, 0))

    padded = jnp.pad(grid, padding, constant_values=padding_value)

    # translate the grid such that the agent is `radius` away from the top and left edges
    translated = jnp.roll(padded, -jnp.asarray(origin), axis=(0, 1))

    # now we need to make it square
    add_padding = [(0, add_padding_axis_0), (0, add_padding_axis_1)]
    for _ in range(len(input_shape) - 2):
        add_padding.append((0, 0))
    translated_square = jnp.pad(translated, add_padding, constant_values=padding_value)

    # crop such that the agent is in the centre of the grid
    # cropped = translated[: 2 * radius + 1, : 2 * radius + 1]

    # rotate such that the agent is facing north
    rotated = lax.switch(
        direction,
        (
            lambda x: jnp.rot90(x, 1),  # 0 = transpose, 1 = flip
            lambda x: jnp.rot90(x, 2),  # 0 = flip, 1 = flip
            lambda x: jnp.rot90(x, 3),  # 0 = flip, 1 = transpose
            lambda x: x,
        ),
        translated_square,
    )

    # cropped = rotated.at[: radius + 1].get(fill_value=padding_value)
    return jnp.asarray(rotated, dtype=grid.dtype)


def agent_position_map(
        grid: Array, origin: Array, padding_value: int = 0
) -> Array:
    """Puts a grid around a given origin, with a given radius.

    Args:
        grid (Array): A 2D grid of shape `(height, width)`.
        origin (Array): The origin of the crop.
        direction (Array): The direction the crop is facing.
        padding_value (int, optional): The padding value. Defaults to 0.

    Returns:
        Array: A cropped grid."""
    input_shape = grid.shape
    radius_r, radius_c = input_shape[0] - 1, input_shape[1] - 1

    # pad with radius
    padding = [(radius_r, 0), (radius_c, 0)]
    for _ in range(len(input_shape) - 2):
        padding.append((0, 0))

    padded = jnp.pad(grid, padding, constant_values=padding_value)

    # translate the grid such that the agent is `radius` away from the top and left edges
    translated = jnp.roll(padded, -jnp.asarray(origin), axis=(0, 1))

    # crop such that the agent is in the centre of the grid
    # cropped = translated[: 2 * radius + 1, : 2 * radius + 1]

    # cropped = rotated.at[: radius + 1].get(fill_value=padding_value)
    return jnp.asarray(translated, dtype=grid.dtype)

if __name__ == "__main__":
    grid = jnp.array([
        [1, 1, 1, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 1, 0, 1, 1, 1],
        [0, 0, 1, 0, 0, 1],
    ], dtype=float)
    # pos = jnp.array([2, 2])
    pos = jnp.array([2, 2])
    direction = jnp.array(2)
    ac_map = agent_centric_map(grid, pos, direction)
    ac_map_np = np.array(ac_map)
    print()
