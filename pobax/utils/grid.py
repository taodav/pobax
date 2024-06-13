# Helpers for agent-centric gridworld mapping
from jax import lax
import jax.numpy as jnp


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




if __name__ == "__main__":
    import numpy as np
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
