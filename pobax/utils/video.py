import jax
from jax import numpy as jnp
import numpy as np
from navix import observations
from navix.rendering.cache import TILE_SIZE


def navix_overlay_obs_on_rgb(obs, states, overlay_alpha: float = 30):
    s = states.env_state.state
    players = s.entities['player']
    directions, positions = players.direction[:, 0], players.position[:, 0]

    obs_grid, obs_goal = obs[..., 0], obs[..., 1]

    obs_grid += ((obs_grid == 0) * 2)

    empty_grid = np.zeros_like(s.grid[0])
    radius = observations.RADIUS
    input_shape = s.grid[0].shape
    padding = [(radius, radius), (radius, radius)]
    for _ in range(len(input_shape) - 2):
        padding.append((0, 0))
    padded_positions = positions + radius

    all_images = []
    for i, (direction, position, obs) in enumerate(zip(directions, padded_positions, obs_grid)):
        patch = np.pad(empty_grid, padding, constant_values=0)
        # we do 3 - direction to recover back the direction we were facing before.
        direction = direction.item()
        rotated_obs = np.rot90(obs, k=3 - direction)

        if direction == 2 or direction == 3:
            start_r, start_c = position[0] - radius, position[1] - radius
        elif direction == 1:
            start_r, start_c = position[0], position[1] - radius
        elif direction == 0:
            start_r, start_c = position[0] - radius, position[1]
        else:
            raise NotImplementedError("what")

        patch[start_r:start_r + rotated_obs.shape[0], start_c:start_c + rotated_obs.shape[1]] = rotated_obs
        # remove padding
        patch = patch[radius:-radius, radius:-radius]
        # make each grid TILE_SIZE
        patch = patch.repeat(TILE_SIZE, axis=0).repeat(TILE_SIZE, axis=1)

        og_rgb = np.array(observations.rgb(jax.tree.map(lambda x: jnp.array(x[i]), s)))

        rgb = np.maximum(og_rgb + (patch[..., None] * overlay_alpha), 0)
        rgb = np.minimum(rgb, 255).astype(np.uint8)
        all_images.append(rgb)

    return all_images
