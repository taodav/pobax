from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from navix import observations
from navix.rendering.cache import TILE_SIZE

from pobax.envs import get_env
from definitions import ROOT_DIR

def overlay_on_rgb(obs, states, overlay_alpha: float = 30):
    s = states.env_state.timestep.state
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
            start_r, start_c = position[0], position[1] - 3
        elif direction == 0:
            start_r, start_c = position[0] - 3, position[1]
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

def gather_and_save_gif():
    steps = 1000
    key = jax.random.PRNGKey(2024)

    env_key, key = jax.random.split(key)
    env, env_params = get_env("Navix-DMLab-Maze-01-v0", env_key,
                              normalize_image=False)

    env_keys = jax.random.split(key, 2)
    obs, env_state = env.reset(env_keys, env_params)

    vmap_sample_action = jax.vmap(env.action_space(env_params).sample)

    @jax.jit
    def env_step(runner_state, _):
        env_state, last_obs, last_done, rng = runner_state
        rng_step, rng_action, rng = jax.random.split(rng, 3)
        rngs_action = jax.random.split(rng_action, 2)
        action = vmap_sample_action(rngs_action)
        obs, env_state, reward, done, info = env.step(rng_step, env_state, action, env_params)

        new_runner_state = env_state, obs, done, rng
        return new_runner_state, (obs, done, env_state)


    runner_state = env_state, obs, jnp.zeros(2).astype(bool), key
    final_runner_state, (obs, dones, env_state) = jax.lax.scan(env_step, runner_state, jnp.arange(steps), steps)

    # plt.imshow(tstep.observation)
    # plt.show()
    np_images = overlay_on_rgb(np.array(obs[:, 0]), jax.tree.map(lambda x: np.array(x[:, 0]), env_state))

    # concat_obs = jnp.concatenate((obs[:, 0], obs[:, 1]), axis=1)
    #
    imgs = [Image.fromarray(img) for img in np_images]
    gif_path = Path(ROOT_DIR, 'results', "navix_maze.gif")
    imgs[0].save(gif_path, save_all=True, append_images=imgs[1:], duration=200, loop=0)

    print(f"saved to {gif_path}")

def test_masking():
    key = jax.random.PRNGKey(2024)

    env_key, key = jax.random.split(key)
    env, env_params = get_env("Navix-DMLab-Maze-01-v0", env_key,
                              normalize_image=False)
    env_keys = jax.random.split(key, 2)
    obs, env_state = env.reset(env_keys, env_params)
    pass

if __name__ == "__main__":
    # jax.disable_jit(True)

    # test_masking()
    gather_and_save_gif()
