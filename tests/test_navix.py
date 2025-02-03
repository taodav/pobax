from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


from pobax.envs import get_env
from definitions import ROOT_DIR

if __name__ == "__main__":

    steps = 1000
    key = jax.random.PRNGKey(2024)

    env_key, key = jax.random.split(key)

    env_keys = jax.random.split(env_key, 2)

    env, env_params = get_env("Navix-DMLab-Maze-RGB-01-v0", env_key,
                              normalize_image=False)
    obs, env_state = env.reset(env_keys, env_params)

    vmap_sample_action = jax.vmap(env.action_space(env_params).sample)

    def env_step(runner_state, _):
        env_state, last_obs, last_done, rng = runner_state
        rng_step, rng_action, rng = jax.random.split(rng, 3)
        rngs_action = jax.random.split(rng_action, 2)
        action = vmap_sample_action(rngs_action)
        obs, env_state, reward, done, info = env.step(rng_step, env_state, action, env_params)

        new_runner_state = env_state, obs, done, rng
        return new_runner_state, (obs, done)

    runner_state = env_state, obs, jnp.zeros(2).astype(bool), key
    final_runner_state, (obs, dones) = jax.lax.scan(env_step, runner_state, jnp.arange(steps), steps)


    # plt.imshow(tstep.observation)
    # plt.show()

    concat_obs = jnp.concatenate((obs[:, 0], obs[:, 1]), axis=1)

    imgs = [Image.fromarray(np.array(img)) for img in concat_obs]
    gif_path = Path(ROOT_DIR, 'results', "navix_maze.gif")
    imgs[0].save(gif_path, save_all=True, append_images=imgs[1:], duration=200, loop=0)

    print(f"saved to {gif_path}")
