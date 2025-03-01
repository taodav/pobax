from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image
from jax_tqdm import scan_tqdm

from pobax.envs import get_env
from pobax.utils.video import navix_overlay_obs_on_rgb

from definitions import ROOT_DIR


def random_policy_gif():
    steps = 2000
    key = jax.random.PRNGKey(2026)
    env_name = "Navix-DMLab-Maze-01-v0"

    env_key, key = jax.random.split(key)
    env, env_params = get_env(env_name, env_key,
                              normalize_image=False)

    env_keys = jax.random.split(key, 2)
    obs, env_state = env.reset(env_keys, env_params)

    vmap_sample_action = jax.vmap(env.action_space(env_params).sample)

    @jax.jit
    @scan_tqdm(steps)
    def env_step(runner_state, _):
        prev_env_state, last_obs, last_done, rng = runner_state

        rng_step, rng_action, rng = jax.random.split(rng, 3)
        rngs_action = jax.random.split(rng_action, 2)
        action = vmap_sample_action(rngs_action)
        obs, env_state, reward, done, info = env.step(rng_step, prev_env_state, action, env_params)

        new_runner_state = env_state, obs, done, rng
        return new_runner_state, (obs, done, env_state)


    runner_state = env_state, obs, jnp.zeros(2).astype(bool), key
    final_runner_state, (obs, dones, env_state) = jax.lax.scan(env_step, runner_state, jnp.arange(steps), steps)

    # plt.imshow(tstep.observation)
    # plt.show()
    np_images = navix_overlay_obs_on_rgb(np.array(obs[:, 0]), jax.tree.map(lambda x: np.array(x[:, 0]), env_state))

    print(f"Collected {steps} samples. Turning into gif now.")
    # concat_obs = jnp.concatenate((obs[:, 0], obs[:, 1]), axis=1)
    #
    imgs = [Image.fromarray(img) for img in np_images]
    vod_path = Path(ROOT_DIR, 'results', "navix_maze.gif")
    imgs[0].save(vod_path, save_all=True, append_images=imgs[1:], duration=100, loop=0)

    pdf_path = Path(ROOT_DIR, 'results', f'{env_name}.pdf')
    imgs[0].save(pdf_path, save_all=True)

    print(f"saved to {vod_path} and {pdf_path}")

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
    random_policy_gif()
