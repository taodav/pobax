import jax
import jax.numpy as jnp
from renderer import transpose_for_display

from pobax.envs import load_brax_env
from pobax.envs.wrappers.pixel import PixelBraxEnv


if __name__ == "__main__":
    env_name = 'ant'
    seed = 2024
    jax.config.update('jax_platform_name', 'gpu')

    key = jax.random.PRNGKey(seed)

    env, env_params = load_brax_env(env_name)
    env = PixelBraxEnv(env)

    reset_key, key = jax.random.split(key)
    img, state = env.reset(reset_key, env_params)

    transpose_for_display((img * 255).astype(jnp.uint8))


    print()


