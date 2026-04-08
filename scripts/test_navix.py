import jax
from pobax.envs import get_env

rand_key = jax.random.PRNGKey(2025)
env_key, rand_key = jax.random.split(rand_key)

env, env_params = get_env("Navix-DMLab-Maze-01-v0", env_key)

reset_key, rand_key = jax.random.split(rand_key)
reset_keys = jax.random.split(rand_key, 4)

obs, env_state = env.reset(reset_keys, env_params)
print(obs)