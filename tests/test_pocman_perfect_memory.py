import jax

# DEBUG
import numpy as np

from pobax.envs.pocman import PocMan, PerfectMemoryWrapper


def test_pocman_perfect_memory():
    seed = 2024
    key = jax.random.PRNGKey(seed)
    reset_key, key = jax.random.split(key)

    env = PocMan()
    env_params = env.default_params
    env = PerfectMemoryWrapper(env)

    obs, state = env.reset(reset_key, env_params)

    step_key, key = jax.random.split(key)
    obs, state, rew, terminal, info = env.step(step_key, state, 1, env_params)
    pass


if __name__ == "__main__":
    jax.disable_jit(True)

    test_pocman_perfect_memory()
