from functools import partial
from time import time
from typing import Union

import jax
import jax.numpy as jnp
from jax_tqdm import scan_tqdm
from tap import Tap

from pobax.envs import get_env


@partial(jax.jit, static_argnames=['env', 'env_params', 'n_steps', 'n_envs'])
def run_n_steps(rng, env, env_params, n_steps: int = int(1e6), n_envs: int = 1):
    action_sample = jax.vmap(env.action_space(env_params).sample)

    @scan_tqdm(n_steps)
    def env_step(runner_state, _):
        env_state, rng = runner_state
        _action_rng, rng = jax.random.split(rng)
        action_rngs = jax.random.split(_action_rng, n_envs)
        action = action_sample(action_rngs)

        step_rng = jax.random.split(rng, n_envs)
        obsv, env_state, reward, done, info = env.step(step_rng, env_state, action, env_params)

        return (env_state, rng), None

    _rng, runner_state_rng, rng = jax.random.split(rng, 3)
    init_rng = jax.random.split(rng, n_envs)
    init_obsv, init_env_state = env.reset(init_rng, env_params)
    step_rng, rng = jax.random.split(rng)
    init_runner_state = (init_env_state, step_rng)
    return jax.lax.scan(
        env_step, init_runner_state, jnp.arange(n_steps), n_steps
    )


class SampleHyperparams(Tap):
    env: str = 'rocksample_11_11'
    n_envs: int = 1
    n_steps: Union[int, str] = int(5e6)

    seed: int = 2024
    platform: str = 'cpu'

    def configure(self) -> None:
        def to_int(s):
            return int(float(s))
        self.add_argument('--n_steps', type=to_int)


if __name__ == "__main__":
    args = SampleHyperparams().parse_args()
    key = jax.random.PRNGKey(args.seed)

    key, env_key = jax.random.split(key)
    env, env_params = get_env(args.env, env_key,
                              normalize_image=False,
                              log_stats=False
                              )

    t = time()
    out = jax.block_until_ready(run_n_steps(key, env, env_params, n_steps=args.n_steps, n_envs=args.n_envs))
    new_t = time()
    total_runtime = new_t - t
    print(f'Total runtime for {args.env} environment with {args.n_envs} envs and {args.n_steps} steps:', total_runtime)
