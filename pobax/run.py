"""
Runs without a jax environment
"""
from functools import partial

import chex
import gymnasium as gym
from gymnasium.wrappers import PixelObservationWrapper
import jax

from pobax.config import PPOHyperparams
from pobax.models import get_network_fn, ScannedRNN
from pobax.algos.ppo import PPO

def make_train(args: PPOHyperparams, rand_key: chex.PRNGKey):
    pixel_wrapper = partial(PixelObservationWrapper)

    env = gym.vector.make(args.env, num_envs=args.num_envs, wrappers=pixel_wrapper, render_mode='rgb_array')

    network_fn, action_size = get_network_fn(env, memoryless=args.memoryless)

    network = network_fn(action_size,
                         double_critic=args.double_critic,
                         hidden_size=args.hidden_size)
    agent = PPO(network, args.double_critic, args.ld_weight, args.alpha, args.vf_coeff)


if __name__ == "__main__":
    args = PPOHyperparams().parse_args()
    jax.config.update('jax_platform_name', args.platform)

    # Check that we're not trying to vmap over hyperparams
    for k, v in args.as_dict().items():
        if isinstance(v, list):
            assert len(v) == 1, "Can't run with multiple hyperparams."
            setattr(args, k, v[0])

    key = jax.random.PRNGKey(args.seed)
    make_train(args, key)

    print()
