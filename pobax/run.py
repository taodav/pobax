"""
Runs without a jax environment
"""
import chex
import gymnasium as gym
import jax

from pobax.config import PPOHyperparams
from pobax.models import get_network_fn, ScannedRNN
from pobax.algos.ppo import PPO

def make_train(args: PPOHyperparams, rand_key: chex.PRNGKey):

    env = gym.vector.make(args.env)

    network_fn, action_size = get_network_fn(env, memoryless=memoryless)

    network = network_fn(action_size,
                         double_critic=double_critic,
                         hidden_size=args.hidden_size)
    agent = PPO()


if __name__ == "__main__":
    args = PPOHyperparams().parse_args()
    jax.config.update('jax_platform_name', args.platform)

    # Check that we're not trying to vmap over hyperparams
    for k, v in args.as_dict().items():
        if isinstance(v, list):
            assert len(v) == 1, "Can't run with multiple hyperparams."
            setattr(args, k, v[0])

    print()
