from pathlib import Path

import chex
import jax
import orbax.checkpoint

from pobax.algos.ppo import PPO
from pobax.envs import get_env
from pobax.config import PPOHyperparams
from pobax.models import get_gymnax_network_fn, ScannedRNN


def load_train_state(fpath: Path, key: chex.PRNGKey):
    env_key, key = jax.random.split(key)
    # load our params
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    restored = orbax_checkpointer.restore(fpath)
    args = restored['args']
    args = PPOHyperparams().from_dict(args)

    env, env_params = get_env(args.env, env_key,
                              gamma=args.gamma,
                              normalize_image=False,
                              action_concat=args.action_concat)

    double_critic = args.double_critic
    memoryless = args.memoryless

    network_fn, action_size = get_gymnax_network_fn(env, env_params, memoryless=memoryless)

    network = network_fn(action_size,
                         double_critic=double_critic,
                         hidden_size=args.hidden_size)

    agent = PPO(network,
                double_critic=double_critic,
                ld_weight=args.ld_weight.item(),
                alpha=args.alpha.item(),
                vf_coeff=args.vf_coeff.item(),
                clip_eps=args.clip_eps, entropy_coeff=args.entropy_coeff)

    ts_dict = restored['final_train_state']

    return env, env_params, args, agent, ts_dict['params'], {'metric': restored['metric']}


if __name__ == "__main__":
    key = jax.random.key(2024)

    ckpt_path = Path('/Users/ruoyutao/Documents/pobax/results/batch_ppo_test/Navix-DMLab-Maze-00-v0_seed(2020)_time(20250206-122849)_9a1918ffc5684688ac6d18ecd9c02d89')

    env, env_params, args, agent, params, metric = load_train_state(ckpt_path, key)

    print()


