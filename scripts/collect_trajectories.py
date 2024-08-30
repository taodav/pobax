from functools import partial
from pathlib import Path
from typing import Union, NamedTuple

import chex
import jax
import jax.numpy as jnp
from jax_tqdm import scan_tqdm
import numpy as np
from tap import Tap
from flax.training import orbax_utils
import orbax.checkpoint

from gymnax.environments import environment
from pobax.models.network import ScannedRNN
from pobax.utils.file_system import load_train_state, make_hash_md5

class CollectHyperparams(Tap):
    collect_path: Union[str, Path]

    update_idx_to_take: int = None

    num_envs: int = 4
    n_samples: int = int(1e6)

    seed: int = 2024
    platform: str = 'gpu'
    study_name: str = 'test'

    def configure(self) -> None:
        self.add_argument('--collect_path', type=Path)

def ppo_step(runner_state, unused,
                    network,
                    env, env_params):
    
    def get_state(s):
        if hasattr(s, 'env_state'):
            return get_state(s.env_state)
        else:
            return s

    (ts, env_state, last_obs, last_done,
        hstate, rng) = runner_state
    rng, _rng = jax.random.split(rng)

    # SELECT ACTION
    ac_in = (last_obs[np.newaxis, :], last_done[np.newaxis, :])
    next_hstate, pi, value, embedding = network.apply(ts.params, hstate, ac_in)
    action = pi.sample(seed=_rng)
    log_prob = pi.log_prob(action)
    value, action, log_prob = (
        value.squeeze(0),
        action.squeeze(0),
        log_prob.squeeze(0),
    )
    embedding = embedding.squeeze(0)

    # STEP ENV
    rng, _rng = jax.random.split(rng)
    rng_step = jax.random.split(_rng, next_hstate.shape[0])
    obsv, next_env_state, reward, done, info = env.step(rng_step, env_state, action, env_params)

    # transition = Transition(
    #     last_done, action, value, reward, log_prob, last_obs, info
    # )
    state = get_state(env_state)
    datum = {
        'x': embedding,
        'observation': last_obs,
    }
    print(datum['x'].shape)
    runner_state = (ts, next_env_state, obsv, done,
                    next_hstate, rng)
    return runner_state, datum


def make_collect(args: CollectHyperparams, key: chex.PRNGKey):
    steps_to_collect = args.n_samples // args.num_envs

    network_key, key = jax.random.split(key, 2)

    args.collect_path = Path(args.collect_path).resolve()

    env, env_params, network_args, network, ts = load_train_state(network_key, args.collect_path,
                                                                                     update_idx_to_take=args.update_idx_to_take,
                                                                                     best_over_rng=True)
    args.study_name = network_args['study_name']

    _env_step = partial(ppo_step, network=network,
                        env=env, env_params=env_params)
    _env_step = scan_tqdm(steps_to_collect)(_env_step)

    ckpts = {
        'network': {'args': network_args, 'ts': ts, 'path': args.collect_path},
    }

    def collect(rng):
        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, args.num_envs)
        obsv, env_state = env.reset(reset_rng, env_params)

        # init hidden state
        init_hstate = ScannedRNN.initialize_carry(args.num_envs, network_args['hidden_size'])
        init_runner_state = (
            ts,
            env_state,
            obsv,
            jnp.zeros(args.num_envs, dtype=bool),
            init_hstate,
            _rng,
        )

        runner_state, dataset = jax.lax.scan(
            _env_step, init_runner_state, jnp.arange(steps_to_collect), steps_to_collect
        )

        # Now we flatten back down
        flat_dataset = jax.tree.map(lambda x: x.reshape(-1, *x.shape[2:]), dataset)

        return flat_dataset

    return collect, ckpts


if __name__ == "__main__":
    # jax.disable_jit(True)
    args = CollectHyperparams().parse_args()
    jax.config.update('jax_platform_name', args.platform)

    key = jax.random.PRNGKey(args.seed)
    make_key, collect_key, key = jax.random.split(key, 3)

    collect_fn, ckpt_info = make_collect(args, make_key)
    collect_fn = jax.jit(collect_fn)

    dataset = collect_fn(collect_key)

    def path_to_str(d: dict):
        for k, v in d.items():
            if isinstance(v, Path):
                d[k] = str(v)
            elif isinstance(v, dict):
                path_to_str(v)


    to_save = {
        'dataset': dataset,
        'args': args.as_dict(),
        'ckpt': ckpt_info,
    }
    path_to_str(to_save)

    save_path = args.collect_path.parent / \
                f'dataset'

    # Save all results with Orbax
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    save_args = orbax_utils.save_args_from_target(to_save)

    print(f"Saving results to {save_path}")
    orbax_checkpointer.save(save_path, to_save, save_args=save_args)

    print("Done.")