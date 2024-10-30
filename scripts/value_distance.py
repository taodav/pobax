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
from pobax.utils.file_system import load_train_state, make_hash_md5, load_info


class CollectHyperparams(Tap):
    dataset_path: Union[str, Path]
    memoryless_lambda0_path: Union[str, Path]
    memoryless_lambda1_path: Union[str, Path]
    memoryless_skip_lambda0_path: Union[str, Path]
    memoryless_skip_lambda1_path: Union[str, Path]
    rnn_skip_lambda0_path: Union[str, Path]
    rnn_skip_lambda1_path: Union[str, Path]
    rnn_lambda0_path: Union[str, Path]
    rnn_lambda1_path: Union[str, Path]

    update_idx_to_take: int = None

    seed: int = 2024
    platform: str = 'gpu'

    def configure(self) -> None:
        self.add_argument('--dataset_path', type=Path)
        self.add_argument('--memoryless_lambda0_path', type=Path)
        self.add_argument('--memoryless_lambda1_path', type=Path)
        self.add_argument('--memoryless_skip_lambda0_path', type=Path)
        self.add_argument('--memoryless_skip_lambda1_path', type=Path)
        self.add_argument('--rnn_skip_lambda0_path', type=Path)
        self.add_argument('--rnn_skip_lambda1_path', type=Path)
        self.add_argument('--rnn_lambda0_path', type=Path)
        self.add_argument('--rnn_lambda1_path', type=Path)

def ppo_step(runner_state, unused, dataset, memoryless_lambda0_network, memoryless_lambda1_network, memoryless_skip_lambda0_network, memoryless_skip_lambda1_network,
                rnn_lambda0_network, rnn_lambda1_network, rnn_skip_lambda0_network, rnn_skip_lambda1_network):

    (timestep, memoryless_lambda0_ts, memoryless_lambda1_ts, memoryless_skip_lambda0_ts, memoryless_skip_lambda1_ts, rnn_lambda0_ts, rnn_lambda1_ts, 
     rnn_skip_lambda0_ts, rnn_skip_lambda1_ts, rnn_lambda0_hstate, rnn_lambda1_hstate, rnn_skip_lambda0_hstate, rnn_skip_lambda1_hstate, rng) = runner_state
    rng, _rng = jax.random.split(rng)

    # Get a single transition from dataset
    last_obs = jnp.expand_dims(dataset['observation'][timestep], axis=0)
    print(last_obs.shape)
    last_done = jnp.expand_dims(dataset['done'][timestep], axis=0)
    V_mc = dataset['V'][timestep]
    ac_in = (last_obs[np.newaxis, :], last_done[np.newaxis, :])

    # VALUE
    next_memoryless_lambda0_hstate, _, memoryless_lambda0_value, _ = memoryless_lambda0_network.apply(memoryless_lambda0_ts.params, None, ac_in)
    next_memoryless_lambda1_hstate, _, memoryless_lambda1_value, _ = memoryless_lambda1_network.apply(memoryless_lambda1_ts.params, None, ac_in)
    next_memoryless_skip_lambda0_hstate, _, memoryless_skip_lambda0_value, _ = memoryless_skip_lambda0_network.apply(memoryless_skip_lambda0_ts.params, None, ac_in)
    next_memoryless_skip_lambda1_hstate, _, memoryless_skip_lambda1_value, _ = memoryless_skip_lambda1_network.apply(memoryless_skip_lambda1_ts.params, None, ac_in)
    next_rnn_lambda0_hstate, _, rnn_lambda0_value, _ = rnn_lambda0_network.apply(rnn_lambda0_ts.params, rnn_lambda0_hstate, ac_in)
    next_rnn_lambda1_hstate, _, rnn_lambda1_value, _ = rnn_lambda1_network.apply(rnn_lambda1_ts.params, rnn_lambda1_hstate, ac_in)
    next_rnn_skip_lambda0_hstate, _, rnn_skip_lambda0_value, _ = rnn_skip_lambda0_network.apply(rnn_skip_lambda0_ts.params, rnn_skip_lambda0_hstate, ac_in)
    next_rnn_skip_lambda1_hstate, _, rnn_skip_lambda1_value, _ = rnn_skip_lambda1_network.apply(rnn_skip_lambda1_ts.params, rnn_skip_lambda1_hstate, ac_in)

    datum = {
        'memoryless_lambda0_value': memoryless_lambda0_value.squeeze(0),
        'memoryless_lambda1_value': memoryless_lambda1_value.squeeze(0),
        'memoryless_skip_lambda0_value': memoryless_skip_lambda0_value.squeeze(0),
        'memoryless_skip_lambda1_value': memoryless_skip_lambda1_value.squeeze(0),
        'rnn_lambda0_value': rnn_lambda0_value.squeeze(0),
        'rnn_lambda1_value': rnn_lambda1_value.squeeze(0),
        'rnn_skip_lambda0_value': rnn_skip_lambda0_value.squeeze(0),
        'rnn_skip_lambda1_value': rnn_skip_lambda1_value.squeeze(0),
        'V_mc': V_mc,
    }
    runner_state = (timestep+1, memoryless_lambda0_ts, memoryless_lambda1_ts, memoryless_skip_lambda0_ts, memoryless_skip_lambda1_ts, rnn_lambda0_ts, rnn_lambda1_ts, rnn_skip_lambda0_ts, rnn_skip_lambda1_ts, 
                    next_rnn_lambda0_hstate, next_rnn_lambda1_hstate, next_rnn_skip_lambda0_hstate, next_rnn_skip_lambda1_hstate, _rng)
    return runner_state, datum

def calculate_value_distance(datum):
    memoryless_lambda0_value = datum['memoryless_lambda0_value']
    memoryless_lambda1_value = datum['memoryless_lambda1_value']
    memoryless_skip_lambda0_value = datum['memoryless_skip_lambda0_value']
    memoryless_skip_lambda1_value = datum['memoryless_skip_lambda1_value']
    rnn_lambda0_value = datum['rnn_lambda0_value']
    rnn_lambda1_value = datum['rnn_lambda1_value']
    rnn_skip_lambda0_value = datum['rnn_skip_lambda0_value']
    rnn_skip_lambda1_value = datum['rnn_skip_lambda1_value']
    V_mc = datum['V_mc']

    value_distance = {
        'memoryless_lambda0': jnp.mean(jnp.abs(memoryless_lambda0_value - V_mc)),
        'memoryless_lambda1': jnp.mean(jnp.abs(memoryless_lambda1_value - V_mc)),
        'memoryless_skip_lambda0': jnp.mean(jnp.abs(memoryless_skip_lambda0_value - V_mc)),
        'memoryless_skip_lambda1': jnp.mean(jnp.abs(memoryless_skip_lambda1_value - V_mc)),
        'rnn_lambda0': jnp.mean(jnp.abs(rnn_lambda0_value - V_mc)),
        'rnn_lambda1': jnp.mean(jnp.abs(rnn_lambda1_value - V_mc)),
        'rnn_skip_lambda0': jnp.mean(jnp.abs(rnn_skip_lambda0_value - V_mc)),
        'rnn_skip_lambda1': jnp.mean(jnp.abs(rnn_skip_lambda1_value - V_mc)),
    }

    return value_distance


def make_collect(args: CollectHyperparams, key: chex.PRNGKey):
    # change relative paths to absolute paths
    args.dataset_path = Path(args.dataset_path).resolve()
    args.memoryless_lambda0_path = Path(args.memoryless_lambda0_path).resolve()
    args.memoryless_lambda1_path = Path(args.memoryless_lambda1_path).resolve()
    args.memoryless_skip_lambda0_path = Path(args.memoryless_skip_lambda0_path).resolve()
    args.memoryless_skip_lambda1_path = Path(args.memoryless_skip_lambda1_path).resolve()
    args.rnn_lambda0_path = Path(args.rnn_lambda0_path).resolve()
    args.rnn_lambda1_path = Path(args.rnn_lambda1_path).resolve()
    args.rnn_skip_lambda0_path = Path(args.rnn_skip_lambda0_path).resolve()
    args.rnn_skip_lambda1_path = Path(args.rnn_skip_lambda1_path).resolve()

    memoryless_lambda0_key, memoryless_lambda1_key, memoryless_skip_lambda0_key, memoryless_skip_lambda1_key, rnn_lambda0_key, rnn_lambda1_key, rnn_skip_lambda0_key, rnn_skip_lambda1_key, key = jax.random.split(key, 9)

    # load dataset
    if args.dataset_path.suffix == '.npy':
        restored = load_info(args.dataset_path)
    else:
        orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        restored = orbax_checkpointer.restore(args.dataset_path)
    
    dataset = restored['dataset']
    steps_to_collect = dataset['observation'].shape[0]
    
    _, _, memoryless_lambda0_args, memoryless_lambda0_network, memoryless_lambda0_ts = load_train_state(memoryless_lambda0_key, args.memoryless_lambda0_path,
                                                                                update_idx_to_take=args.update_idx_to_take,
                                                                                best_over_rng=True)
    _, _, memoryless_lambda1_args, memoryless_lambda1_network, memoryless_lambda1_ts = load_train_state(memoryless_lambda1_key, args.memoryless_lambda1_path,
                                                                                update_idx_to_take=args.update_idx_to_take,
                                                                                best_over_rng=True)
    _, _, memoryless_skip_lambda0_args, memoryless_skip_lambda0_network, memoryless_skip_lambda0_ts = load_train_state(memoryless_skip_lambda0_key, args.memoryless_skip_lambda0_path,
                                                                                update_idx_to_take=args.update_idx_to_take,
                                                                                best_over_rng=True)
    _, _, memoryless_skip_lambda1_args, memoryless_skip_lambda1_network, memoryless_skip_lambda1_ts = load_train_state(memoryless_skip_lambda1_key, args.memoryless_skip_lambda1_path,
                                                                                update_idx_to_take=args.update_idx_to_take,
                                                                                best_over_rng=True)
    _, _, rnn_lambda0_args, rnn_lambda0_network, rnn_lambda0_ts = load_train_state(rnn_lambda0_key, args.rnn_lambda0_path,
                                                                                update_idx_to_take=args.update_idx_to_take,
                                                                                best_over_rng=True)
    _, _, rnn_lambda1_args, rnn_lambda1_network, rnn_lambda1_ts = load_train_state(rnn_lambda1_key, args.rnn_lambda1_path,
                                                                                update_idx_to_take=args.update_idx_to_take,
                                                                                best_over_rng=True)
    _, _, rnn_skip_lambda0_args, rnn_skip_lambda0_network, rnn_skip_lambda0_ts = load_train_state(rnn_skip_lambda0_key, args.rnn_skip_lambda0_path,
                                                                                update_idx_to_take=args.update_idx_to_take,
                                                                                best_over_rng=True)
    _, _, rnn_skip_lambda1_args, rnn_skip_lambda1_network, rnn_skip_lambda1_ts = load_train_state(rnn_skip_lambda1_key, args.rnn_skip_lambda1_path,
                                                                                update_idx_to_take=args.update_idx_to_take,
                                                                                best_over_rng=True)

    _env_step = partial(ppo_step, dataset=dataset, memoryless_lambda0_network=memoryless_lambda0_network,
                        memoryless_lambda1_network=memoryless_lambda1_network,
                        memoryless_skip_lambda0_network=memoryless_skip_lambda0_network,
                        memoryless_skip_lambda1_network=memoryless_skip_lambda1_network,
                        rnn_lambda0_network=rnn_lambda0_network,
                        rnn_lambda1_network=rnn_lambda1_network,
                        rnn_skip_lambda0_network=rnn_skip_lambda0_network,
                        rnn_skip_lambda1_network=rnn_skip_lambda1_network)
    
    _env_step = scan_tqdm(steps_to_collect)(_env_step)

    ckpts = {
        'memoryless_lambda0': {'args': memoryless_lambda0_args, 'ts': memoryless_lambda0_ts, 'path': args.memoryless_lambda0_path},
        'memoryless_lambda1': {'args': memoryless_lambda1_args, 'ts': memoryless_lambda1_ts, 'path': args.memoryless_lambda1_path},
        'memoryless_skip_lambda0': {'args': memoryless_skip_lambda0_args, 'ts': memoryless_skip_lambda0_ts, 'path': args.memoryless_skip_lambda0_path},
        'memoryless_skip_lambda1': {'args': memoryless_skip_lambda1_args, 'ts': memoryless_skip_lambda1_ts, 'path': args.memoryless_skip_lambda1_path},
        'rnn_lambda0': {'args': rnn_lambda0_args, 'ts': rnn_lambda0_ts, 'path': args.rnn_lambda0_path},
        'rnn_lambda1': {'args': rnn_lambda1_args, 'ts': rnn_lambda1_ts, 'path': args.rnn_lambda1_path},
        'rnn_skip_lambda0': {'args': rnn_skip_lambda0_args, 'ts': rnn_skip_lambda0_ts, 'path': args.rnn_skip_lambda0_path},
        'rnn_skip_lambda1': {'args': rnn_skip_lambda1_args, 'ts': rnn_skip_lambda1_ts, 'path': args.rnn_skip_lambda1_path},
    }

    def collect(rng):
        # INIT ENV
        rng, _rng = jax.random.split(rng)
        # init hidden state
        init_rnn_lambda0_hstate = ScannedRNN.initialize_carry(1, rnn_lambda0_args['hidden_size'])
        init_rnn_lambda1_hstate = ScannedRNN.initialize_carry(1, rnn_lambda1_args['hidden_size'])
        init_rnn_skip_lambda0_hstate = ScannedRNN.initialize_carry(1, rnn_skip_lambda0_args['hidden_size'])
        init_rnn_skip_lambda1_hstate = ScannedRNN.initialize_carry(1, rnn_skip_lambda1_args['hidden_size'])
        init_runner_state = (
            0, # timestep
            memoryless_lambda0_ts,
            memoryless_lambda1_ts,
            memoryless_skip_lambda0_ts,
            memoryless_skip_lambda1_ts,
            rnn_lambda0_ts,
            rnn_lambda1_ts,
            rnn_skip_lambda0_ts,
            rnn_skip_lambda1_ts,
            init_rnn_lambda0_hstate,
            init_rnn_lambda1_hstate,
            init_rnn_skip_lambda0_hstate,
            init_rnn_skip_lambda1_hstate,
            _rng,
        )

        runner_state, value_distance_dataset = jax.lax.scan(
            _env_step, init_runner_state, jnp.arange(steps_to_collect), steps_to_collect
        )

        return value_distance_dataset

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

    print(dataset['memoryless_lambda0_value'][1:100])
    print(dataset['memoryless_lambda1_value'][1:100])
    print(dataset['memoryless_skip_lambda0_value'][1:100])
    print(dataset['memoryless_skip_lambda1_value'][1:100])
    print(dataset['rnn_lambda0_value'][1:100])
    print(dataset['rnn_lambda1_value'][1:100])
    print(dataset['rnn_skip_lambda0_value'][1:100])
    print(dataset['rnn_skip_lambda1_value'][1:100])
    print(dataset['V_mc'][1:100])
    # value_distance = calculate_value_distance(dataset)
    # print(value_distance)   

    to_save = {
        'dataset': dataset,
        # 'value_distance': value_distance,
        'args': args.as_dict(),
        'ckpt': ckpt_info,
    }
    path_to_str(to_save)

    save_path = args.dataset_path.parent.parent / \
                f'value_distance'

    # Save all results with Orbax
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    save_args = orbax_utils.save_args_from_target(to_save)

    print(f"Saving results to {save_path}")
    orbax_checkpointer.save(save_path, to_save, save_args=save_args)

    print("Done.")