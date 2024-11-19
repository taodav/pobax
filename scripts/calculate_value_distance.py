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

from collections import defaultdict
from gymnax.environments import environment
from pobax.models.network import ScannedRNN
from pobax.utils.file_system import load_train_state, load_info, numpyify_and_save
from train_value_approximator import ValueNetwork
from flax.training.train_state import TrainState
import optax

class CalculateValueHyperparams(Tap):
    dataset_path: Union[str, Path]
    value_network_path: Union[str, Path]

    update_idx_to_take: int = None

    gamma: float = 0.99

    seed: int = 2024
    platform: str = 'gpu'
    study_name: str = 'test'

    def configure(self) -> None:
        self.add_argument('--value_network_path', type=Path)
        self.add_argument('--dataset_path', type=Path)


def collect_step(runner_state, unused,
                    dataset, network):
    
    def get_state(s):
        if hasattr(s, 'env_state'):
            return get_state(s.env_state)
        else:
            return s

    (ts, timestep, hstate, rng) = runner_state
    rng, _rng = jax.random.split(rng)

    last_obs = dataset['observation'][timestep]
    last_done = dataset['done'][timestep]
    last_obs = jnp.expand_dims(last_obs, axis=0)
    last_done = jnp.expand_dims(last_done, axis=0)
    # SELECT ACTION
    ac_in = (last_obs[np.newaxis, :], last_done[np.newaxis, :])
    _, value = network.apply(ts.params, hstate, ac_in)
    value = value.squeeze(0)

    datum = {
        'v_mc': dataset['V'][timestep],
        'value': value,
        'distance': jnp.abs(dataset['V'][timestep] - value),
    }
    runner_state = (ts, timestep+1, hstate, rng)
    return runner_state, datum

def calculate_value_distance(v_mc, value):
    return jnp.mean(jnp.abs(v_mc - value))

def make_collect(args: CalculateValueHyperparams, key: chex.PRNGKey):

    network_key, key = jax.random.split(key, 2)

    args.dataset_path = Path(args.dataset_path).resolve()
    args.value_network_path = Path(args.value_network_path).resolve()

    # load dataset
    if args.dataset_path.suffix == '.npy':
        dataset_restored = load_info(args.dataset_path)
    else:
        orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        dataset_restored = orbax_checkpointer.restore(args.dataset_path)
    
    # load value network
    if args.dataset_path.suffix == '.npy':
        value_restored = load_info(args.value_network_path)
    else:
        orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        value_restored = orbax_checkpointer.restore(args.value_network_path)

    dataset = dataset_restored['dataset']
    steps_to_collect = dataset['observation'].shape[0]
    network_args = value_restored['args']
    network = ValueNetwork(approximator=network_args['approximator'], hidden_size=network_args['hidden_size'], n_hidden_layers=network_args['n_hidden_layers'])
    unpacked_ts = value_restored['final_train_state']
    args.target = network_args['target']
    args.approximator = network_args['approximator']
    tx = optax.adam(network_args['lr'])
    params = unpacked_ts['params']
    ts = TrainState.create(apply_fn=network.apply,
                           params=params,
                           tx=tx)


    _env_step = partial(collect_step, dataset=dataset, network=network)
    _env_step = scan_tqdm(steps_to_collect)(_env_step)

    ckpts = {
        'network': {'args': network_args, 'ts': ts, 'path': args.value_network_path},
    }

    def collect(rng):
        # INIT ENV
        rng, _rng = jax.random.split(rng)

        # init hidden state
        init_hstate = ScannedRNN.initialize_carry(1, network_args['hidden_size'])
        init_runner_state = (
            ts,
            0,
            init_hstate,
            _rng,
        ) 

        runner_state, value_dict = jax.lax.scan(
            _env_step, init_runner_state, jnp.arange(steps_to_collect), steps_to_collect
        )

        return value_dict

    return collect, ckpts


if __name__ == "__main__":
    # jax.disable_jit(True)
    args = CalculateValueHyperparams().parse_args()
    jax.config.update('jax_platform_name', args.platform)

    key = jax.random.PRNGKey(args.seed)
    make_key, collect_key, key = jax.random.split(key, 3)

    collect_fn, ckpt_info = make_collect(args, make_key)
    collect_fn = jax.jit(collect_fn)

    value_dict = collect_fn(collect_key)

    def path_to_str(d: dict):
        for k, v in d.items():
            if isinstance(v, Path):
                d[k] = str(v)
            elif isinstance(v, dict):
                path_to_str(v)


    to_save = {
        'value_info': value_dict,
        'distance': jnp.mean(value_dict['distance'], axis=0),
        'args': args.as_dict(),
        # 'ckpt': ckpt_info,
    }
    print(value_dict['distance'].shape)
    print(to_save['distance'])
    path_to_str(to_save)
    save_path = args.value_network_path.parent / \
                f'distance_{args.approximator}_{args.target}'
    
    numpyify_and_save(save_path, to_save)
    print("Done.")



