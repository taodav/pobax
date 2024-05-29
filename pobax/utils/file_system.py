from collections import OrderedDict
import hashlib
import importlib
from pathlib import Path
import time
from typing import Union

import jax
import jax.numpy as jnp
import numpy as np
from flax.training.train_state import TrainState
import optax
import orbax.checkpoint

from pobax.envs import get_env
from pobax.models import get_network_fn
from pobax.config import Hyperparams

from definitions import ROOT_DIR


def get_results_path(args: Hyperparams, return_npy: bool = True):
    results_dir = Path(ROOT_DIR, 'results')
    results_dir.mkdir(exist_ok=True)

    args_hash = make_hash_md5(args.as_dict())
    time_str = time.strftime("%Y%m%d-%H%M%S")

    if args.study_name is not None:
        results_dir /= args.study_name
    results_dir.mkdir(exist_ok=True)
    results_path = results_dir / f"{args.env}_seed({args.seed})_time({time_str})_{args_hash}{'.npy' if return_npy else ''}"
    return results_path


def make_hash_md5(o):
    return hashlib.md5(str(o).encode('utf-8')).hexdigest()


def numpyify_dict(info: Union[dict, OrderedDict, jnp.ndarray, np.ndarray, list, tuple]):
    """
    Converts all jax.numpy arrays to numpy arrays in a nested dictionary.
    """
    if isinstance(info, jnp.ndarray):
        return np.array(info)
    elif isinstance(info, dict):
        return {k: numpyify_dict(v) for k, v in info.items()}
    elif isinstance(info, OrderedDict):
        return OrderedDict([(k, numpyify_dict(v)) for k, v in info.items()])
    elif isinstance(info, list):
        return [numpyify_dict(i) for i in info]
    elif isinstance(info, tuple):
        return tuple(numpyify_dict(i) for i in info)

    return info


def numpyify_and_save(path: Path, info: Union[dict, jnp.ndarray, np.ndarray, list, tuple]):
    numpy_dict = numpyify_dict(info)
    np.save(path, numpy_dict)


def import_module_to_var(fpath: Path, var_name: str) -> Union[dict, list]:
    spec = importlib.util.spec_from_file_location(var_name, fpath)
    var_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(var_module)
    instantiated_var = getattr(var_module, var_name)
    return instantiated_var


def load_info(results_path: Path) -> dict:
    return np.load(results_path, allow_pickle=True).item()


def load_train_state(key: jax.random.PRNGKey, fpath: Path):
    # load our params
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    restored = orbax_checkpointer.restore(fpath)
    args = restored['args']
    unpacked_ts = restored['out']['runner_state'][0]


    env, env_params = get_env(args['env'], key,
                              args['gamma'],
                              action_concat=args['action_concat'])

    network_fn, action_size = get_network_fn(env, env_params, memoryless=args['memoryless'])

    network = network_fn(action_size,
                         double_critic=args['double_critic'],
                         hidden_size=args['hidden_size'])
    tx = optax.adam(args['lr'][0])
    ts = TrainState.create(apply_fn=network.apply,
                           params=jax.tree_map(lambda x: x[0, 0, 0, 0, 0, 0], unpacked_ts['params']),
                           tx=tx)

    return env, env_params, args, network, ts

