from collections import OrderedDict
import hashlib
import importlib
import inspect
from pathlib import Path
import sys
import time
from typing import Union
from types import FunctionType, CellType

import jax
import jax.numpy as jnp
import numpy as np
from flax.training.train_state import TrainState
import optax
import orbax.checkpoint

from pobax.envs import get_env
from pobax.models import get_gymnax_network_fn
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

    network_fn, action_size = get_gymnax_network_fn(env, env_params, memoryless=args['memoryless'])

    network = network_fn(action_size,
                         double_critic=args['double_critic'],
                         hidden_size=args['hidden_size'])
    tx = optax.adam(args['lr'][0])
    ts = TrainState.create(apply_fn=network.apply,
                           params=jax.tree_map(lambda x: x[0, 0, 0, 0, 0, 0], unpacked_ts['params']),
                           tx=tx)

    return env, env_params, args, network, ts


def get_fn_from_module(entry: str, fn_name: str = 'make_train'):
    """
    Gets a function based off of an entry string, and a function name.
    :param entry: string (without the python call) of the entrypoint. So for example, 'batch_run_ppo.py' for a call to
                  python batch_run_ppo.py, or '-m pobax.algos.ppo' for python -m pobax.algos.ppo
    :param fn_name: name of the function we want to load in the module
    :return: the function in the module.
    """
    if entry.startswith('-m'):
        module_entry = entry.split(' ')[-1]
        # Load the module
        module = importlib.import_module(module_entry)
    else:
        # assume here that the entry point is the project root
        assert entry.endswith('.py')
        fpath = Path(ROOT_DIR, entry)
        module_name = fpath.stem

        # Create a module spec
        spec = importlib.util.spec_from_file_location(module_name, fpath)

        # Create a module from the spec
        module = importlib.util.module_from_spec(spec)

        # Execute the module
        spec.loader.exec_module(module)

        # Add the module to sys.modules
        sys.modules[module_name] = module

    # Get the function from the module
    fn = getattr(module, fn_name)
    return fn


def get_inner_fn_arguments(fn: FunctionType, inner_fn_name: str = 'train'):
    # Get the code object of the outer function
    # outer_code = outer_function.__code__
    outer_code = fn.__code__

    # Extract the constants from the outer function's code object
    constants = outer_code.co_consts

    # Find the nested function within the constants
    nested_func_code = None
    for const in constants:
        # if inspect.iscode(const) and const.co_name == 'nested_function':
        if inspect.iscode(const) and const.co_name == inner_fn_name:
            nested_func_code = const
            break

    # Dummy closure for free variables
    dummy_closure = tuple(CellType() for _ in nested_func_code.co_freevars)

    # Create a function object from the nested function's code object
    nested_function = FunctionType(nested_func_code, globals(), inner_fn_name, None, dummy_closure)

    # Get the arguments of the nested function
    args = inspect.signature(nested_function).parameters
    return list(args.keys())
