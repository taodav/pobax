import argparse
from collections import OrderedDict
import importlib
from pathlib import Path
import pickle
import re
import sys

import jax
import jax.numpy as jnp
import orbax.checkpoint
import numpy as np
from tqdm import tqdm

from definitions import ROOT_DIR
from pobax.utils.file_system import load_info


def get_total_size(obj, seen=None):
    """Recursively finds size of objects in bytes."""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()

    # Add object's id to seen to avoid double counting of objects
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    seen.add(obj_id)

    if isinstance(obj, dict):
        size += sum((get_total_size(k, seen) + get_total_size(v, seen)) for k, v in obj.items())
    elif isinstance(obj, (list, tuple, set, frozenset)):
        size += sum(get_total_size(item, seen) for item in obj)
    elif isinstance(obj, np.ndarray):
        # Numpy array, account for the full memory usage of the array
        size += obj.nbytes

    return size


def combine_seeds_and_envs(x: jnp.ndarray):
    # Here, dim=-1 is the NUM_ENVS parameter. We take the mean over this.
    # dim=-2 is the NUM_STEPS parameter.
    # dim=-3 is the NUM_UPDATES, which is TOTAL_TIMESTEPS // NUM_STEPS // NUM_ENVS.
    # dim=-4 is n_seeds.
    # We take the mean and std_err to the mean over dimensions -1 and -4.
    envs_seeds_swapped = jnp.swapaxes(x, -2, -4).swapaxes(-3, -4)

    # We take the mean over NUM_ENVS dimension.
    mean_over_num_envs = envs_seeds_swapped.mean(axis=-1)
    return mean_over_num_envs


def get_first_returns(returned_episode: jnp.ndarray, returns: jnp.ndarray):
    first_episode_ends = returned_episode.argmax(axis=-2)

    mesh_inputs = [np.arange(dim) for dim in first_episode_ends.shape]
    grids = np.meshgrid(*mesh_inputs, indexing='ij')
    grids_before, grids_after = grids[:-1], grids[-1:]
    return returns[(*grids_before, first_episode_ends, *grids_after)]


def get_final_eval(final_eval: dict):
    # Get final eval metrics
    disc_returns = final_eval['returned_discounted_episode_returns']

    first_episode_ends = final_eval['returned_episode'].argmax(axis=-2)

    mesh_inputs = [np.arange(dim) for dim in first_episode_ends.shape]
    grids = np.meshgrid(*mesh_inputs, indexing='ij')
    grids_before, grids_after = grids[:-1], grids[-1:]
    final_first_disc_returns = disc_returns[(*grids_before, first_episode_ends, *grids_after)]
    return final_first_disc_returns


def parse_exp_dir(study_path, study_hparam_path):
    # TODO: THIS
    train_sign_hparams = ['vf_coeff', 'lambda0', 'lr']

    study_paths = list(study_path.iterdir())

    scores, final_scores, envs, best_hyperparams = [], [], [], {}
    for results_path in tqdm(study_paths):
        restored = load_info(results_path)

        config, args = restored['config'], restored['args']

        # Get online metrics
        online_eval = restored['out']['metric']
        online_disc_returns = online_eval['returned_discounted_episode_returns']

        final_eval = restored['out']['final_eval_metric']
        # we take the mean over axis=-2 here, since this dimension might be different
        # for the final eval.
        final_n_episodes = final_eval['returned_episode'].sum(axis=-2, keepdims=True)
        final_disc_returns = final_eval['returned_discounted_episode_returns'].sum(axis=-2, keepdims=True)
        final_disc_returns /= (final_n_episodes + (final_n_episodes == 0).astype(float))  # add the 0 mask to prevent division by 0.

        # we add a num_updates dimension
        final_disc_returns = np.expand_dims(final_disc_returns, -3)

        del restored
        seeds_combined = combine_seeds_and_envs(online_disc_returns)

        final_seeds_combined = combine_seeds_and_envs(final_disc_returns)
        scores.append(seeds_combined[0, 0, 0])
        final_scores.append(final_seeds_combined[0, 0, 0])
        envs.append(args['env'])
        best_hyperparams[args['env']] = {k: args[k] for k in train_sign_hparams}


    dim_ref = ['vf_coeff', 'lambda0', 'lr', 'num_update', 'num_steps', 'seeds', 'env']

    parsed_res = {
        'envs': envs,
        'scores': np.stack(scores, axis=-1),
        'final_scores': np.stack(final_scores, axis=-1),
        'dim_ref': dim_ref,
        'hyperparams': best_hyperparams,
        'hidden_size': args['hidden_size']
    }
    return parsed_res


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('study_path', type=str)
    args = parser.parse_args()

    study_path = Path(args.study_path)
    study_hparam_path = Path(ROOT_DIR, 'scripts', 'hyperparams', study_path.stem + '.py')

    parsed_res_path = study_path / "best_hyperparam_per_env_res.pkl"

    parsed_res = parse_exp_dir(study_path, study_hparam_path)

    print(f"Saving parsed results to {parsed_res_path}")
    with open(parsed_res_path, 'wb') as f:
        pickle.dump(parsed_res, f, protocol=4)