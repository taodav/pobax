import argparse
import pickle
from pathlib import Path

import numpy as np

def argmax_last(array):
    shape = array.shape
    array = array.reshape((-1, shape[-1]))
    ravelmax = np.argmax(array, axis=0)
    return np.unravel_index(ravelmax, shape[:-1]) + (np.arange(shape[-1]),)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('parsed_study_path', type=str)
    parser.add_argument('--measure', type=str, default='auc')
    args = parser.parse_args()

    # The math says that this doesn't matter...
    mean_order = ['seeds', 'num_update', 'num_steps']

    parsed_path = Path(args.parsed_study_path)

    with open(parsed_path, 'rb') as f:
        parsed_res = pickle.load(f)
    scores = parsed_res['scores']
    # HACK to take care of nans
    scores[np.isnan(scores)] = (-10000)
    # scores = fill_in_first_n_with_zeros(scores)
    # leave_first_n = 10
    # scores = scores[..., leave_first_n:, :, :, :]

    all_hyperparams = parsed_res['hyperparams']

    # SELECING HYPERPARAMS
    # optimize_lambda_discrep = 1  # False
    # alpha = 0  # 1
    # preselected_hyperparams = {
    #     'optimize_lambda_discrep': False,
    #     'alpha': 1.
    # }
    # scores = scores[optimize_lambda_discrep, alpha]

    mean_score = scores
    changing_mean_order = parsed_res['dim_ref'].copy()
    for axis_name in mean_order:
        axis = changing_mean_order.index(axis_name)
        mean_score = mean_score.mean(axis=axis)
        changing_mean_order.remove(axis_name)

    max_idxes = argmax_last(mean_score)

    swapped = scores.swapaxes(len(max_idxes) - 1, -1)
    max_scores = swapped[max_idxes].swapaxes(0, -1)

    best_hyperparams = {}
    for env, env_max_idx in zip(parsed_res['envs'], np.stack(max_idxes[:-1], axis=-1)):
        env_best_hparam = {}
        for idx, k in zip(env_max_idx, all_hyperparams):
            env_best_hparam[k] = all_hyperparams[k][idx]
        best_hyperparams[env] = env_best_hparam

    best_hparam_res = {
        'hyperparams': best_hyperparams,
        'scores': max_scores,
        'dim_ref': parsed_res['dim_ref'][len(max_idxes) - 1:],
        'envs': parsed_res['envs'],
        'all_hyperparams': parsed_res['all_hyperparams'],
        'discounted': parsed_res['discounted']
    }

    file_name = "best_hyperparam_per_env_res.pkl"
    if best_hparam_res['discounted']:
        file_name = "best_hyperparam_per_env_res_discounted.pkl"
    best_hparam_path = parsed_path.parent / file_name
    with open(best_hparam_path, 'wb') as f:
        pickle.dump(best_hparam_res, f)

    print(f"Saved best hyperparams to {best_hparam_path}.")

