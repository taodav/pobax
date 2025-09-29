import argparse
import pickle
from pathlib import Path

import numpy as np


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('parsed_study_path', type=str)
    parser.add_argument('--measure', type=str, default='auc',
                        help='What measure do we use? (auc | final)')
    args = parser.parse_args()

    # The math says that this doesn't matter...
    mean_order = ['seeds', 'num_update', 'num_steps']

    parsed_path = Path(args.parsed_study_path)

    with open(parsed_path, 'rb') as f:
        parsed_res = pickle.load(f)
    scores = parsed_res['scores']

    best_hyperparams = {}
    max_scores = {}
    best_fpaths = {}

    for env, results_dict in scores.items():
        if args.measure == 'auc':
            score = results_dict['scores']
        elif args.measure == 'final':
            score = results_dict['final_scores']
        else:
            raise NotImplementedError

        mean_score = score
        changing_mean_order = parsed_res['dim_ref'].copy()
        for axis_name in mean_order:
            axis = changing_mean_order.index(axis_name)
            mean_score = mean_score.mean(axis=axis)
            changing_mean_order.remove(axis_name)

        assert len(mean_score.shape) == 1
        max_idx = np.argmax(mean_score)
        max_score = score[max_idx]

        best_hyperparams[env] = {}
        for k, vals in parsed_res['swept_hyperparams'][env].items():
            best_hyperparams[env][k] = vals[max_idx]
        best_fpaths[env] = {'fpath': results_dict['fpaths'], 'max_idx': max_idx}

        max_scores[env] = max_score

    best_hparam_res = {
        'hyperparams': best_hyperparams,
        'scores': max_scores,
        'dim_ref': parsed_res['dim_ref'][1:],
        'envs': parsed_res['envs'],
        'all_hyperparams': parsed_res['all_hyperparams'],
        'discounted': parsed_res['discounted'],
        'fpaths': best_fpaths
    }

    file_name = "best_hyperparam_per_env_res.pkl"
    if best_hparam_res['discounted']:
        file_name = "best_hyperparam_per_env_res_discounted.pkl"
    best_hparam_path = parsed_path.parent / file_name
    with open(best_hparam_path, 'wb') as f:
        pickle.dump(best_hparam_res, f)

    print(f"Saved best hyperparams to {best_hparam_path}.")

