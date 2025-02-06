from pathlib import Path
import pickle

import numpy as np

from definitions import ROOT_DIR


if __name__ == "__main__":
    paths = [
        Path(ROOT_DIR, 'results', f'masked_mujoco_ppo', 'parsed_hparam_scores.pkl'),
        Path(ROOT_DIR, 'results', f'masked_mujoco_ppo2', 'parsed_hparam_scores.pkl'),
    ]

    all_scores = []
    all_final_scores = []

    for results_path in paths[::-1]:
        with open(results_path, 'rb') as f:
            parsed_res = pickle.load(f)

        scores, final_scores = parsed_res['scores'], parsed_res['final_scores']
        if all_scores:
            assert scores.shape == all_scores[-1].shape
        all_scores.append(scores)
        all_final_scores.append(final_scores)

    seeds_dim = parsed_res['dim_ref'].index('seeds')

    combined_scores = np.concatenate(all_scores, axis=seeds_dim)
    combined_final_scores = np.concatenate(all_final_scores, axis=-2)

    parsed_res['scores'] = combined_scores
    parsed_res['final_scores'] = combined_final_scores
    parsed_res['parent_paths'] = paths

    new_path = results_path.parent / 'combined_parsed_hparam_scores.pkl'

    print(f"Saving combined parsed results to {new_path}")
    with open(new_path, 'wb') as f:
        pickle.dump(parsed_res, f, protocol=4)

    print("Done")
