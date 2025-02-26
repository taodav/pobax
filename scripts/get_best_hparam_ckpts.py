from pathlib import Path
import pickle

import numpy as np
import orbax.checkpoint
from tqdm import tqdm

from definitions import ROOT_DIR

if __name__ == "__main__":
    ckpt_path = Path(ROOT_DIR, 'results/pendulum/pendulum_v_ppo/best_hyperparam_per_env_res.pkl')

    parent_dir = ckpt_path.parent
    with open(ckpt_path, "rb") as f:
        best_res = pickle.load(f)

    env = best_res['envs'][0]
    assert len(best_res['envs']) == 1

    hparam_index_ref = {}
    final_tstate = None
    study_paths = [s for s in parent_dir.iterdir() if s.is_dir()]
    for results_path in tqdm(study_paths):
        orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()

        restored = orbax_checkpointer.restore(results_path)
        args = restored['args']
        right_res = True
        for hparam, val in best_res['hyperparams'][env].items():
            if val not in args[hparam]:
                right_res = False
                break
            else:
                hparam_index_ref[hparam] = np.where(args[hparam] == val)[0].item()

        if not right_res:
            hparam_index_ref = {}
            continue

        final_tstate = restored['final_train_state']



    print()
