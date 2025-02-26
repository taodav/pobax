from pathlib import Path
import re
import shutil

import numpy as np
import orbax.checkpoint
from tqdm import tqdm

from definitions import ROOT_DIR

labels = {
    'hidden_size': 'hsize',
    'num_envs': 'nenvs'
}


if __name__ == "__main__":
    hparam = 'hidden_size'
    parent_dir = Path(ROOT_DIR, 'results/walker_v_hsize_sweep')
    study_hparam_parent_dir = Path(ROOT_DIR, 'scripts/hyperparams/masked_mujoco/ablation')

    study_paths = [s for s in parent_dir.iterdir() if s.is_dir()]
    for study_path in tqdm(study_paths):

        for results_path in study_path.iterdir():
            if not results_path.is_dir():
                continue
            orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()

            restored = orbax_checkpointer.restore(results_path)
            args = restored['args']

            new_study_name = f'{study_path.name}_{labels[hparam]}_{args[hparam]}'
            new_dir = study_path / new_study_name
            new_dir.mkdir(exist_ok=True)

            new_path = new_dir / results_path.name
            shutil.move(results_path, new_path)

            # now we make the new hparam file
            study_hparam_file = study_hparam_parent_dir / f'{study_path.name}.py'
            new_study_hparam_file = study_hparam_parent_dir / 'indv' / f'{new_study_name}.py'
            new_study_hparam_file.parent.mkdir(exist_ok=True)
            if not new_study_hparam_file.exists():
                with open(study_hparam_file, 'r') as infile:
                    content = infile.read()
                pattern = rf"('{hparam}':).*?(,\n)"
                replacement = rf"\1 {args[hparam]}\2"

                new_content = re.sub(pattern, replacement, content)

                with open(new_study_hparam_file, 'w') as outfile:
                    outfile.write(new_content)
                print(f"written new hyperparams to {new_study_hparam_file}")



    print(f"Split files from {parent_dir}")