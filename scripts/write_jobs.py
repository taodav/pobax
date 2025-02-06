"""
Script to convert hyperparams/XYZ.py files into a .txt file
where every line of the .txt file is one experiment.
"""
import argparse
import numpy as np
from typing import List, Iterable
from pathlib import Path
from itertools import product

from pobax.utils.file_system import import_module_to_var

from definitions import ROOT_DIR


def generate_runs(run_dicts: List[dict],
                  runs_dir: Path,
                  experiment_name: str = None,
                  runs_fname: str = 'runs.txt',
                  main_fname: str = 'run.py',
                  exclude_dict: dict = None) -> None:
    """
    :param run_dicts: A list of dictionaries, each specifying a job to run.
    :param runs_dir: Directory to put the runs
    :param runs_fname: What do we call our run file?
    :param main_fname: what is our python entry script?
    :return: nothing. We write to runs_dir/runs_fname
    """

    runs_path = runs_dir / runs_fname

    if runs_path.is_file():
        runs_path.unlink()

    f = open(runs_path, 'a+')

    num_runs = 0
    for run_dict in run_dicts:
        keys, values = [], []

        for k, v in run_dict.items():
            keys.append(k)
            if not (isinstance(v, list) or isinstance(v, np.ndarray)):
                v = [v]
            values.append(v)

        for i, args in enumerate(product(*values)):

            arg = {k: v for k, v in zip(keys, args)}

            if exclude_dict is not None:
                exclude = True
                for ek, ev_list in exclude_dict.items():
                    assert ek in arg, f'Argument {ek} not in args dict.'
                    if not isinstance(ev_list, Iterable):
                        ev_list = [ev_list]

                    exclude &= (arg[ek] in ev_list)

                if exclude:
                    continue

            run_string = f"python {main_fname}"

            for k, v in arg.items():

                if v is True:
                    run_string += f" --{k}"
                elif v is False or v is None:
                    continue
                else:
                    if isinstance(v, list):
                        v = ' '.join(v)
                    run_string += f" --{k} {v}"

            if experiment_name is not None and 'study_name' not in run_dict:
                run_string += f" --study_name {experiment_name}"

            run_string += "\n"
            f.write(run_string)
            num_runs += 1

            print(num_runs, run_string)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('hyperparam_file', type=str)
    parser.add_argument('--local', action='store_true')
    args = parser.parse_args()

    runs_dir = Path(ROOT_DIR, 'scripts', 'runs')

    hparam_path = Path(args.hyperparam_file).resolve()
    hparams = import_module_to_var(hparam_path, 'hparams')

    results_dir = Path(ROOT_DIR, 'results')
    # if not args.local:
    #     # Here we assume we want to write to the scratch directory in CC.
    #     results_dir = Path("/home/taodav/scratch/uncertainty/results")

    # Make our directories if it doesn't exist
    runs_dir.mkdir(parents=True, exist_ok=True)

    main_fname = 'run.py'
    if 'entry' in hparams:
        main_fname = hparams['entry']

    exclude_dict = None
    if 'exclude' in hparams:
        exclude_dict = hparams['exclude']

    generate_runs(hparams['args'],
                  runs_dir,
                  runs_fname=hparams['file_name'],
                  main_fname=main_fname,
                  experiment_name=hparam_path.stem,
                  exclude_dict=exclude_dict)

    print(f"Runs wrote to {runs_dir / hparams['file_name']}")