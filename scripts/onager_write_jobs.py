"""
Converts and runs a specified hyperparams/XYZ.py hyperparam file
into an Onager prelaunch script.
"""
import os
import numpy as np
import argparse
from typing import List
from pathlib import Path

from pobax.utils.file_system import import_module_to_var

from definitions import ROOT_DIR


def generate_onager_runs(run_dicts: List[dict],
                         experiment_name: str,
                         main_fname: str = 'run.py',
                         exclude: dict = None) -> None:
    """
    :param run_dicts: A list of dictionaries, each specifying a job to run.
    Each run_dict in this list corresponds to one call of `onager prelaunch`
    :param main_fname: what is our python entry script?
    :return: nothing. We execute an onager prelaunch script
    """
    onager_prelaunch = "onager prelaunch"
    jobname = f"+jobname {experiment_name}"

    for i, run_dict in enumerate(run_dicts):
        command = f"python {main_fname}"
        if len(run_dicts) > 1:
            jobname += f"_{i}"

        prelaunch_list = [onager_prelaunch, jobname]
        arg_list = []

        for k, v in run_dict.items():
            if v is None:
                continue

            if not (isinstance(v, list) or isinstance(v, np.ndarray)):
                if isinstance(v, bool):
                    if v:
                        command += f" --{k}"
                    else:
                        continue
                else:
                    command += f" --{k} {v}"
            else:
                if all([isinstance(el, bool) for el in v]):
                    arg_string = f"+flag --{k}"
                else:
                    assert not isinstance(v[0], list), "No functionality for passing in a list of lists." \
                                                       "Please pass in a list of space-separated strings instead."
                    arg_string = f"+arg --{k} {' '.join(map(str, v))}"
                arg_list.append(arg_string)

        command += f" --study_name {experiment_name}"

        prelaunch_list.append(f'+command "{command}"')

        if exclude is not None:
            for k, v in exclude.items():
                if not isinstance(v, list):
                    v = [v]
                prelaunch_list.append(f"+exclude --{k} {' '.join(map(str, v))}")

        prelaunch_list += arg_list

        prelaunch_string = ' '.join(prelaunch_list)
        print(f"Launching prelaunch script: {prelaunch_string}")
        os.chdir(ROOT_DIR)
        os.system(prelaunch_string)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('hyperparam_file', type=str)
    parser.add_argument('--study_name', default=None, type=str)
    parser.add_argument('--local', action='store_true')
    args = parser.parse_args()

    hparam_path = Path(args.hyperparam_file).resolve()
    hparams = import_module_to_var(hparam_path, 'hparams')

    main_fname = 'run.py'
    if 'entry' in hparams:
        main_fname = hparams['entry']

    exclude = None
    if 'exclude' in hparams:
        exclude = hparams['exclude']

    exp_name = hparam_path.stem
    if args.study_name is not None:
        exp_name = args.study_name

    generate_onager_runs(hparams['args'], exp_name, main_fname=main_fname, exclude=exclude)
