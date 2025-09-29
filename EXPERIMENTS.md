# Running experiments

All our experiment hyperparameters are defined in the `scripts/hyperparams` directory.
Each file in this directory contains a Python dictionary that specifies the entry point,
and arguments to sweep over.

Arguments listed as a list will be swept over. There are two ways to sweep over arguments:
1. Pass all swept-over hyperparameters to JAX and let `vmap` sweep over these hyperparameters.
2. Create a new run script for each hyperparameter listed in the list.

We would obviously like to use 1. as much as possible and let JAX sweep over any hyperparameters
it can, seeing as `vmap` uses much less compute resources than running a separate program.
But alas there are hyperparameters that _cannot_ be `vmap`ed over easily. For example, the
neural network `hidden_size` hyperparameter cannot be `vmap`ed over, since this changes the 
sizes of our tensors across each run with a different `hidden_size`. Another reason to not use
`vmap` to sweep over hyperparameters is lack of GPU VRAM. If your experiments need too much memory,
you can split your jobs up with 2.

Take **special note** that list arguments `.join`ed into a string will be `vmap`ed over,
whereas Python lists will be swept over as separate programs as in 2.

**NOTE: Currently Madrona (visual Mujoco) runs do not work with sweeping over `vmap`.**

To run experiments defined in the `hyperparams` directory, we have
two methods for writing experiment run program scripts. 

### Option 1: a `.txt` file full of runs.
you need to write jobs according to the `write_jobs.py` script
in this directory. For example, in the `scripts` directory:

```shell
python write_jobs.py hyperparams/tmaze/tmaze_10_ppo.py
```

This will create a `runs` file with `runs_tmaze_10_ppo.txt`, where
each line is an experiment run.

To run jobs on a slurm-based cluster, you first go to the launching directory and revise single_slurm_job.sh file
line 15 `input_file="../runs/runs_tmaze_10_ppo.txt"`. Then you can
call

```shell
cd launching
nano single_slurm_job.sh
sbatch single_slurm_job.sh
```

### Option 2: `onager`

[Onager](https://github.com/camall3n/onager) is an experiment running library developed in our lab. You can
install it with
```shell
pip install onager
```

With this package installed, you can launch `prelaunch` your `tmaze_10_ppo` job by first
changing directory into the `scripts` directory, and running
```shell
python onager_write_jobs.py hyperparams/tmaze/tmaze_10_ppo.py
```
This will prelaunch an experiment with the job name `tmaze_10_ppo`.

Even more convenient is prelaunching all experiments in a directory with `directory_prelaunch.sh`; 
simply change `PRELAUNCH_DIR` to point to the hyperparameter directory you want (in our case `hyperparams/tmaze`, 
and run:
```shell
./directory_prelaunch.sh
```
This will prelaunch the jobs 
```
tmaze_10_ppo tmaze_10_ppo_LD tmaze_10_ppo_memoryless tmaze_10_ppo_perfect_memory_memoryless tmaze_10_transformer
```

Now to launch prelaunched experiments, you can use the script `scripts/launching/onager_slurm_launch.sh`: just
change the `PRELAUNCH_NAMES` variables with all the job names you want to launch, and run:
```shell
./launching/onager_slurm_launch.sh
```
to run the job on slurm-based cluster.

## Parsing job results

After finishing a job, the results should be in the `results`
folder in the root of the project directory. in our example, we should have a directory
named `results/tmaze_10_ppo` with all our results.

To parse our results, navigate back to `scripts` first. 
Then we combine all our results into a single tensor with
```shell
python parse_batch_experiments.py ../results/tmaze_10_ppo
```
NOTE: for this script to work, your experiment hyperparameters
have to be defined in `{JOB_NAME}.py` in the `scripts/hyperparams` directory.

This should create the file `tmaze_10_ppo/parsed_hparam_scores.pkl` file. Add the `--discounted` 
flag to calculate discounted returns instead.

Finally, to get the best hyperparameters, pass this file to `scripts/best_hyperparams_per_env.py`:
```shell
python best_hyperparams_per_env.py 
```
Which should make a `best_hyperparam_per_env_res.pkl` file, or `best_hyperparam_per_env_res_discounted.pkl` if
the results were discounted. This file is a dictionary with the following fields:
```python
{
    'hyperparams': best_hyperparams,  # The best hyperparameters swept and selected
    'scores': max_scores,  # The scores across time for each of these runs
    'dim_ref': parsed_res['dim_ref'][1:],  # A reference for each dimension of the `scores` tensor
    'envs': parsed_res['envs'],  # Environments run
    'all_hyperparams': parsed_res['all_hyperparams'],  # All the hyperparameters needed for the runs
    'discounted': parsed_res['discounted'],  # Are these discounted results or not?
    'fpaths': best_fpaths  # The best files from the selected hyperparameters
}
```
## Useful scripts

### Syncing files from a remote server with `rsync`

Pull everything from `directory/to/results` into current directory,
excluding directories that match "*_seed*":
```shell
rsync -zLurP --exclude "*_seed*/" src:"directory/to/results" ./
```
