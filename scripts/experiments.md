# Running experiments

To run experiments defined in the `hyperparams` directory,
you need to write jobs according to the `write_jobs.py` script
in this directory. For example, in the `scripts` directory:

```shell
python write_jobs.py hyperparams/craftax_memoryless_ppo_no_frame.py
```

This will create a `runs` file with `runs_craftax_memoryless_ppo_no_frame.txt`, where
each line is an experiment run.

To run jobs, you first go to the launching directory and revise single_slurm_job.sh file
line 15 `input_file="../runs/runs_craftax_memoryless_ppo_no_frame.txt"`. Then you can
call

```shell
cd launching
nano single_slurm_job.sh
sbatch single_slurm_job.sh
```

In craftax experiments, you should do the above process 3 more times for
craftax_memoryless_ppo.py, craftax_ppo_LD.py, craftax_ppo.py

After finishing a job, the results should be in the `results`
folder in the root of the project directory that's
named `results/craftax_memoryless_ppo_no_frame`.

You can get the best hyperparameters of a sweep by first
going back into `scripts`, then running

```shell
python best_parse_batch_experiments.py ../results/craftax_memoryless_ppo_no_frame
```

After revising for the correct parameter, you can also use

```shell
sbatch best_parse_batch_experiments_slurm.sh
```

which will create a `pkl` file `best_hyperparam_per_env_res.pkl`.
You can visualize this file by calling

```shell
python plot_best_hyperparams.py
```

Note: I comment out os.environ['MUJOCO_GL'] = 'egl' in ppo_no_jit_env.py to run on my local machine.
If you want to run mujoco on cluster, you should uncomment this line.
