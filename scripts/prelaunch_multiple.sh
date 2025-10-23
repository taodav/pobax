: '
Uses onager to prelaunch multiple experiments.
'
PRELAUNCH_PATHS='navix/ablation/navix_01_ppo_hsize_sweep.py navix/ablation/navix_01_ppo_memoryless_hsize_sweep.py navix/ablation/navix_01_ppo_perfect_memory_hsize_sweep.py'

for item in $PRELAUNCH_PATHS;
do
  python onager_write_jobs.py "hyperparams/$item"
done
