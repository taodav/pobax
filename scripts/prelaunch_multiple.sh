#PRELAUNCH_PATHS='rocksample/rocksample_15_15_ppo.py rocksample/rocksample_15_15_ppo_LD.py rocksample/rocksample_15_15_ppo_memoryless.py rocksample/rocksample_15_15_ppo_perfect_memory_memoryless.py rocksample/rocksample_15_15_ppo_perfect_memory.py'
PRELAUNCH_PATHS='navix/ablation/navix_01_ppo_hsize_sweep.py navix/ablation/navix_01_ppo_memoryless_hsize_sweep.py navix/ablation/navix_01_ppo_perfect_memory_hsize_sweep.py'

for item in $PRELAUNCH_PATHS;
do
  python onager_write_jobs.py "hyperparams/$item"
done
