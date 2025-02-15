PRELAUNCH_PATHS='rocksample/rocksample_11_11_ppo.py rocksample/rocksample_11_11_ppo_LD.py rocksample/rocksample_11_11_ppo_memoryless.py rocksample/rocksample_11_11_ppo_perfect_memory_memoryless.py rocksample/rocksample_11_11_ppo_perfect_memory.py'

for item in $PRELAUNCH_PATHS;
do
  python onager_write_jobs.py "hyperparams/$item"
done
