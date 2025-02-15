PRELAUNCH_PATHS='battleship/battleship_10_ppo.py battleship/battleship_10_ppo_LD.py battleship/battleship_10_ppo_memoryless.py battleship/battleship_10_ppo_perfect_memory_memoryless.py battleship/battleship_10_ppo_perfect_memory.py'

for item in $PRELAUNCH_PATHS;
do
  python onager_write_jobs.py "hyperparams/$item"
done
