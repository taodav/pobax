cd ../../

PRELAUNCH_NAMES='battleship_10_ppo battleship_10_ppo_memoryless battleship_10_ppo_LD battleship_10_ppo_perfect_memory battleship_10_ppo_perfect_memory_memoryless'

for item in $PRELAUNCH_NAMES;
do
  onager launch \
      --backend slurm \
      --jobname "$item" \
      --mem 32 \
      --cpus 3 \
      --duration 2-00:00:00 \
      --venv venv \
      --gpus 1 \
      --partition 3090-gcondo \
      #-x gpu2106,gpu2257,gpu2505 \
      # --tasks-per-node 5
done
