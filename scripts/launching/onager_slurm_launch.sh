cd ../../

PRELAUNCH_NAMES='rocksample_11_11_perfect_mem_ppo rocksample_15_15_perfect_mem_ppo battleship_10_perfect_mem_ppo'

for item in $PRELAUNCH_NAMES;
do
  onager launch \
      --backend slurm \
      --jobname "$item" \
      --mem 24 \
      --cpus 3 \
      --duration 2-00:00:00 \
      --venv venv \
      --gpus 1 \
      --partition gpus \
      # --tasks-per-node 5
done
