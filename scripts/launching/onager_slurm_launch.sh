cd ../../

PRELAUNCH_NAMES='navix_01_ppo navix_01_ppo_memoryless navix_01_ppo_LD navix_02_ppo navix_02_ppo_memoryless navix_02_ppo_LD navix_03_ppo navix_03_ppo_memoryless navix_03_ppo_LD'

for item in $PRELAUNCH_NAMES;
do
  onager launch \
      --backend slurm \
      --jobname "$item" \
      --mem 24 \
      --cpus 3 \
      --duration 0-12:00:00 \
      --venv venv \
      --gpus 1 \
      --partition 3090-gcondo \
      -x gpu2106 \
      # --tasks-per-node 5
done
