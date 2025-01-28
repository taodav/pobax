cd ../../

PRELAUNCH_NAMES='masked_mujoco_ppo masked_mujoco_ppo_memoryless masked_mujoco_ppo_observable'

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
