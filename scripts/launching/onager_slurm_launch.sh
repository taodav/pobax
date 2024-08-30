cd ../../

PRELAUNCH_NAMES='atari_memoryless_ppo_LD'

python onager_write_jobs.py hyperparams/$PRELAUNCH_NAMES.py

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