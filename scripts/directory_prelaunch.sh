PRELAUNCH_DIR='hyperparams/masked_mujoco/best/ppo'

for item in $PRELAUNCH_DIR/*;
do
  echo "---------Prelaunching $item---------"
  python onager_write_jobs.py $item
done
