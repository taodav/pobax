PRELAUNCH_PATHS='hyperparams/navix/navix_01_ppo.py hyperparams/navix/navix_01_ppo_memoryless.py hyperparams/navix/navix_01_ppo_observable.py hyperparams/navix/navix_02_ppo.py hyperparams/navix/navix_02_ppo_memoryless.py hyperparams/navix/navix_02_ppo_observable.py hyperparams/navix/navix_03_ppo.py hyperparams/navix/navix_03_ppo_memoryless.py hyperparams/navix/navix_03_ppo_observable.py'

for item in $PRELAUNCH_PATHS;
do
  python onager_write_jobs.py "$item"
done
