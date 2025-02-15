RESULTS_PATHS='navix_01_ppo navix_01_ppo_LD navix_01_ppo_memoryless navix_02_ppo navix_02_ppo_LD navix_03_ppo_memoryless navix_03_ppo_LD navix_03_ppo_memoryless'

for item in $RESULTS_PATHS;
do
  python parse_batch_experiments.py --discounted "../results/$item"
  python best_hyperparams_per_env.py "../results/$item/parsed_hparam_scores.pkl"
done
