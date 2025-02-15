RESULTS_PATHS='../results/navix_02_ppo ../results/navix_02_ppo_memoryless ../results/navix_02_ppo_observable'

for item in $RESULTS_PATHS;
do
  python parse_batch_experiments.py --discounted "$item"
  python best_hyperparams_per_env.py "$item/parsed_hparam_scores.pkl"
done
