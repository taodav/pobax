RESULTS_PATHS='../results/navix_01_ppo_memoryless'

for item in $RESULTS_PATHS;
do
  python parse_batch_experiments.py "$item"
  python best_hyperparams_per_env.py "$item/parsed_hparam_scores.pkl"
done
