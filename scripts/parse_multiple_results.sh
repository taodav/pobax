RESULTS_PATHS='../results/rocksample_15_15_transformer'

for item in $RESULTS_PATHS;
do
  python parse_batch_experiments.py --discounted "$item"
  python best_hyperparams_per_env.py "$item/parsed_hparam_scores.pkl"
done
