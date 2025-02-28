RESULTS_DIR='../results/pocman'

for item in $RESULTS_DIR/*/;
do
  printf "\n"
  echo "Parsing $item"
  python parse_batch_experiments.py --discounted "$item"
  python best_hyperparams_per_env.py "$item/parsed_hparam_scores_discounted.pkl"
  python parse_batch_experiments.py "$item"
  python best_hyperparams_per_env.py "$item/parsed_hparam_scores.pkl"
done
