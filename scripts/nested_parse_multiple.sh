RESULTS_DIR='../results/cartpole_repeat_sweep'

for nested_dir in $RESULTS_DIR/*/;
do
  for item in $nested_dir/*/;
  do
    printf "\n"
    echo "Parsing $item"
    python parse_batch_experiments.py --discounted "$item"
    python best_hyperparams_per_env.py "$item/parsed_hparam_scores_discounted.pkl"
    python parse_batch_experiments.py "$item"
    python best_hyperparams_per_env.py "$item/parsed_hparam_scores.pkl"
  done
done
