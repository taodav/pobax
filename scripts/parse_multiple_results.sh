RESULTS_DIR='../results/gd_sf_obs_diff/navix_03'

for item in $RESULTS_DIR/*/;
do
  printf "\n"
  if [ -d "$item" ]; then
    echo "Parsing $item"
    python parse_batch_experiments.py --discounted "$item"
    python best_hyperparams_per_env.py "$item/parsed_hparam_scores_discounted.pkl"
    python parse_batch_experiments.py "$item"
    python best_hyperparams_per_env.py "$item/parsed_hparam_scores.pkl"
  else
    echo "Skipping $item"
  fi
done
