RESULTS_DIR='../results/gd_sf_obs_diff/walker_v/walker_v_ppo_gd_sf_random_proj_obs_diff_discrep'

printf "\n"
echo "Parsing $RESULTS_DIR"
python parse_batch_experiments.py --discounted "$RESULTS_DIR"
python best_hyperparams_per_env.py "$RESULTS_DIR/parsed_hparam_scores_discounted.pkl"
python parse_batch_experiments.py "$RESULTS_DIR"
python best_hyperparams_per_env.py "$RESULTS_DIR/parsed_hparam_scores.pkl"
