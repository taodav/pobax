RESULTS_DIR='../results/masked_mujoco_best'
#RESULTS_PATHS='navix/navix_01_ppo navix/navix_01_ppo_LD navix_01_ppo_memoryless navix_02_ppo navix_02_ppo_LD navix_03_ppo_memoryless navix_03_ppo_LD navix_03_ppo_memoryless'

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
