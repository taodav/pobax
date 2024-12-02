#!/bin/bash
#SBATCH --partition=gpus          # Specify the partition to run on
#SBATCH --gres=gpu:4              # Request 1 GPU resource
#SBATCH --nodelist=gpu2001
#SBATCH --time=24:00:00            # Request 1 hour of runtime
#SBATCH --mem=32G                  # Request 8GB of memory
#SBATCH -J best_hyperparams_per_env      # Specify a job name
#SBATCH -o best_hyperparams_per_env-%j.out # Specify an output file for stdout
#SBATCH -e best_hyperparams_per_env-%j.err # Specify an error file for stderr
cd ..
# Activate the virtual environment
source ~/pobax/bin/activate

python best_hyperparams_per_env.py ../results/ant_memoryless_ppo_no_frame/parsed_hparam_scores.pkl