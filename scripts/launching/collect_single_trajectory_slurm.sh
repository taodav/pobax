#!/bin/bash
#SBATCH --partition=gpus          # Specify the partition to run on
#SBATCH --gres=gpu:4              # Request 1 GPU resource
#SBATCH --nodelist=gpu2001
#SBATCH --time=12:00:00            # Request 1 hour of runtime
#SBATCH --mem=32G                  # Request 8GB of memory
#SBATCH -J combine_probe_datasets      # Specify a job name
#SBATCH -o combine_probe_datasets-%j.out # Specify an output file for stdout
#SBATCH -e combine_probe_datasets-%j.err # Specify an error file for stderr
cd ..
# Activate the virtual environment
source ~/pobax/bin/activate

python collect_single_trajectory.py --collect_path ../results/reacher_rnn_lambda0_ppo_best/training_results