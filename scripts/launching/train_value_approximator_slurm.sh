#!/bin/bash
#SBATCH --partition=gpus          # Specify the partition to run on
#SBATCH --gres=gpu:4              # Request 1 GPU resource
#SBATCH --nodelist=gpu1903
#SBATCH --time=24:00:00            # Request 1 hour of runtime
#SBATCH --mem=32G                  # Request 8GB of memory
#SBATCH -J train_value_approximator      # Specify a job name
#SBATCH -o train_value_approximator-%j.out # Specify an output file for stdout
#SBATCH -e train_value_approximator-%j.err # Specify an error file for stderr
cd ..

source ~/pobax/bin/activate

python train_value_approximator.py --dataset_path ../results/reacher_rnn_lambda0_ppo_best/dataset --target td --approximator rnn_skip
