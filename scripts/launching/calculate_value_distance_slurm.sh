#!/bin/bash
#SBATCH --partition=gpus          # Specify the partition to run on
#SBATCH --gres=gpu:4              # Request 1 GPU resource
#SBATCH --nodelist=gpu1903
#SBATCH --time=24:00:00            # Request 1 hour of runtime
#SBATCH --mem=32G                  # Request 8GB of memory
#SBATCH -J calculate_value_distance      # Specify a job name
#SBATCH -o calculate_value_distance-%j.out # Specify an output file for stdout
#SBATCH -e calculate_value_distance-%j.err # Specify an error file for stderr
cd ..

source ~/pobax/bin/activate

python calculate_value_distance.py --dataset_path ../results/dataset --value_network_path ../results/value_distance_mlp_mc