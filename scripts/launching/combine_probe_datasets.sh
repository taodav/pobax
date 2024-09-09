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

ENV_NAME="ant"
# Run the Python script
# python combine_probe_datasets.py --dataset_0_path ../results/${ENV_NAME}_memoryless_no_skip_ppo/dataset --dataset_1_path ../results/${ENV_NAME}_memoryless_no_skip_ppo_LD/dataset
python combine_probe_datasets.py --dataset_0_path ../results/combined_probe_datasets/combined_probe_datasets.npy --dataset_1_path ../results/${ENV_NAME}_memoryless_ppo/dataset
# python combine_probe_datasets.py --dataset_0_path ../results/combined_probe_datasets/combined_probe_datasets.npy --dataset_1_path ../results/${ENV_NAME}_memoryless_ppo_LD/dataset
# python combine_probe_datasets.py --dataset_0_path ../results/combined_probe_datasets/combined_probe_datasets.npy --dataset_1_path ../results/${ENV_NAME}_rnn_ppo/dataset
# python combine_probe_datasets.py --dataset_0_path ../results/combined_probe_datasets/combined_probe_datasets.npy --dataset_1_path ../results/${ENV_NAME}_rnn_ppo_LD/dataset
# python combine_probe_datasets.py --dataset_0_path ../results/combined_probe_datasets/combined_probe_datasets.npy --dataset_1_path ../results/${ENV_NAME}_rnn_skip_ppo/dataset
# python combine_probe_datasets.py --dataset_0_path ../results/combined_probe_datasets/combined_probe_datasets.npy --dataset_1_path ../results/${ENV_NAME}_rnn_skip_ppo_LD/dataset