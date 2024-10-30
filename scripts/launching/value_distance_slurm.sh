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

# Paths to your dataset and model checkpoints
DATASET_PATH="../results/reacher_rnn_lambda0_ppo_best/dataset"
MEMORYLESS_LAMBDA0_PATH="../results/memoryless_lambda0/training_results"
MEMORYLESS_LAMBDA1_PATH="../results/memoryless_lambda1/training_results"
MEMORYLESS_SKIP_LAMBDA0_PATH="../results/memoryless_skip_connection_lambda0/training_results"
MEMORYLESS_SKIP_LAMBDA1_PATH="../results/memoryless_skip_connection_lambda1/training_results"
RNN_SKIP_LAMBDA0_PATH="../results/rnn_lambda0/training_results"
RNN_SKIP_LAMBDA1_PATH="../results/rnn_lambda1/training_results"
RNN_LAMBDA0_PATH="../results/rnn_skip_lambda0/training_results"
RNN_LAMBDA1_PATH="../results/rnn_skip_lambda1/training_results"

# Name of your Python script
PYTHON_SCRIPT="../value_distance.py"

# Run the Python script
python $PYTHON_SCRIPT \
    --dataset_path $DATASET_PATH \
    --memoryless_lambda0_path $MEMORYLESS_LAMBDA0_PATH \
    --memoryless_lambda1_path $MEMORYLESS_LAMBDA1_PATH \
    --memoryless_skip_lambda0_path $MEMORYLESS_SKIP_LAMBDA0_PATH \
    --memoryless_skip_lambda1_path $MEMORYLESS_SKIP_LAMBDA1_PATH \
    --rnn_skip_lambda0_path $RNN_SKIP_LAMBDA0_PATH \
    --rnn_skip_lambda1_path $RNN_SKIP_LAMBDA1_PATH \
    --rnn_lambda0_path $RNN_LAMBDA0_PATH \
    --rnn_lambda1_path $RNN_LAMBDA1_PATH \
