#!/bin/bash
#SBATCH --partition=gpus         # Partition to run on
#SBATCH --gres=gpu:8             # Request 4 GPU resources for collecting trajectories
#SBATCH --time=24:00:00          # Request 24 hours of runtime
#SBATCH --mem=32G                # Request 32GB of memory
#SBATCH -J collect_trajectories  # Job name for collecting trajectories
#SBATCH -o collect_trajectories-%j.out # Output file
#SBATCH -e collect_trajectories-%j.err # Error file

# Activate the virtual environment
source ~/pobax/bin/activate
cd ..
ENV_NAME="reacher"  # Replace this with the desired environment name
# Define paths for trajectory collection
RESULTS_DIR="../results"
MEMORYLESS_PATH="${RESULTS_DIR}/${ENV_NAME}_memoryless_ppo/training_results"
MEMORYLESS_LD_PATH="${RESULTS_DIR}/${ENV_NAME}_memoryless_ppo_LD/training_results"
MEMORYLESS_SKIP_PATH="${RESULTS_DIR}/${ENV_NAME}_memoryless_no_skip_ppo/training_results"
MEMORYLESS_SKIP_LD_PATH="${RESULTS_DIR}/${ENV_NAME}_memoryless_no_skip_ppo_LD/training_results"
RNN_PATH="${RESULTS_DIR}/${ENV_NAME}_rnn_ppo/training_results"
RNN_LD_PATH="${RESULTS_DIR}/${ENV_NAME}_rnn_ppo_LD/training_results"
RNN_SKIP_PATH="${RESULTS_DIR}/${ENV_NAME}_rnn_skip_ppo/training_results"
RNN_SKIP_LD_PATH="${RESULTS_DIR}/${ENV_NAME}_rnn_skip_ppo_LD/training_results"

# Loop over all directories in the results directory
for results_dir in ${RESULTS_DIR}/*; do
    BEHAVIOR_PATH="${results_dir}/training_results"
    DATASET_DIR="${results_dir}/dataset"
    DIR_NAME=$(basename "${results_dir}")

    # Check if the dataset directory already exists
    if [ -d "$DATASET_DIR" ]; then
        echo "Dataset directory already exists in ${results_dir}, skipping..."
        continue
    fi
    if [ "$DIR_NAME" == "combined_probe_datasets" ]; then
        echo "Directory is combined_probe_datasets, skipping..."
        continue
    fi

    # Run collect_trajectories
    echo "Running collect_trajectories for ${BEHAVIOR_PATH}"
    python collect_all_trajectories.py \
    --memoryless_path "$MEMORYLESS_PATH" \
    --memoryless_LD_path "$MEMORYLESS_LD_PATH" \
    --memoryless_skip_path "$MEMORYLESS_SKIP_PATH" \
    --memoryless_skip_LD_path "$MEMORYLESS_SKIP_LD_PATH" \
    --rnn_path "$RNN_PATH" \
    --rnn_LD_path "$RNN_LD_PATH" \
    --rnn_skip_path "$RNN_SKIP_PATH" \
    --rnn_skip_LD_path "$RNN_SKIP_LD_PATH" \
    --behavior_path "$BEHAVIOR_PATH"
done