#!/bin/bash
cd ..

# Ensure the script exits if any command fails
set -e

# Define the directories
HYPERPARAMS_DIR="hyperparams"
RUN_DIR="runs"
RESULTS_DIR="../results"

# Loop over all .py files in the hyperparams directory
for hyperparams_file in ${HYPERPARAMS_DIR}/*.py; do
    # Extract the base name (without directory and extension) to use as work name
    HYPERPARAMS_FILE=$(basename "$hyperparams_file" .py)
    RUN_FILE="${RUN_DIR}/runs_${HYPERPARAMS_FILE}.txt"
    COLLECT_PATH="${RESULTS_DIR}/${HYPERPARAMS_FILE}/training_results"
    DATASET_PATH="${RESULTS_DIR}/${HYPERPARAMS_FILE}/dataset"

    if [ -d "$COLLECT_PATH" ]; then
        echo "Training results already exist for ${HYPERPARAMS_FILE}, skipping..."
        continue
    fi

    # Step 1: Run write_job
    echo "Running write_job for ${HYPERPARAMS_FILE}"
    python write_jobs.py "${hyperparams_file}"

    # Step 2: Train the model
    if [ -f "$RUN_FILE" ]; then
        echo "Processing file: $RUN_FILE"
        # Read the single line in the file and execute it
        command=$(cat "$RUN_FILE")
        echo "Executing: $command"
        eval $command  # Execute the command
        echo "Finished processing $RUN_FILE"
    else
        echo "Error: $RUN_FILE not found."
        exit 1
    fi

done

MEMORYLESS_PATH="${RESULTS_DIR}/reacher_memoryless_ppo/training_results"
MEMORYLESS_LD_PATH="${RESULTS_DIR}/reacher_memoryless_ppo_LD/training_results"
MEMORYLESS_SKIP_PATH="${RESULTS_DIR}/reacher_memoryless_no_skip_ppo/training_results"
MEMORYLESS_SKIP_LD_PATH="${RESULTS_DIR}/reacher_memoryless_no_skip_ppo_LD/training_results"
RNN_PATH="${RESULTS_DIR}/reacher_rnn_ppo/training_results"
RNN_LD_PATH="${RESULTS_DIR}/reacher_rnn_ppo_LD/training_results"
RNN_SKIP_PATH="${RESULTS_DIR}/reacher_rnn_skip_ppo/training_results"
RNN_SKIP_LD_PATH="${RESULTS_DIR}/reacher_rnn_skip_ppo_LD/training_results"
# loop over all directories in the results directory
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
    # Step 3: Run collect_trajectories
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

echo "All processes completed successfully."
