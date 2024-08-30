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

    # Step 3: Run collect_trajectories
    echo "Running collect_trajectories for ${COLLECT_PATH}"
    python collect_trajectories.py --collect_path "$COLLECT_PATH"

    # Step 4: Run train_probe
    echo "Running train_probe for ${DATASET_PATH}"
    python train_probe.py --dataset_path "$DATASET_PATH"

    echo "Process for ${HYPERPARAMS_FILE} completed successfully."
done

echo "All processes completed successfully."
