#!/bin/bash
#SBATCH --partition=gpus         # Partition to run on
#SBATCH --gres=gpu:8             # Request 8 GPU resources
#SBATCH --time=12:00:00          # Request 12 hours of runtime
#SBATCH --mem=32G                # Request 32GB of memory
#SBATCH -J ${HYPERPARAMS_FILE}   # Specify a job name based on the hyperparameters file
#SBATCH -o ${HYPERPARAMS_FILE}-%j.out # Specify an output file
#SBATCH -e ${HYPERPARAMS_FILE}-%j.err # Specify an error file
cd ..
# Ensure the script exits if any command fails
set -e

# Define the work name and paths
HYPERPARAMS_FILE="half_cheetah_memoryless_no_skip_ppo"  # Replace with your actual work name
HYPERPARAMS_DIR="hyperparams"
RUN_DIR="runs"
RUN_FILE="${RUN_DIR}/runs_${HYPERPARAMS_FILE}.txt"
RESULTS_DIR="../results"
COLLECT_PATH="${RESULTS_DIR}/${HYPERPARAMS_FILE}/training_results"
DATASET_PATH="${RESULTS_DIR}/${HYPERPARAMS_FILE}/dataset"

# Activate the virtual environment
source ~/pobax/bin/activate

# Step 1: Run write_job
echo "Running write_job for ${HYPERPARAMS_FILE}"
python write_jobs.py "${HYPERPARAMS_DIR}/${HYPERPARAMS_FILE}.py"

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
python collect_trajectories.py --collect_path $COLLECT_PATH

# Step 4: Run train_probe
echo "Running train_probe for ${DATASET_PATH}"
python train_probe.py --dataset_path $DATASET_PATH

echo "Process completed successfully."
