#!/bin/bash
#SBATCH --partition=gpus         # Partition to run on
#SBATCH --gres=gpu:8             # Request 8 GPU resources for each job
#SBATCH --time=72:00:00          # Request 72 hours of runtime for each job
#SBATCH --mem=32G                # Request 32GB of memory for each job
#SBATCH -J batch_jobs            # Specify a base job name
#SBATCH -o batch_jobs-%j.out     # Specify a base output file
#SBATCH -e batch_jobs-%j.err     # Specify a base error file

# Activate the virtual environment
source ~/pobax/bin/activate

# Change directory to the script's parent directory
cd ..

# Define the hyperparameters directory
HYPERPARAMS_DIR="hyperparams"
ENV_NAME="ant"  # Replace this with the desired environment name

# Initialize an empty list to track job IDs
job_ids=""

# Step 1: Loop over each .py file in the hyperparams directory and submit training jobs
for hyperparams_file in ${HYPERPARAMS_DIR}/*.py; do
    HYPERPARAMS_FILE=$(basename "$hyperparams_file" .py)

    # Submit a job for each hyperparameters file and capture the job ID
    job_id=$(sbatch --parsable <<EOT
#!/bin/bash
#SBATCH --partition=gpus         # Partition to run on
#SBATCH --gres=gpu:8             # Request 4 GPU resources
#SBATCH --time=72:00:00          # Request 72 hours of runtime
#SBATCH --mem=32G                # Request 32GB of memory
#SBATCH -J $HYPERPARAMS_FILE     # Job name based on the hyperparameters file
#SBATCH -o ${HYPERPARAMS_FILE}-%j.out # Output file
#SBATCH -e ${HYPERPARAMS_FILE}-%j.err # Error file

# Activate the virtual environment
source ~/pobax/bin/activate

# Define the work name and paths
HYPERPARAMS_FILE="$HYPERPARAMS_FILE"
HYPERPARAMS_DIR="hyperparams"
RUN_DIR="runs"
RESULTS_DIR="../results"
RUN_FILE="\${RUN_DIR}/runs_\${HYPERPARAMS_FILE}.txt"
COLLECT_PATH="\${RESULTS_DIR}/\${HYPERPARAMS_FILE}/training_results"
DATASET_PATH="\${RESULTS_DIR}/\${HYPERPARAMS_FILE}/dataset"

if [ -d "$COLLECT_PATH" ]; then
    echo "Training results already exist for ${HYPERPARAMS_FILE}, skipping..."
    continue
fi

# Step 1: Run write_job
echo "Running write_job for \${HYPERPARAMS_FILE}"
python write_jobs.py "\${HYPERPARAMS_DIR}/\${HYPERPARAMS_FILE}.py"

# Step 2: Train the model
if [ -f "\$RUN_FILE" ]; then
    echo "Processing file: \$RUN_FILE"
    # Read the single line in the file and execute it
    command=\$(cat "\$RUN_FILE")
    echo "Executing: \$command"
    eval \$command  # Execute the command
    echo "Finished processing \$RUN_FILE"
else
    echo "Error: \$RUN_FILE not found."
    exit 1
fi

echo "Training for \${HYPERPARAMS_FILE} completed."
EOT
    )

    # Add the job ID to the list
    job_ids="${job_ids}:${job_id}"
done

# Remove leading colon from job_ids
job_ids=${job_ids#:}

# Step 2: Submit a new job to collect trajectories once all training jobs are completed
sbatch --dependency=afterok:${job_ids} <<EOT
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

# Define paths for trajectory collection
RESULTS_DIR="../results"
MEMORYLESS_PATH="\${RESULTS_DIR}/\${ENV_NAME}_memoryless_ppo/training_results"
MEMORYLESS_LD_PATH="\${RESULTS_DIR}/\${ENV_NAME}_memoryless_ppo_LD/training_results"
MEMORYLESS_SKIP_PATH="\${RESULTS_DIR}/\${ENV_NAME}_memoryless_no_skip_ppo/training_results"
MEMORYLESS_SKIP_LD_PATH="\${RESULTS_DIR}/\${ENV_NAME}_memoryless_no_skip_ppo_LD/training_results"
RNN_PATH="\${RESULTS_DIR}/\${ENV_NAME}_rnn_ppo/training_results"
RNN_LD_PATH="\${RESULTS_DIR}/\${ENV_NAME}_rnn_ppo_LD/training_results"
RNN_SKIP_PATH="\${RESULTS_DIR}/\${ENV_NAME}_rnn_skip_ppo/training_results"
RNN_SKIP_LD_PATH="\${RESULTS_DIR}/\${ENV_NAME}_rnn_skip_ppo_LD/training_results"

# Loop over all directories in the results directory
for results_dir in \${RESULTS_DIR}/*; do
    BEHAVIOR_PATH="\${results_dir}/training_results"
    DATASET_DIR="\${results_dir}/dataset"
    DIR_NAME=\$(basename "\${results_dir}")

    # Check if the dataset directory already exists
    if [ -d "\$DATASET_DIR" ]; then
        echo "Dataset directory already exists in \${results_dir}, skipping..."
        continue
    fi
    if [ "\$DIR_NAME" == "combined_probe_datasets" ]; then
        echo "Directory is combined_probe_datasets, skipping..."
        continue
    fi

    # Run collect_trajectories
    echo "Running collect_trajectories for \${BEHAVIOR_PATH}"
    python collect_all_trajectories.py \
    --memoryless_path "\$MEMORYLESS_PATH" \
    --memoryless_LD_path "\$MEMORYLESS_LD_PATH" \
    --memoryless_skip_path "\$MEMORYLESS_SKIP_PATH" \
    --memoryless_skip_LD_path "\$MEMORYLESS_SKIP_LD_PATH" \
    --rnn_path "\$RNN_PATH" \
    --rnn_LD_path "\$RNN_LD_PATH" \
    --rnn_skip_path "\$RNN_SKIP_PATH" \
    --rnn_skip_LD_path "\$RNN_SKIP_LD_PATH" \
    --behavior_path "\$BEHAVIOR_PATH"
done

echo "All processes completed successfully."
EOT

