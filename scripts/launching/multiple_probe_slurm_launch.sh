#!/bin/bash
#SBATCH --partition=gpus         # Partition to run on
#SBATCH --gres=gpu:8             # Request 8 GPU resources for each job
#SBATCH --time=12:00:00          # Request 12 hours of runtime for each job
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

# Loop over each .py file in the hyperparams directory and submit a job
for hyperparams_file in ${HYPERPARAMS_DIR}/*.py; do
    HYPERPARAMS_FILE=$(basename "$hyperparams_file" .py)
    
    # Submit a job for each hyperparameters file
    sbatch <<EOT
#!/bin/bash
#SBATCH --partition=gpus         # Partition to run on
#SBATCH --gres=gpu:4             # Request 8 GPU resources
#SBATCH --time=12:00:00          # Request 12 hours of runtime
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
RUN_FILE="\${RUN_DIR}/runs_\${HYPERPARAMS_FILE}.txt"
RESULTS_DIR="../results"
COLLECT_PATH="\${RESULTS_DIR}/\${HYPERPARAMS_FILE}/training_results"
DATASET_PATH="\${RESULTS_DIR}/\${HYPERPARAMS_FILE}/dataset"

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

# Step 3: Run collect_trajectories
echo "Running collect_trajectories for \${COLLECT_PATH}"
python collect_trajectories.py --collect_path \$COLLECT_PATH

# Step 4: Run train_probe
echo "Running train_probe for \${DATASET_PATH}"
python train_probe.py --dataset_path \$DATASET_PATH

echo "Process for \${HYPERPARAMS_FILE} completed successfully."
EOT

done
