#!/bin/bash
#SBATCH --partition=gpus          # Specify the partition to run on
#SBATCH --gres=gpu:4              # Request 1 GPU per job
#SBATCH --time=12:00:00           # Request 12 hours of runtime
#SBATCH --mem=32G                 # Request 32GB of memory
#SBATCH -J train_probe            # Specify a job name
#SBATCH -o train_probe-%j.out     # Specify an output file for stdout
#SBATCH -e train_probe-%j.err     # Specify an error file for stderr

RESULTS_DIR="../results"
COMBINED_DATASET_PATH="${RESULTS_DIR}/combined_probe_datasets/combined_probe_datasets.npy"

input_keys=("memoryless_embedding" "memoryless_LD_embedding" "memoryless_skip_embedding" "memoryless_skip_LD_embedding" "rnn_embedding" "rnn_LD_embedding" "rnn_skip_embedding" "rnn_skip_LD_embedding")

for input_key in "${input_keys[@]}"
do
  sbatch <<EOT
#!/bin/bash
#SBATCH --partition=gpus          # Specify the partition to run on
#SBATCH --gres=gpu:4              # Request 1 GPU per job
#SBATCH --nodelist=gpu2001
#SBATCH --time=12:00:00           # Request 12 hours of runtime
#SBATCH --mem=32G                 # Request 32GB of memory
#SBATCH -J probe_${input_key}  # Specify a job name
#SBATCH -o probe_${input_key}-%j.out  # Specify an output file for stdout
#SBATCH -e probe_${input_key}-%j.err  # Specify an error file for stderr

# Activate the virtual environment
cd ..
source ~/pobax/bin/activate

echo "Running train_probe for ${input_key}"
python train_probe.py --dataset_path "$COMBINED_DATASET_PATH" --input_key "$input_key"

EOT
done

echo "All jobs submitted successfully."
