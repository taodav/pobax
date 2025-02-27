#!/bin/bash
#SBATCH --partition=gpu          # Specify the partition to run on
#SBATCH --gres=gpu:1             # Request 1 GPU resource
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00            # Request 1 hour of runtime
#SBATCH --mem=64G                  # Request 8GB of memory
#SBATCH -J best_parse_batch_experiments      # Specify a job name
#SBATCH -o best_parse_batch_experiments-%j.out # Specify an output file for stdout
#SBATCH -e best_parse_batch_experiments-%j.err # Specify an error file for stderr
cd ..
# Activate the virtual environment
source ~/pobax_baseline/bin/activate

python parse_single_seed.py ../results/halfcheetah_p_transformer --discounted
