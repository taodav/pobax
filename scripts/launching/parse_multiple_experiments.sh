#!/bin/bash
#SBATCH --partition=gpus          # Specify the partition to run on
#SBATCH --gres=gpu:4              # Request 1 GPU resource
#SBATCH --time=12:00:00            # Request 1 hour of runtime
#SBATCH --mem=32G                  # Request 8GB of memory
#SBATCH -J parse_experiments      # Specify a job name
#SBATCH -o parse_experiments-%j.out # Specify an output file for stdout
#SBATCH -e parse_experiments-%j.err # Specify an error file for stderr
cd ..
# Activate the virtual environment
source ~/pobax/bin/activate

# Run the Python script
python parse_multiple_experiments.py