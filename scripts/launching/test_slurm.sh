#!/bin/bash
#SBATCH --partition=gpus          # Specify the partition to run on
#SBATCH --gres=gpu:1              # Request 1 GPU resource
#SBATCH --nodelist=gpu2001
#SBATCH --time=24:00:00            # Request 1 hour of runtime
#SBATCH --mem=32G                  # Request 8GB of memory
#SBATCH -J collect_single_trajectory      # Specify a job name
#SBATCH -o collect_single_trajectory-%j.out # Specify an output file for stdout
#SBATCH -e collect_single_trajectory-%j.err # Specify an error file for stderr
cd ..
# Activate the virtual environment
source ~/pobax/bin/activate

python -m pobax.algos.ppo_no_jit_env --debug --env ant --memoryless