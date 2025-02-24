#!/bin/bash
#SBATCH --partition=gpus          # Specify the partition to run on
#SBATCH --gres=gpu:1             # Request 1 GPU resource
#SBATCH --nodelist=gpu2002
#SBATCH --time=24:00:00            # Request 1 hour of runtime
#SBATCH --mem=64G                  # Request 8GB of memory
#SBATCH -J best_parse_batch_experiments      # Specify a job name
#SBATCH -o best_parse_batch_experiments-%j.out # Specify an output file for stdout
#SBATCH -e best_parse_batch_experiments-%j.err # Specify an error file for stderr
cd ..
# Activate the virtual environment
source ~/pobax/bin/activate

python parse_single_seed.py ../results/walker2d_memoryless_ppo_no_frame_madrona
