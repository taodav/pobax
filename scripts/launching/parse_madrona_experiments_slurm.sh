#!/bin/bash
#SBATCH --partition=gpus          # Specify the partition to run on
#SBATCH --gres=gpu:4              # Request 1 GPU resource
#SBATCH --time=24:00:00            # Request 1 hour of runtime
#SBATCH --mem=32G                  # Request 8GB of memory
#SBATCH -J parse_madrona_experiments      # Specify a job name
#SBATCH -o parse_madrona_experiments-%j.out # Specify an output file for stdout
#SBATCH -e parse_madrona_experiments-%j.err # Specify an error file for stderr
cd ..
# Activate the virtual environment
source ~/pobax/bin/activate

python parse_madrona_experiments.py ../results/hopper_ppo_LD_madrona
