#!/bin/bash
#SBATCH --partition=gpus          # Specify the partition to run on
#SBATCH --gres=gpu:1              # Request 1 GPU resource
#SBATCH --nodelist=gpu2001
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH -J $test_slurm
#SBATCH -o $test_%j.out
#SBATCH -e $test_%j.err
cd ..
# Activate the virtual environment
source ~/pobax/bin/activate

# Use this when you want to rerun the scripts a lot of times
python visualize_gymnax_env.py
