#!/bin/bash
#SBATCH --partition=compute
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH -J ${job_name}
#SBATCH -o ${job_name}_%j.out
#SBATCH -e ${job_name}_%j.err
cd ..
# Activate the virtual environment
source ~/pobax/bin/activate

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

srun python -m pobax.algos.ppo_no_jit_env --env ant --action_concat --lr 0.00025 --lambda0 0.95 --lambda1 0.95 --alpha 1 --ld_weight 0 --hidden_size 128 --entropy_coeff 0.01 --steps_log_freq 8 --update_log_freq 10 --total_steps 5000 --seed 2024 --platform cpu --debug --study_name ant_ppo