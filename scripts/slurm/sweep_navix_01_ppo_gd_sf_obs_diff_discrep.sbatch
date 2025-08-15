#!/usr/bin/env bash
#SBATCH --job-name=navix_gdppo
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=48:00:00
# Set the correct line count after step 1 (N lines in runs file). Limit concurrency with % if desired.
#SBATCH --array=1-3%3
#SBATCH --output=slurm_logs/navix_gdppo_F_%A_%a.out
#SBATCH --error=slurm_logs/navix_gdppo_F_%A_%a.err

set -euo pipefail

cd /nas/ucb/peterkoepernik/pobax
source .venv/bin/activate

mkdir -p slurm_logs

RUNS_FILE="scripts/runs/runs_navix_01_ppo_gd_sf_obs_diff_discrep_F.txt"
CMD="$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$RUNS_FILE")"

# Helpful JAX/XLA settings on shared GPUs
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90

echo "[$SLURM_ARRAY_TASK_ID] Running: $CMD"
srun uv run $CMD