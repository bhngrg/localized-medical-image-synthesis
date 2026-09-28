#!/bin/bash
#SBATCH --job-name=geom_response_assoc
#SBATCH --account=fdtbiotech
#SBATCH --partition=a30_normal_q
#SBATCH --gres=gpu:a30:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/analysis/baseline_mask_sensitivity/geometry_response_associations/associations_%j.out
#SBATCH --error=logs/analysis/baseline_mask_sensitivity/geometry_response_associations/associations_%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

CONDA_ENV="${CONDA_ENV:-brats-nnunet}"

mkdir -p \
    logs/analysis/baseline_mask_sensitivity/geometry_response_associations \
    results/baseline_mask_sensitivity/geometry_response_associations

if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
else
    echo "conda is not available on PATH." >&2
    exit 1
fi

conda activate "${CONDA_ENV}"

echo "================================================================"
echo "STAGE 5: GEOMETRY / BOUNDARY RESPONSE ASSOCIATIONS"
echo "================================================================"
echo "Job ID:       ${SLURM_JOB_ID}"
echo "Node:         $(hostname)"
echo "Start time:   $(date)"
echo "CPUs:         ${SLURM_CPUS_PER_TASK}"
echo "Conda env:    ${CONDA_ENV}"
echo

python scripts/analyze_baseline_geometry_response_associations.py \
    --bootstrap-replicates 10000 \
    --bootstrap-seed 2026

echo
echo "================================================================"
echo "STAGE 5 COMPLETE"
echo "================================================================"
echo "End time: $(date)"
