#!/bin/bash
#SBATCH --job-name=baseline_boundary
#SBATCH --account=fdtbiotech
#SBATCH --partition=a30_normal_q
#SBATCH --gres=gpu:a30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/analysis/baseline_mask_sensitivity/boundary_response/boundary_%j.out
#SBATCH --error=logs/analysis/baseline_mask_sensitivity/boundary_response/boundary_%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

mkdir -p \
  logs/analysis/baseline_mask_sensitivity/boundary_response \
  outputs/baseline_mask_sensitivity/boundary_response

CONDA_ENV="${CONDA_ENV:-brats-nnunet}"

activate_environment() {
    if [[ "${CONDA_DEFAULT_ENV:-}" == "${CONDA_ENV}" ]]; then
        return
    fi

    if command -v conda >/dev/null 2>&1; then
        eval "$(conda shell.bash hook)"
    elif [[ -n "${CONDA_EXE:-}" && -x "${CONDA_EXE}" ]]; then
        eval "$("${CONDA_EXE}" shell.bash hook)"
    else
        echo "Conda is not available in the job environment." >&2
        exit 1
    fi

    conda activate "${CONDA_ENV}"
}

activate_environment

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "Baseline boundary-response analysis"
echo "==================================="
echo "Host                : $(hostname)"
echo "Date                : $(date)"
echo "Git commit          : $(git rev-parse HEAD)"
echo "SLURM_JOB_ID        : ${SLURM_JOB_ID:-}"
echo "SLURM_JOB_PARTITION : ${SLURM_JOB_PARTITION:-}"
echo "Conda environment   : ${CONDA_ENV}"
echo
nvidia-smi
echo

python scripts/analyze_baseline_boundary_response.py \
  --device cuda \
  "$@"
