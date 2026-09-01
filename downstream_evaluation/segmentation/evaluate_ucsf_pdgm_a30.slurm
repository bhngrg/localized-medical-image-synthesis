#!/bin/bash
#SBATCH --job-name=ucsf_pdgm_eval
#SBATCH --account=fdtbiotech
#SBATCH --partition=a30_normal_q
#SBATCH --gres=gpu:a30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/evaluation/downstream_segmentation/ucsf_pdgm/ucsf_pdgm_eval_%j.out
#SBATCH --error=logs/evaluation/downstream_segmentation/ucsf_pdgm/ucsf_pdgm_eval_%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

mkdir -p \
  logs/evaluation/downstream_segmentation/ucsf_pdgm \
  outputs/downstream_segmentation/evaluations

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

echo "UCSF-PDGM external segmentation evaluation"
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "SLURM_JOB_PARTITION=${SLURM_JOB_PARTITION:-}"
echo

python -m downstream_evaluation.segmentation.evaluate_ucsf_pdgm \
  --device cuda \
  "$@"
