#!/bin/bash
#SBATCH --job-name=downstream_seg
#SBATCH --account=fdtbiotech
#SBATCH --partition=a30_normal_q
#SBATCH --gres=gpu:a30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=downstream_evaluation/logs/training/downstream_seg_%j.out
#SBATCH --error=downstream_evaluation/logs/training/downstream_seg_%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

mkdir -p \
  downstream_evaluation/logs/training \
  outputs/downstream_segmentation/runs

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
export CUBLAS_WORKSPACE_CONFIG=:4096:8

if [ "$#" -lt 1 ]; then
    echo "Usage: sbatch $0 <regime> [additional trainer arguments...]" >&2
    echo "Regimes:" >&2
    echo "  real_only" >&2
    echo "  real_plus_br_lora_mean" >&2
    echo "  real_plus_br_lora_posterior" >&2
    exit 2
fi

REGIME="$1"
shift

case "${REGIME}" in
    real_only|real_plus_br_lora_mean|real_plus_br_lora_posterior)
        ;;
    *)
        echo "Unknown downstream regime: ${REGIME}" >&2
        exit 2
        ;;
esac

echo "Downstream segmentation"
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "SLURM_JOB_PARTITION=${SLURM_JOB_PARTITION:-}"
echo "Regime=${REGIME}"
echo

python scripts/train_downstream_segmentation.py \
  --regime "${REGIME}" \
  --device cuda \
  "$@"
