#!/bin/bash
#SBATCH --job-name=seg_repro_diag
#SBATCH --account=fdtbiotech
#SBATCH --partition=a30_normal_q
#SBATCH --gres=gpu:a30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --mail-user=bhanug@vt.edu
#SBATCH --mail-type=BEGIN,END
#SBATCH --output=downstream_evaluation/logs/seg_repro_diag_a30_normal_q_%j.out
#SBATCH --error=downstream_evaluation/logs/seg_repro_diag_a30_normal_q_%j.err

set -euo pipefail

echo "Downstream segmentation GPU reproducibility diagnostic"
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "SLURM_JOB_PARTITION=${SLURM_JOB_PARTITION:-unset}"

REPO_ROOT="${SLURM_SUBMIT_DIR}"
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

cd "${REPO_ROOT}"

mkdir -p downstream_evaluation/logs

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8

echo
echo "Runtime configuration"
echo "=============================================================================="
echo "Repository root   : ${REPO_ROOT}"
echo "Conda environment : ${CONDA_ENV}"
echo "Partition         : ${SLURM_JOB_PARTITION:-unknown}"
echo "Python            : $(which python)"
echo "Python version    : $(python --version 2>&1)"
echo "=============================================================================="

echo
echo "GPU"
echo "=============================================================================="
nvidia-smi
echo "=============================================================================="

python -m downstream_evaluation.segmentation.diagnose_gpu_reproducibility

echo
echo "=============================================================================="
echo "Reproducibility diagnostic complete"
echo "Host: $(hostname)"
echo "Done: $(date)"
echo "=============================================================================="
