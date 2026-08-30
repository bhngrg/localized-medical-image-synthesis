#!/bin/bash
#SBATCH --job-name=ucsf_pdgm_eval
#SBATCH --account=fdtbiotech
#SBATCH --partition=a30_normal_q
#SBATCH --gres=gpu:a30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --mail-user=bhanug@vt.edu
#SBATCH --mail-type=BEGIN,END
#SBATCH --output=downstream_evaluation/logs/ucsf_pdgm_eval_a30_normal_q_%j.out
#SBATCH --error=downstream_evaluation/logs/ucsf_pdgm_eval_a30_normal_q_%j.err

set -euo pipefail

echo "UCSF-PDGM external segmentation evaluation"
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "SLURM_JOB_PARTITION=${SLURM_JOB_PARTITION:-unset}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

REPO_ROOT="${SLURM_SUBMIT_DIR}"
CONDA_ENV="${CONDA_ENV:-brats-nnunet}"
OUTPUT_DIR="outputs/downstream_segmentation/ucsf_pdgm_external_202"

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

if [[ -e "${OUTPUT_DIR}" ]]; then
    echo "Refusing to overwrite existing evaluation output:" >&2
    echo "  ${OUTPUT_DIR}" >&2
    exit 1
fi

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo
echo "Runtime configuration"
echo "=============================================================================="
echo "Repository root   : ${REPO_ROOT}"
echo "Conda environment : ${CONDA_ENV}"
echo "Partition         : ${SLURM_JOB_PARTITION:-unknown}"
echo "Python            : $(which python)"
echo "Python version    : $(python --version 2>&1)"
echo "Output directory  : ${OUTPUT_DIR}"
echo "=============================================================================="

echo
echo "GPU"
echo "=============================================================================="
nvidia-smi

python - <<'PY'
import torch

print("Torch version    :", torch.__version__)
print("Torch CUDA build :", torch.version.cuda)
print("CUDA available   :", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA is not available inside the allocated A30 job."
    )

print("CUDA device      :", torch.cuda.get_device_name(0))
PY

echo "=============================================================================="

echo
echo "Starting full UCSF-PDGM evaluation"
echo "Start time: $(date)"
echo "=============================================================================="

python -m downstream_evaluation.segmentation.evaluate_ucsf_pdgm

echo
echo "=============================================================================="
echo "UCSF-PDGM evaluation complete"
echo "Host: $(hostname)"
echo "Done: $(date)"
echo "=============================================================================="
