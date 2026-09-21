#!/bin/bash
#SBATCH --job-name=br_lora_pair_diag
#SBATCH --account=fdtbiotech
#SBATCH --partition=t4_normal_q
#SBATCH --gres=gpu:t4:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/evaluation/downstream_segmentation/br_lora_pair_diagnostics/br_lora_pair_diag_%j.out
#SBATCH --error=logs/evaluation/downstream_segmentation/br_lora_pair_diagnostics/br_lora_pair_diag_%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

mkdir -p \
  logs/evaluation/downstream_segmentation/br_lora_pair_diagnostics

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

MANIFEST="${PAIR_DIAG_MANIFEST:-downstream_evaluation/manifests/br_lora_library_design_10000/br_lora_library_design_10000.csv}"
FOLDERS="${PAIR_DIAG_FOLDERS:-data/folders.yaml}"
OUTPUT_DIR="${PAIR_DIAG_OUTPUT_DIR:-results/downstream_segmentation/synthetic_pair_diagnostics/br_lora}"

echo "BR-LoRA synthetic-pair alignment diagnostic"
echo "=============================================================================="
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "SLURM_JOB_PARTITION=${SLURM_JOB_PARTITION:-}"
echo "Manifest=${MANIFEST}"
echo "Folders=${FOLDERS}"
echo "Output directory=${OUTPUT_DIR}"
echo

rm -rf "${OUTPUT_DIR}"

python -m downstream_evaluation.segmentation.analyze_br_lora_pair_alignment \
  --manifest "${MANIFEST}" \
  --folders "${FOLDERS}" \
  --output-dir "${OUTPUT_DIR}"
