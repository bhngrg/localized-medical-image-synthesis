#!/bin/bash
#SBATCH --job-name=br_lora_lib
#SBATCH --account=fdtbiotech
#SBATCH --partition=a30_normal_q
#SBATCH --gres=gpu:a30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/production/br_lora/library/a30/batch_%j.out
#SBATCH --error=logs/production/br_lora/library/a30/batch_%j.err

set -euo pipefail

if [[ -z "${SLURM_SUBMIT_DIR:-}" ]]; then
    echo "SLURM_SUBMIT_DIR is not available." >&2
    echo "Submit this launcher with sbatch from the repository root." >&2
    exit 1
fi

REPO_ROOT="$(
    cd "${SLURM_SUBMIT_DIR}"
    pwd
)"

if [[ ! -f "${REPO_ROOT}/scripts/run_br_lora_library_batch.py" ]] || \
   [[ ! -f "${REPO_ROOT}/scripts/evaluate_br_lora_external.py" ]]; then
    echo "SLURM_SUBMIT_DIR does not appear to be the repository root:" >&2
    echo "${REPO_ROOT}" >&2
    echo "Submit this launcher with sbatch from the repository root." >&2
    exit 1
fi

cd "${REPO_ROOT}"

if [[ $# -lt 1 ]]; then
    echo "Usage: sbatch scripts/run_br_lora_library_batch_a30.slurm batch_0001 [runner args...]" >&2
    exit 1
fi

BATCH="$1"
shift

if [[ ! "${BATCH}" =~ ^batch_[0-9]{4}$ ]]; then
    echo "Invalid batch identifier: ${BATCH}" >&2
    echo "Expected form: batch_0001" >&2
    exit 1
fi

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
        echo "Activate ${CONDA_ENV} before submission or expose conda on PATH." >&2
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

CHECKPOINT="${CHECKPOINT:-checkpoints/peft/br_lora/a30_full_train/final.pt}"
H5_ROOT="${H5_ROOT:-/scratch/bhanug/archive/BraTS2020_H5_Rebuilt}"
VALIDATION_DATASET="${VALIDATION_DATASET:-/scratch/bhanug/archive/MICCAI_BraTS2020_ValidationData/validation_dataset.yaml}"
STAGING_ROOT="${STAGING_ROOT:-/scratch/bhanug/br_lora_a30_staging}"
FOLDERS_FILE="${FOLDERS_FILE:-data/folders.yaml}"

for required_file in \
    "${CHECKPOINT}" \
    "${VALIDATION_DATASET}" \
    "${FOLDERS_FILE}"
do
    if [[ ! -f "${required_file}" ]]; then
        echo "Required file not found: ${required_file}" >&2
        exit 1
    fi
done

if [[ ! -d "${H5_ROOT}" ]]; then
    echo "H5 root not found: ${H5_ROOT}" >&2
    exit 1
fi

mkdir -p \
    "logs/production/br_lora/library/a30" \
    "${STAGING_ROOT}"

echo "BR-LoRA A30 synthetic-library production"
echo "========================================="
echo "Host                     : $(hostname)"
echo "Date                     : $(date)"
echo "Git commit               : $(git rev-parse HEAD)"
echo "SLURM_JOB_ID             : ${SLURM_JOB_ID:-}"
echo "SLURM_JOB_PARTITION      : ${SLURM_JOB_PARTITION:-}"
echo "Conda environment        : ${CONDA_DEFAULT_ENV:-}"
echo "Batch                    : ${BATCH}"
echo "Checkpoint               : ${CHECKPOINT}"
echo "H5 root                  : ${H5_ROOT}"
echo "Validation dataset       : ${VALIDATION_DATASET}"
echo "Staging root             : ${STAGING_ROOT}"
echo "Posterior samples        : 100"
echo "Evaluation seed          : 42"
echo "Device                   : cuda"
echo

if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi
    echo
fi

python scripts/run_br_lora_library_batch.py \
    --batch "${BATCH}" \
    --h5-root "${H5_ROOT}" \
    --validation-dataset "${VALIDATION_DATASET}" \
    --checkpoint "${CHECKPOINT}" \
    --staging-root "${STAGING_ROOT}" \
    --folders-file "${FOLDERS_FILE}" \
    --posterior-samples 100 \
    --seed 42 \
    --device cuda \
    "$@"
