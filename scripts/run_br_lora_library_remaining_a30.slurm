#!/bin/bash
#SBATCH --job-name=br_lora_rem
#SBATCH --account=fdtbiotech
#SBATCH --partition=a30_normal_q
#SBATCH --gres=gpu:a30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --array=2-40%8
#SBATCH --output=logs/production/br_lora/library/a30/remaining_%A_%a.out
#SBATCH --error=logs/production/br_lora/library/a30/remaining_%A_%a.err

set -euo pipefail

if [[ -z "${SLURM_SUBMIT_DIR:-}" ]]; then
    echo "SLURM_SUBMIT_DIR is not available." >&2
    exit 1
fi

REPO_ROOT="$(
    cd "${SLURM_SUBMIT_DIR}"
    pwd
)"

cd "${REPO_ROOT}"

FOLDERS_FILE="${FOLDERS_FILE:-${REPO_ROOT}/data/folders.yaml}"

read_folder_path() {
    local key="$1"

    python - "${FOLDERS_FILE}" "${key}" <<'PY_CONFIG'
from pathlib import Path
import sys
import yaml

config_path = Path(sys.argv[1])
key = sys.argv[2]

if not config_path.is_file():
    raise SystemExit(
        "Missing machine-specific folders configuration:\n"
        f"{config_path}\n"
        "Copy data/folders.example.yaml to data/folders.yaml "
        "and configure the required paths."
    )

payload = yaml.safe_load(config_path.read_text()) or {}
value = payload.get(key)

if not value:
    raise SystemExit(
        f"Required folders.yaml key is not configured: {key}"
    )

print(Path(value).expanduser().resolve())
PY_CONFIG
}

if [[ ! -f scripts/run_br_lora_library_batch.py ]]; then
    echo "Missing production runner." >&2
    exit 1
fi

if [[ ! "${SLURM_ARRAY_TASK_ID:-}" =~ ^[0-9]+$ ]]; then
    echo "SLURM_ARRAY_TASK_ID is invalid." >&2
    exit 1
fi

BATCH="$(
    printf 'batch_%04d' "${SLURM_ARRAY_TASK_ID}"
)"

CONDA_ENV="${CONDA_ENV:-brats-nnunet}"

if [[ "${CONDA_DEFAULT_ENV:-}" != "${CONDA_ENV}" ]]; then
    if command -v conda >/dev/null 2>&1; then
        eval "$(conda shell.bash hook)"
    elif [[ -n "${CONDA_EXE:-}" && -x "${CONDA_EXE}" ]]; then
        eval "$("${CONDA_EXE}" shell.bash hook)"
    else
        echo "Conda is not available." >&2
        exit 1
    fi

    conda activate "${CONDA_ENV}"
fi

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8

CHECKPOINT="${CHECKPOINT:-checkpoints/peft/br_lora/a30_full_train/final.pt}"

if [[ -z "${H5_ROOT:-}" ]]; then
    H5_ROOT="$(read_folder_path h5_root)"
fi

if [[ -z "${VALIDATION_DATASET:-}" ]]; then
    VALIDATION_DATASET="$(read_folder_path yaml_validation_dataset_path)"
fi

if [[ -z "${STAGING_ROOT:-}" ]]; then
    STAGING_ROOT="$(read_folder_path br_lora_staging_root)"
fi

echo "BR-LoRA A30 remaining-library production"
echo "========================================="
echo "Host                     : $(hostname)"
echo "Date                     : $(date)"
echo "Git commit               : $(git rev-parse HEAD)"
echo "SLURM_ARRAY_JOB_ID       : ${SLURM_ARRAY_JOB_ID:-}"
echo "SLURM_ARRAY_TASK_ID      : ${SLURM_ARRAY_TASK_ID:-}"
echo "Batch                    : ${BATCH}"
echo "Checkpoint               : ${CHECKPOINT}"
echo "Staging root             : ${STAGING_ROOT}"
echo "Posterior samples        : 100"
echo "Evaluation seed          : 42"
echo "Device                   : cuda"
echo

python scripts/run_br_lora_library_batch.py \
    --batch "${BATCH}" \
    --h5-root "${H5_ROOT}" \
    --validation-dataset "${VALIDATION_DATASET}" \
    --checkpoint "${CHECKPOINT}" \
    --staging-root "${STAGING_ROOT}" \
    --folders-file "${FOLDERS_FILE}" \
    --posterior-samples 100 \
    --seed 42 \
    --device cuda
