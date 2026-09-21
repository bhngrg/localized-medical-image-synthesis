#!/bin/bash
#SBATCH --job-name=posterior_cache
#SBATCH --account=fdtbiotech
#SBATCH --partition=t4_normal_q
#SBATCH --gres=gpu:t4:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/training/downstream_segmentation/posterior_cache/posterior_cache_%j.out
#SBATCH --error=logs/training/downstream_segmentation/posterior_cache/posterior_cache_%j.err

set -euo pipefail

SCRIPT_DIR="$(
    cd "$(dirname "${BASH_SOURCE[0]}")"
    pwd
)"

REPO_ROOT="$(
    cd "${SCRIPT_DIR}/../.."
    pwd
)"

FOLDERS_FILE="${REPO_ROOT}/data/folders.yaml"

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

if [[ -z "${POSTERIOR_CACHE_LIBRARY_ROOT:-}" ]]; then
    POSTERIOR_CACHE_LIBRARY_ROOT="$(read_folder_path br_lora_library_root)"
fi

if [[ -z "${POSTERIOR_CACHE_OUTPUT_ROOT:-}" ]]; then
    POSTERIOR_CACHE_OUTPUT_ROOT="$(
        read_folder_path downstream_posterior_shard_cache_root
    )"
fi

cd "${REPO_ROOT}"

mkdir -p \
  logs/training/downstream_segmentation/posterior_cache

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

MANIFEST="${POSTERIOR_CACHE_MANIFEST:-downstream_evaluation/manifests/br_lora_library_design_10000/br_lora_library_design_10000.csv}"
LIBRARY_ROOT="${POSTERIOR_CACHE_LIBRARY_ROOT}"
OUTPUT_ROOT="${POSTERIOR_CACHE_OUTPUT_ROOT}"

SEED="${POSTERIOR_CACHE_SEED:-42}"
EPOCHS="${POSTERIOR_CACHE_EPOCHS:-20}"
SHARD_SIZE="${POSTERIOR_CACHE_SHARD_SIZE:-500}"

echo "Downstream BR-LoRA posterior shard-cache build"
echo "=============================================================================="
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "SLURM_JOB_PARTITION=${SLURM_JOB_PARTITION:-}"
echo "Manifest=${MANIFEST}"
echo "Library root=${LIBRARY_ROOT}"
echo "Output root=${OUTPUT_ROOT}"
echo "Seed=${SEED}"
echo "Epochs=${EPOCHS}"
echo "Shard size=${SHARD_SIZE}"
echo

python -m downstream_evaluation.segmentation.build_posterior_shard_cache \
  --manifest "${MANIFEST}" \
  --library-root "${LIBRARY_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --seed "${SEED}" \
  --epochs "${EPOCHS}" \
  --shard-size "${SHARD_SIZE}"
