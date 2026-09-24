#!/bin/bash
#SBATCH --job-name=peft_lib
#SBATCH --account=fdtbiotech
#SBATCH --partition=a30_normal_q
#SBATCH --gres=gpu:a30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/production/peft/library/a30/batch_%j.out
#SBATCH --error=logs/production/peft/library/a30/batch_%j.err

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

if [[ ! -f "${REPO_ROOT}/scripts/run_adaptation_library_batch.py" ]] || \
   [[ ! -f "${REPO_ROOT}/scripts/evaluate_adaptation_external.py" ]]; then
    echo "SLURM_SUBMIT_DIR does not appear to be the repository root:" >&2
    echo "${REPO_ROOT}" >&2
    echo "Submit this launcher with sbatch from the repository root." >&2
    exit 1
fi

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

if [[ $# -lt 2 ]]; then
    echo "Usage: sbatch scripts/run_adaptation_library_batch_a30.slurm <method> batch_0001 [runner args...]" >&2
    echo "Methods:" >&2
    echo "  regional_lora" >&2
    echo "  dora" >&2
    echo "  lokr" >&2
    echo "  bitfit" >&2
    exit 2
fi

METHOD="$1"
BATCH="$2"
shift 2

case "${METHOD}" in
    regional_lora|dora|lokr|bitfit)
        ;;
    *)
        echo "Unknown deterministic adaptation method: ${METHOD}" >&2
        exit 2
        ;;
esac

if [[ ! "${BATCH}" =~ ^batch_[0-9]{4}$ ]]; then
    echo "Invalid batch identifier: ${BATCH}" >&2
    echo "Expected form: batch_0001" >&2
    exit 2
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

CHECKPOINT="${CHECKPOINT:-checkpoints/peft/${METHOD}/full_train/latest.pt}"

if [[ -z "${H5_ROOT:-}" ]]; then
    H5_ROOT="$(read_folder_path h5_root)"
fi

if [[ -z "${VALIDATION_DATASET:-}" ]]; then
    VALIDATION_DATASET="$(read_folder_path yaml_validation_dataset_path)"
fi

if [[ -z "${REFERENCE_BR_LORA_LIBRARY_ROOT:-}" ]]; then
    REFERENCE_BR_LORA_LIBRARY_ROOT="$(read_folder_path br_lora_library_root)"
fi

if [[ -z "${STAGING_ROOT:-}" ]]; then
    STAGING_ROOT="$(read_folder_path "${METHOD}_staging_root")"
fi

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

for required_directory in \
    "${H5_ROOT}" \
    "${REFERENCE_BR_LORA_LIBRARY_ROOT}"
do
    if [[ ! -d "${required_directory}" ]]; then
        echo "Required directory not found: ${required_directory}" >&2
        exit 1
    fi
done

mkdir -p \
    "logs/production/peft/library/a30" \
    "${STAGING_ROOT}"

echo "Deterministic PEFT A30 synthetic-library production"
echo "==================================================="
echo "Host                     : $(hostname)"
echo "Date                     : $(date)"
echo "Git commit               : $(git rev-parse HEAD)"
echo "SLURM_JOB_ID             : ${SLURM_JOB_ID:-}"
echo "SLURM_JOB_PARTITION      : ${SLURM_JOB_PARTITION:-}"
echo "Method                   : ${METHOD}"
echo "Batch                    : ${BATCH}"
echo "Conda environment        : ${CONDA_DEFAULT_ENV:-}"
echo "Checkpoint               : ${CHECKPOINT}"
echo "H5 root                  : ${H5_ROOT}"
echo "Validation dataset       : ${VALIDATION_DATASET}"
echo "Reference BR-LoRA library: ${REFERENCE_BR_LORA_LIBRARY_ROOT}"
echo "Staging root             : ${STAGING_ROOT}"
echo "Evaluation seed          : 42"
echo "Device                   : cuda"
echo

if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi
    echo
fi

python scripts/run_adaptation_library_batch.py \
    --batch "${BATCH}" \
    --method "${METHOD}" \
    --checkpoint "${CHECKPOINT}" \
    --h5-root "${H5_ROOT}" \
    --validation-dataset "${VALIDATION_DATASET}" \
    --reference-br-lora-library-root "${REFERENCE_BR_LORA_LIBRARY_ROOT}" \
    --staging-root "${STAGING_ROOT}" \
    --folders-file "${FOLDERS_FILE}" \
    --seed 42 \
    --device cuda \
    "$@"
