#!/bin/bash
#SBATCH --job-name=brats_nnunet_resencL
#SBATCH --account=fdtbiotech
#SBATCH --partition=l40s_normal_q
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4-00:00:00
#SBATCH --array=0-4%5
#SBATCH --output=logs/screening/brats_nnunet/training/nnunet_resencL_l40s_normal_q_%A_%a.out
#SBATCH --error=logs/screening/brats_nnunet/training/nnunet_resencL_l40s_normal_q_%A_%a.err

set -euo pipefail

echo "BraTS 2020 nnU-Net ResEnc L screening training"
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "SLURM_ARRAY_JOB_ID=${SLURM_ARRAY_JOB_ID}"
echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
echo "SLURM_JOB_PARTITION=${SLURM_JOB_PARTITION:-unset}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

SCRIPT_DIR="$(
    cd "$(dirname "${BASH_SOURCE[0]}")"
    pwd
)"

REPO_ROOT="$(
    cd "${SCRIPT_DIR}/../../.."
    pwd
)"

FOLDERS_FILE="${REPO_ROOT}/data/folders.yaml"

NNUNET_CONDA_ENV="${NNUNET_CONDA_ENV:-brats-nnunet}"

activate_nnunet_environment() {
    if [[ "${CONDA_DEFAULT_ENV:-}" == "${NNUNET_CONDA_ENV}" ]]; then
        return
    fi

    if command -v conda >/dev/null 2>&1; then
        eval "$(conda shell.bash hook)"
    elif [[ -n "${CONDA_EXE:-}" && -x "${CONDA_EXE}" ]]; then
        eval "$("${CONDA_EXE}" shell.bash hook)"
    else
        echo "Conda is not available in the job environment." >&2
        echo "Activate ${NNUNET_CONDA_ENV} before submission or expose conda on PATH." >&2
        exit 1
    fi

    conda activate "${NNUNET_CONDA_ENV}"
}

activate_nnunet_environment

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

if [[ -z "${NNUNET_ARCHIVE_ROOT:-}" ]]; then
    NNUNET_ARCHIVE_ROOT="$(read_folder_path nnunet_archive_root)"
fi

if [[ -z "${NNUNET_RUN_ROOT:-}" ]]; then
    NNUNET_RUN_ROOT="$(read_folder_path nnunet_run_root)"
fi

NNUNET_RAW="${NNUNET_ARCHIVE_ROOT}/nnUNet_raw"
NNUNET_PREPROCESSED="${NNUNET_ARCHIVE_ROOT}/nnUNet_preprocessed"

NNUNET_RESULTS="${NNUNET_RUN_ROOT}/nnUNet_results_l40s_normal_q"

LOG_DIR="${REPO_ROOT}/logs/screening/brats_nnunet/training"

DATASET_ID="500"
DATASET_NAME="Dataset500_BraTS2020Screening"
CONFIGURATION="3d_fullres"
PLANS="nnUNetResEncUNetLPlans"
DEVICE="cuda"
FOLD="${SLURM_ARRAY_TASK_ID}"

cd "${REPO_ROOT}"

# Repository log directory is retained for runtime provenance.
mkdir -p "${LOG_DIR}"

# Scratch output directories are created immediately before use.
mkdir -p \
  "${NNUNET_RUN_ROOT}" \
  "${NNUNET_RESULTS}"

echo
echo "Checking archived nnU-Net inputs..."

if [[ ! -d "${NNUNET_RAW}/${DATASET_NAME}" ]]; then
    echo "Missing nnU-Net raw dataset:"
    echo "${NNUNET_RAW}/${DATASET_NAME}"
    exit 1
fi

if [[ ! -d "${NNUNET_PREPROCESSED}/${DATASET_NAME}" ]]; then
    echo "Missing nnU-Net preprocessed dataset:"
    echo "${NNUNET_PREPROCESSED}/${DATASET_NAME}"
    exit 1
fi

if [[ ! -f "${NNUNET_PREPROCESSED}/${DATASET_NAME}/${PLANS}.json" ]]; then
    echo "Missing nnU-Net plans file:"
    echo "${NNUNET_PREPROCESSED}/${DATASET_NAME}/${PLANS}.json"
    exit 1
fi

export nnUNet_raw="${NNUNET_RAW}"
export nnUNet_preprocessed="${NNUNET_PREPROCESSED}"
export nnUNet_results="${NNUNET_RESULTS}"

export NNUNET_DATASET_ID="${DATASET_ID}"
export NNUNET_CONFIGURATION="${CONFIGURATION}"
export NNUNET_PLANS="${PLANS}"
export NNUNET_DEVICE="${DEVICE}"

# Avoid unnecessary nested CPU threading.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo
echo "Training configuration"
echo "=============================================================================="
echo "Dataset                  : ${DATASET_ID}"
echo "Dataset name             : ${DATASET_NAME}"
echo "Configuration            : ${CONFIGURATION}"
echo "Plans                    : ${PLANS}"
echo "Fold                     : ${FOLD}"
echo "Device                   : ${DEVICE}"
echo "Partition                : ${SLURM_JOB_PARTITION:-unknown}"
echo "nnUNet_raw               : ${nnUNet_raw}"
echo "nnUNet_preprocessed      : ${nnUNet_preprocessed}"
echo "nnUNet_results           : ${nnUNet_results}"
echo "Python                   : $(which python)"
echo "Python version           : $(python --version 2>&1)"
echo "nnUNetv2_train           : $(command -v nnUNetv2_train)"
echo "=============================================================================="

echo
echo "GPU"
echo "=============================================================================="

nvidia-smi

python - <<'PY'
import torch
from importlib.metadata import version

print("Torch version      :", torch.__version__)
print("Torch CUDA build   :", torch.version.cuda)
print("CUDA available     :", torch.cuda.is_available())
print("nnU-Net version    :", version("nnunetv2"))

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA is not available inside the allocated L40S job."
    )

print("CUDA device count  :", torch.cuda.device_count())
print("CUDA device        :", torch.cuda.get_device_name(0))

properties = torch.cuda.get_device_properties(0)

print(
    "GPU memory (GiB)   :",
    round(
        properties.total_memory
        / (1024 ** 3),
        2,
    ),
)
PY

echo "=============================================================================="

# Re-create scratch output directories immediately before training writes.
mkdir -p \
  "${NNUNET_RUN_ROOT}" \
  "${NNUNET_RESULTS}"

echo
echo "Starting nnU-Net fold ${FOLD}"
echo "Start time: $(date)"
echo "=============================================================================="

"${REPO_ROOT}/screening/brats_nnunet/training/train_fold.sh" \
  "${FOLD}"

# Re-create scratch output directories immediately after training.
mkdir -p \
  "${NNUNET_RUN_ROOT}" \
  "${NNUNET_RESULTS}"

echo
echo "=============================================================================="
echo "Fold ${FOLD} complete"
echo "Host: $(hostname)"
echo "Done: $(date)"
echo "=============================================================================="
