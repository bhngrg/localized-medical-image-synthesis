#!/bin/bash
#SBATCH --job-name=brats_nnunet_resencL
#SBATCH --account=fdtbiotech
#SBATCH --partition=l40s_normal_q
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4-00:00:00
#SBATCH --array=0-4%5
#SBATCH --output=/home/bhanug/localized-medical-image-synthesis/screening/brats_nnunet/logs/nnunet_resencL_l40s_normal_q_%A_%a.out
#SBATCH --error=/home/bhanug/localized-medical-image-synthesis/screening/brats_nnunet/logs/nnunet_resencL_l40s_normal_q_%A_%a.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=bhanug@vt.edu

set -euo pipefail

echo "BraTS 2020 nnU-Net ResEnc L screening training"
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "SLURM_ARRAY_JOB_ID=${SLURM_ARRAY_JOB_ID}"
echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
echo "SLURM_JOB_PARTITION=${SLURM_JOB_PARTITION:-unset}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

REPO_ROOT="/home/bhanug/localized-medical-image-synthesis"

NNUNET_ARCHIVE_ROOT="/scratch/bhanug/archive/nnUNet_brats_screening"
NNUNET_RAW="${NNUNET_ARCHIVE_ROOT}/nnUNet_raw"
NNUNET_PREPROCESSED="${NNUNET_ARCHIVE_ROOT}/nnUNet_preprocessed"

NNUNET_RUN_ROOT="/scratch/bhanug/nnUNet_brats_screening"
NNUNET_RESULTS="${NNUNET_RUN_ROOT}/nnUNet_results_l40s_normal_q"

LOG_DIR="${REPO_ROOT}/screening/brats_nnunet/logs"

DATASET_ID="500"
DATASET_NAME="Dataset500_BraTS2020Screening"
CONFIGURATION="3d_fullres"
PLANS="nnUNetResEncUNetLPlans"
DEVICE="cuda"
FOLD="${SLURM_ARRAY_TASK_ID}"

cd "${REPO_ROOT}"

# Home-directory logs must exist before Slurm writes runtime output.
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

source \
  /apps/common/software/Miniforge3/25.11.0-1/etc/profile.d/conda.sh

conda activate brats-nnunet

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
