#!/bin/bash
#SBATCH --job-name=brats_nnunet_predict
#SBATCH --account=fdtbiotech
#SBATCH --partition=l40s_normal_q
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/home/bhanug/localized-medical-image-synthesis/screening/brats_nnunet/logs/nnunet_predict_l40s_normal_q_%j.out
#SBATCH --error=/home/bhanug/localized-medical-image-synthesis/screening/brats_nnunet/logs/nnunet_predict_l40s_normal_q_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=bhanug@vt.edu

set -euo pipefail

echo "BraTS 2020 nnU-Net ResEnc L validation prediction"
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "SLURM_JOB_PARTITION=${SLURM_JOB_PARTITION:-unset}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

REPO_ROOT="/home/bhanug/localized-medical-image-synthesis"

NNUNET_ARCHIVE_ROOT="/scratch/bhanug/archive/nnUNet_brats_screening"
NNUNET_RAW="${NNUNET_ARCHIVE_ROOT}/nnUNet_raw"
NNUNET_PREPROCESSED="${NNUNET_ARCHIVE_ROOT}/nnUNet_preprocessed"

NNUNET_RUN_ROOT="/scratch/bhanug/nnUNet_brats_screening"
NNUNET_RESULTS="${NNUNET_RUN_ROOT}/nnUNet_results_l40s_normal_q"

INPUT_DIR="${NNUNET_RAW}/Dataset500_BraTS2020Screening/imagesTs"

PREDICTION_DIR="${NNUNET_RUN_ROOT}/validation_predictions_l40s_normal_q"

LOG_DIR="${REPO_ROOT}/screening/brats_nnunet/logs"

DATASET_ID="500"
DATASET_NAME="Dataset500_BraTS2020Screening"
CONFIGURATION="3d_fullres"
PLANS="nnUNetResEncUNetLPlans"
TRAINER="nnUNetTrainer"
DEVICE="cuda"
CHECKPOINT="checkpoint_final.pth"

cd "${REPO_ROOT}"

# Home-directory logs must exist before runtime output is written.
mkdir -p "${LOG_DIR}"

# Scratch output directories are created immediately before use.
mkdir -p \
  "${NNUNET_RUN_ROOT}" \
  "${NNUNET_RESULTS}" \
  "${PREDICTION_DIR}"

echo
echo "Checking archived nnU-Net inputs..."

if [[ ! -d "${NNUNET_RAW}/${DATASET_NAME}" ]]; then
    echo "Missing nnU-Net raw dataset:"
    echo "${NNUNET_RAW}/${DATASET_NAME}"
    exit 1
fi

if [[ ! -d "${INPUT_DIR}" ]]; then
    echo "Missing nnU-Net validation input directory:"
    echo "${INPUT_DIR}"
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

MODEL_ROOT="${NNUNET_RESULTS}/${DATASET_NAME}/${TRAINER}__${PLANS}__${CONFIGURATION}"

if [[ ! -d "${MODEL_ROOT}" ]]; then
    echo "Missing trained nnU-Net model directory:"
    echo "${MODEL_ROOT}"
    exit 1
fi

for FOLD in 0 1 2 3 4; do
    CHECKPOINT_PATH="${MODEL_ROOT}/fold_${FOLD}/${CHECKPOINT}"

    if [[ ! -f "${CHECKPOINT_PATH}" ]]; then
        echo "Missing fold checkpoint:"
        echo "${CHECKPOINT_PATH}"
        exit 1
    fi
done

EXPECTED_IMAGE_COUNT=500
EXPECTED_SUBJECT_COUNT=125

IMAGE_COUNT=$(
    find "${INPUT_DIR}" \
      -maxdepth 1 \
      -type f \
      -name "BraTS20_Validation_*_000?.nii.gz" \
      | wc -l
)

SUBJECT_COUNT=$(
    find "${INPUT_DIR}" \
      -maxdepth 1 \
      -type f \
      -name "BraTS20_Validation_*_0000.nii.gz" \
      | wc -l
)

if [[ "${IMAGE_COUNT}" -ne "${EXPECTED_IMAGE_COUNT}" ]]; then
    echo "Unexpected validation image count."
    echo "Expected: ${EXPECTED_IMAGE_COUNT}"
    echo "Observed: ${IMAGE_COUNT}"
    exit 1
fi

if [[ "${SUBJECT_COUNT}" -ne "${EXPECTED_SUBJECT_COUNT}" ]]; then
    echo "Unexpected validation subject count."
    echo "Expected: ${EXPECTED_SUBJECT_COUNT}"
    echo "Observed: ${SUBJECT_COUNT}"
    exit 1
fi

source \
  /apps/common/software/Miniforge3/25.11.0-1/etc/profile.d/conda.sh

conda activate brats-nnunet

export nnUNet_raw="${NNUNET_RAW}"
export nnUNet_preprocessed="${NNUNET_PREPROCESSED}"
export nnUNet_results="${NNUNET_RESULTS}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo
echo "Prediction configuration"
echo "=============================================================================="
echo "Dataset                  : ${DATASET_ID}"
echo "Dataset name             : ${DATASET_NAME}"
echo "Configuration            : ${CONFIGURATION}"
echo "Plans                    : ${PLANS}"
echo "Trainer                  : ${TRAINER}"
echo "Folds                    : 0 1 2 3 4"
echo "Checkpoint               : ${CHECKPOINT}"
echo "Device                   : ${DEVICE}"
echo "Partition                : ${SLURM_JOB_PARTITION:-unknown}"
echo "Input directory          : ${INPUT_DIR}"
echo "Prediction directory     : ${PREDICTION_DIR}"
echo "Validation image files   : ${IMAGE_COUNT}"
echo "Validation subjects      : ${SUBJECT_COUNT}"
echo "nnUNet_raw               : ${nnUNet_raw}"
echo "nnUNet_preprocessed      : ${nnUNet_preprocessed}"
echo "nnUNet_results           : ${nnUNet_results}"
echo "Python                   : $(which python)"
echo "Python version           : $(python --version 2>&1)"
echo "nnUNetv2_predict         : $(command -v nnUNetv2_predict)"
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

# Re-create scratch output directories immediately before prediction writes.
mkdir -p \
  "${NNUNET_RUN_ROOT}" \
  "${NNUNET_RESULTS}" \
  "${PREDICTION_DIR}"

echo
echo "Starting nnU-Net validation prediction"
echo "Start time: $(date)"
echo "=============================================================================="

nnUNetv2_predict \
  -i "${INPUT_DIR}" \
  -o "${PREDICTION_DIR}" \
  -d "${DATASET_ID}" \
  -c "${CONFIGURATION}" \
  -tr "${TRAINER}" \
  -p "${PLANS}" \
  -f 0 1 2 3 4 \
  -chk "${CHECKPOINT}" \
  -device "${DEVICE}"

# Re-create scratch output directories immediately after prediction.
mkdir -p \
  "${NNUNET_RUN_ROOT}" \
  "${NNUNET_RESULTS}" \
  "${PREDICTION_DIR}"

echo
echo "Auditing prediction outputs..."

PREDICTION_COUNT=$(
    find "${PREDICTION_DIR}" \
      -maxdepth 1 \
      -type f \
      -name "BraTS20_Validation_*.nii.gz" \
      | wc -l
)

echo "Prediction masks         : ${PREDICTION_COUNT}"

if [[ "${PREDICTION_COUNT}" -ne "${EXPECTED_SUBJECT_COUNT}" ]]; then
    echo "Unexpected prediction count."
    echo "Expected: ${EXPECTED_SUBJECT_COUNT}"
    echo "Observed: ${PREDICTION_COUNT}"
    exit 1
fi

echo
echo "=============================================================================="
echo "nnU-Net validation prediction complete"
echo "Prediction masks         : ${PREDICTION_COUNT}"
echo "Prediction directory     : ${PREDICTION_DIR}"
echo "Host: $(hostname)"
echo "Done: $(date)"
echo "=============================================================================="