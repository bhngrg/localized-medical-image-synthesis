#!/usr/bin/env bash

set -euo pipefail

DATASET_ID="${NNUNET_DATASET_ID:-500}"
CONFIGURATION="${NNUNET_CONFIGURATION:-3d_fullres}"
PLANS="${NNUNET_PLANS:-nnUNetResEncUNetLPlans}"
DEVICE="${NNUNET_DEVICE:-cuda}"

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 <fold>"
    echo
    echo "fold must be one of: 0 1 2 3 4"
    exit 2
fi

FOLD="$1"

case "${FOLD}" in
    0|1|2|3|4)
        ;;
    *)
        echo "Invalid fold: ${FOLD}"
        echo "Expected one of: 0 1 2 3 4"
        exit 2
        ;;
esac

required_variables=(
    nnUNet_raw
    nnUNet_preprocessed
    nnUNet_results
)

for variable in "${required_variables[@]}"; do
    if [[ -z "${!variable:-}" ]]; then
        echo "Required environment variable is not set: ${variable}"
        exit 2
    fi
done

if ! command -v nnUNetv2_train >/dev/null 2>&1; then
    echo "nnUNetv2_train is not available on PATH."
    exit 2
fi

echo "=============================================================================="
echo "nnU-Net BRATS SCREENING TRAINING"
echo "=============================================================================="
echo "Dataset                  : ${DATASET_ID}"
echo "Configuration            : ${CONFIGURATION}"
echo "Plans                    : ${PLANS}"
echo "Fold                     : ${FOLD}"
echo "Device                   : ${DEVICE}"
echo "nnUNet_raw               : ${nnUNet_raw}"
echo "nnUNet_preprocessed      : ${nnUNet_preprocessed}"
echo "nnUNet_results           : ${nnUNet_results}"
echo "=============================================================================="
echo

nnUNetv2_train \
    "${DATASET_ID}" \
    "${CONFIGURATION}" \
    "${FOLD}" \
    -p "${PLANS}" \
    -device "${DEVICE}" \
    --npz
