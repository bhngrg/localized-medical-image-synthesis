#!/bin/bash
#SBATCH --job-name=peft_adapt
#SBATCH --account=fdtbiotech
#SBATCH --partition=a30_normal_q
#SBATCH --gres=gpu:a30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/training/peft/peft_adapt_%j.out
#SBATCH --error=logs/training/peft/peft_adapt_%j.err

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

if [[ ! -f "${REPO_ROOT}/scripts/train_adaptation.py" ]] || \
   [[ ! -f "${REPO_ROOT}/configs/adaptation.yaml" ]] || \
   [[ ! -f "${REPO_ROOT}/configs/baseline_patch_x0_full_train.yaml" ]]; then
    echo "SLURM_SUBMIT_DIR does not appear to be the repository root:" >&2
    echo "${REPO_ROOT}" >&2
    echo "Submit this launcher with sbatch from the repository root." >&2
    exit 1
fi

cd "${REPO_ROOT}"

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

if [[ "$#" -lt 1 ]]; then
    echo "Usage: sbatch $0 <method> [additional trainer arguments...]" >&2
    echo "Methods:" >&2
    echo "  regional_lora" >&2
    echo "  dora" >&2
    echo "  lokr" >&2
    echo "  bitfit" >&2
    exit 2
fi

METHOD="$1"
shift

case "${METHOD}" in
    regional_lora|dora|lokr|bitfit)
        ;;
    *)
        echo "Unknown deterministic adaptation method: ${METHOD}" >&2
        exit 2
        ;;
esac

BASELINE_CONFIG="${BASELINE_CONFIG:-configs/baseline_patch_x0_full_train.yaml}"
ADAPTATION_CONFIG="${ADAPTATION_CONFIG:-configs/adaptation.yaml}"
BASE_CHECKPOINT="${BASE_CHECKPOINT:-checkpoints/baseline/full_train/final_patch_x0_diffusion_full_train.pt}"
H5_ROOT="${H5_ROOT:-/scratch/bhanug/archive/BraTS2020_H5_Rebuilt}"
MANIFEST="${MANIFEST:-/scratch/bhanug/tumor_inpainting_doc/data/BraTS/BraTS20 Training Metadata.csv}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-checkpoints/peft/${METHOD}/full_train}"

for required_file in \
    "${BASELINE_CONFIG}" \
    "${ADAPTATION_CONFIG}" \
    "${BASE_CHECKPOINT}" \
    "${MANIFEST}"
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
    "logs/training/peft/${METHOD}/full_train" \
    "${CHECKPOINT_DIR}"

echo "Deterministic PEFT full-training run"
echo "===================================="
echo "Host                     : $(hostname)"
echo "Date                     : $(date)"
echo "Git commit               : $(git rev-parse HEAD)"
echo "SLURM_JOB_ID             : ${SLURM_JOB_ID:-}"
echo "SLURM_JOB_PARTITION      : ${SLURM_JOB_PARTITION:-}"
echo "Method                   : ${METHOD}"
echo "Conda environment        : ${CONDA_DEFAULT_ENV:-}"
echo "Baseline config          : ${BASELINE_CONFIG}"
echo "Adaptation config        : ${ADAPTATION_CONFIG}"
echo "Base checkpoint          : ${BASE_CHECKPOINT}"
echo "H5 root                  : ${H5_ROOT}"
echo "Manifest                 : ${MANIFEST}"
echo "Checkpoint directory     : ${CHECKPOINT_DIR}"
echo "Epoch target             : 50"
echo

if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi
    echo
fi

python scripts/train_adaptation.py \
    --method "${METHOD}" \
    --baseline-config "${BASELINE_CONFIG}" \
    --adaptation-config "${ADAPTATION_CONFIG}" \
    --base-checkpoint "${BASE_CHECKPOINT}" \
    --h5-root "${H5_ROOT}" \
    --manifest "${MANIFEST}" \
    --checkpoint-dir "${CHECKPOINT_DIR}" \
    --epochs 50 \
    --device cuda \
    "$@"
