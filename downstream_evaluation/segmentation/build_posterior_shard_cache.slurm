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

cd "${SLURM_SUBMIT_DIR}"

mkdir -p \
  logs/training/downstream_segmentation/posterior_cache

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

MANIFEST="${POSTERIOR_CACHE_MANIFEST:-downstream_evaluation/manifests/br_lora_library_design_10000/br_lora_library_design_10000.csv}"
LIBRARY_ROOT="${POSTERIOR_CACHE_LIBRARY_ROOT:-/scratch/bhanug/br_lora_library/batches}"
OUTPUT_ROOT="${POSTERIOR_CACHE_OUTPUT_ROOT:-/scratch/bhanug/br_lora_library/downstream_seed42_epoch20_shards}"

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

python downstream_evaluation/segmentation/build_posterior_shard_cache.py \
  --manifest "${MANIFEST}" \
  --library-root "${LIBRARY_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --seed "${SEED}" \
  --epochs "${EPOCHS}" \
  --shard-size "${SHARD_SIZE}"
