#!/bin/bash
#SBATCH --job-name=baseline_perturb
#SBATCH --account=fdtbiotech
#SBATCH --partition=a30_normal_q
#SBATCH --gres=gpu:a30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/analysis/baseline_mask_sensitivity/perturbation_response/perturbation_%j.out
#SBATCH --error=logs/analysis/baseline_mask_sensitivity/perturbation_response/perturbation_%j.err

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

if [[ ! -f "${REPO_ROOT}/scripts/analyze_baseline_perturbation_response.py" ]] || \
   [[ ! -f "${REPO_ROOT}/checkpoints/baseline/full_train/final_patch_x0_diffusion_full_train.pt" ]] || \
   [[ ! -f "${REPO_ROOT}/outputs/baseline_mask_sensitivity/signed_distance/signed_distance_perturbation_catalog.csv" ]] || \
   [[ ! -f "${REPO_ROOT}/outputs/baseline_mask_sensitivity/signed_distance/signed_distance_topology_audit.csv" ]] || \
   [[ ! -f "${REPO_ROOT}/data/folders.yaml" ]]; then
    echo "SLURM_SUBMIT_DIR does not appear to contain the required Stage-7 inputs:" >&2
    echo "${REPO_ROOT}" >&2
    echo "Submit this launcher with sbatch from the repository root." >&2
    exit 1
fi

cd "${REPO_ROOT}"

mkdir -p \
    logs/analysis/baseline_mask_sensitivity/perturbation_response \
    outputs/baseline_mask_sensitivity/perturbation_response

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

echo "Stage 7: baseline perturbation-response analysis"
echo "================================================"
echo "Host                     : $(hostname)"
echo "Date                     : $(date)"
echo "Git commit               : $(git rev-parse HEAD)"
echo "SLURM_JOB_ID             : ${SLURM_JOB_ID:-}"
echo "SLURM_JOB_PARTITION      : ${SLURM_JOB_PARTITION:-}"
echo "CPUs                     : ${SLURM_CPUS_PER_TASK:-}"
echo "Conda environment        : ${CONDA_DEFAULT_ENV:-}"
echo "Expected baseline slices : 19,941"
echo "Selected radii           : -4, -2, -1, 0, +1, +2, +4"
echo "Conditions per slice     : 7"
echo "Full-cohort conditions   : 139,587"
echo "Inference timestep       : 150"
echo "Noise seed base          : 2026"
echo "Noise pairing            : fixed within slice"
echo "Appearance conditioning  : fixed from original mask"
echo "Training                 : none"
echo

if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi
    echo
fi

python -u scripts/analyze_baseline_perturbation_response.py \
    --device cuda \
    "$@"

echo
echo "================================================"
echo "STAGE 7 COMPLETE"
echo "================================================"
echo "End time: $(date)"
