#!/bin/bash
#SBATCH --job-name=perturb_topology
#SBATCH --account=fdtbiotech
#SBATCH --partition=a30_normal_q
#SBATCH --gres=gpu:a30:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/analysis/baseline_mask_sensitivity/perturbation_topology_response/analysis_%j.out
#SBATCH --error=logs/analysis/baseline_mask_sensitivity/perturbation_topology_response/analysis_%j.err

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

if [[ ! -f "${REPO_ROOT}/scripts/analyze_baseline_perturbation_topology_response.py" ]] || \
   [[ ! -f "${REPO_ROOT}/outputs/baseline_mask_sensitivity/perturbation_response/baseline_perturbation_response_catalog.csv" ]]; then
    echo "SLURM_SUBMIT_DIR does not appear to contain the required Stage-8 inputs:" >&2
    echo "${REPO_ROOT}" >&2
    echo "Submit this launcher with sbatch from the repository root." >&2
    exit 1
fi

cd "${REPO_ROOT}"

mkdir -p \
    logs/analysis/baseline_mask_sensitivity/perturbation_topology_response \
    results/baseline_mask_sensitivity/perturbation_topology_response

PYTHON="${PYTHON:-${HOME}/.conda/envs/brats-nnunet/bin/python}"

if [[ ! -x "${PYTHON}" ]]; then
    echo "Required Python interpreter not found: ${PYTHON}" >&2
    exit 1
fi

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "Stage 8: perturbation / topology response analysis"
echo "=================================================="
echo "Host                     : $(hostname)"
echo "Date                     : $(date)"
echo "Git commit               : $(git rev-parse HEAD)"
echo "SLURM_JOB_ID             : ${SLURM_JOB_ID:-}"
echo "SLURM_JOB_PARTITION      : ${SLURM_JOB_PARTITION:-}"
echo "CPUs                     : ${SLURM_CPUS_PER_TASK:-}"
echo "Python                   : ${PYTHON}"
echo "Expected response rows   : 139,587"
echo "Expected baseline slices : 19,941"
echo "Expected volumes         : 369"
echo "Analysis radii           : -4, -2, -1, +1, +2, +4"
echo "Responses                : 4"
echo "Topology comparisons     : 24"
echo "Bootstrap replicates     : 10,000"
echo "Bootstrap seed           : 2026"
echo "Training                 : none"
echo "Model inference          : none"
echo "GPU computation          : none"
echo

"${PYTHON}" -u scripts/analyze_baseline_perturbation_topology_response.py \
    --bootstrap-replicates 10000 \
    --bootstrap-seed 2026 \
    "$@"

echo
echo "=================================================="
echo "STAGE 8 COMPLETE"
echo "=================================================="
echo "End time: $(date)"
