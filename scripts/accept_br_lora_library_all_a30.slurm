#!/bin/bash
#SBATCH --job-name=br_lora_accept
#SBATCH --account=fdtbiotech
#SBATCH --partition=a30_normal_q
#SBATCH --gres=gpu:a30:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=logs/production/br_lora/library/a30/accept_all_%j.out
#SBATCH --error=logs/production/br_lora/library/a30/accept_all_%j.err

set -euo pipefail

if [[ -z "${SLURM_SUBMIT_DIR:-}" ]]; then
    echo "SLURM_SUBMIT_DIR is not available." >&2
    exit 1
fi

REPO_ROOT="$(
    cd "${SLURM_SUBMIT_DIR}"
    pwd
)"

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

if [[ ! -f scripts/accept_br_lora_library_batch.py ]]; then
    echo "Acceptance script not found." >&2
    exit 1
fi

CONDA_ENV="${CONDA_ENV:-brats-nnunet}"

if [[ "${CONDA_DEFAULT_ENV:-}" != "${CONDA_ENV}" ]]; then
    if command -v conda >/dev/null 2>&1; then
        eval "$(conda shell.bash hook)"
    elif [[ -n "${CONDA_EXE:-}" && -x "${CONDA_EXE}" ]]; then
        eval "$("${CONDA_EXE}" shell.bash hook)"
    else
        echo "Conda is not available." >&2
        exit 1
    fi

    conda activate "${CONDA_ENV}"
fi

PYTHON="${PYTHON:-$(command -v python)}"

if [[ -z "${PYTHON}" || ! -x "${PYTHON}" ]]; then
    echo "Required Python interpreter not found." >&2
    exit 1
fi

if [[ -z "${STAGING_ROOT:-}" ]]; then
    STAGING_ROOT="$(read_folder_path br_lora_staging_root)"
fi

if [[ -z "${LIBRARY_ROOT:-}" ]]; then
    LIBRARY_ROOT="$(read_folder_path br_lora_library_root)"
fi

MASTER="${LIBRARY_ROOT}/manifests/br_lora_library_manifest.csv"

mkdir -p logs/production/br_lora/library/a30

echo "======================================================================"
echo "BR-LoRA A30 LIBRARY ACCEPTANCE"
echo "======================================================================"
echo "Host             : $(hostname)"
echo "Date             : $(date)"
echo "Git commit       : $(git rev-parse HEAD)"
echo "SLURM_JOB_ID     : ${SLURM_JOB_ID:-}"
echo "Python           : ${PYTHON}"
echo "Staging root     : ${STAGING_ROOT}"
echo "Library root     : ${LIBRARY_ROOT}"
echo

for n in $(seq 1 40); do
    batch="$(printf 'batch_%04d' "${n}")"
    expected_after=$((n * 250))

    echo
    echo "======================================================================"
    echo "CHECKING ${batch}"
    echo "======================================================================"

    #
    # Determine whether this batch has already been accepted.
    #
    rows=0

    if [[ -f "${MASTER}" ]]; then
        rows="$(
            "${PYTHON}" - "${MASTER}" <<'PY'
import sys
import pandas as pd

path = sys.argv[1]
print(len(pd.read_csv(path)))
PY
        )"
    fi

    if (( rows >= expected_after )); then
        echo "${batch}: already represented in master."
        echo "Current master rows: ${rows}"
        echo "${batch}: SKIP"
        continue
    fi

    expected_before=$(((n - 1) * 250))

    if (( rows != expected_before )); then
        echo "ERROR: master is not in the required pre-state." >&2
        echo "Batch           : ${batch}" >&2
        echo "Expected rows   : ${expected_before}" >&2
        echo "Observed rows   : ${rows}" >&2
        exit 1
    fi

    echo "Master rows before: ${rows}"
    echo "Accepting ${batch}..."

    "${PYTHON}" scripts/accept_br_lora_library_batch.py \
        --batch "${batch}" \
        --staging-root "${STAGING_ROOT}" \
        --library-root "${LIBRARY_ROOT}" \
        --folders-file "${FOLDERS_FILE}"

    rows_after="$(
        "${PYTHON}" - "${MASTER}" <<'PY'
import sys
import pandas as pd

path = sys.argv[1]
print(len(pd.read_csv(path)))
PY
    )"

    if (( rows_after != expected_after )); then
        echo "ERROR: unexpected master size after acceptance." >&2
        echo "Batch           : ${batch}" >&2
        echo "Expected rows   : ${expected_after}" >&2
        echo "Observed rows   : ${rows_after}" >&2
        exit 1
    fi

    echo "${batch}: ACCEPTED"
done

echo
echo "======================================================================"
echo "ALL BR-LoRA BATCHES ACCOUNTED FOR"
echo "======================================================================"

"${PYTHON}" - "${MASTER}" <<'PY'
import sys
import pandas as pd

path = sys.argv[1]
df = pd.read_csv(path)

print("Master rows        :", len(df))
print("First library ID   :", df["library_case_id"].iloc[0])
print("Last library ID    :", df["library_case_id"].iloc[-1])
print("Unique IDs         :", df["library_case_id"].nunique())
print("Unique pair keys   :", df["pair_key"].nunique())
print("Unique donor slices:", df["donor_h5_file"].nunique())

assert len(df) == 10000
assert df["library_case_id"].nunique() == 10000
assert df["pair_key"].nunique() == 10000
assert df["donor_h5_file"].nunique() == 10000

print()
print("FINAL MASTER SIZE CHECK: PASS")
PY

echo
echo "Finished: $(date)"
