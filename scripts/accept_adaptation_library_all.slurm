#!/bin/bash
#SBATCH --job-name=peft_accept
#SBATCH --account=fdtbiotech
#SBATCH --partition=a30_normal_q
#SBATCH --gres=gpu:a30:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=logs/production/peft/library/a30/accept_all_%j.out
#SBATCH --error=logs/production/peft/library/a30/accept_all_%j.err

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

if [[ ! -f scripts/accept_adaptation_library_batch.py ]]; then
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

METHODS=(
    regional_lora
    dora
    lokr
    bitfit
)

mkdir -p logs/production/peft/library

echo "======================================================================"
echo "DETERMINISTIC PEFT LIBRARY ACCEPTANCE"
echo "======================================================================"
echo "Host             : $(hostname)"
echo "Date             : $(date)"
echo "Git commit       : $(git rev-parse HEAD)"
echo "SLURM_JOB_ID     : ${SLURM_JOB_ID:-}"
echo "Python           : ${PYTHON}"
echo

LIBRARY_ROOTS=()

for method in "${METHODS[@]}"; do
    staging_root="$(read_folder_path "${method}_staging_root")"
    library_root="$(read_folder_path "${method}_library_root")"

    LIBRARY_ROOTS+=("${library_root}")

    echo
    echo "======================================================================"
    echo "METHOD: ${method}"
    echo "======================================================================"
    echo "Staging root     : ${staging_root}"
    echo "Library root     : ${library_root}"

    for n in $(seq 1 40); do
        batch="$(printf 'batch_%04d' "${n}")"

        accepted_batch="${library_root}/${batch}"
        acceptance_audit="${library_root}/_provenance/${method}_${batch}_acceptance.json"

        echo
        echo "----------------------------------------------------------------------"
        echo "${method} / ${batch}"
        echo "----------------------------------------------------------------------"

        if [[ -d "${accepted_batch}" && -f "${acceptance_audit}" ]]; then
            echo "Already accepted: SKIP"
            continue
        fi

        if [[ -d "${accepted_batch}" || -f "${acceptance_audit}" ]]; then
            echo "ERROR: partial accepted state detected." >&2
            echo "Accepted batch   : ${accepted_batch}" >&2
            echo "Acceptance audit : ${acceptance_audit}" >&2
            exit 1
        fi

        "${PYTHON}" scripts/accept_adaptation_library_batch.py \
            --batch "${batch}" \
            --method "${method}" \
            --staging-root "${staging_root}" \
            --library-root "${library_root}" \
            --folders-file "${FOLDERS_FILE}"

        if [[ ! -d "${accepted_batch}" ]]; then
            echo "ERROR: accepted batch was not created." >&2
            exit 1
        fi

        if [[ ! -f "${acceptance_audit}" ]]; then
            echo "ERROR: acceptance audit was not created." >&2
            exit 1
        fi

        echo "${method} / ${batch}: ACCEPTED"
    done
done

echo
echo "======================================================================"
echo "ALL DETERMINISTIC PEFT BATCHES ACCOUNTED FOR"
echo "======================================================================"

"${PYTHON}" - "${LIBRARY_ROOTS[@]}" <<'PY'
from pathlib import Path
import json
import re
import sys

methods = (
    "regional_lora",
    "dora",
    "lokr",
    "bitfit",
)

if len(sys.argv) != len(methods) + 1:
    raise SystemExit(
        "Expected one configured library root per adaptation method."
    )

library_roots = {
    method: Path(root).expanduser().resolve()
    for method, root in zip(methods, sys.argv[1:])
}

overall_ok = True

for method in methods:
    root = library_roots[method]
    provenance = root / "_provenance"

    batches = [
        root / f"batch_{n:04d}"
        for n in range(1, 41)
    ]

    missing_batches = [
        p.name for p in batches
        if not p.is_dir()
    ]

    case_ids = []

    for batch in batches:
        if batch.is_dir():
            case_ids.extend(
                sorted(
                    p.name
                    for p in batch.iterdir()
                    if p.is_dir()
                    and re.fullmatch(r"synthetic_\d{6}", p.name)
                )
            )

    missing_audits = []

    for n in range(1, 41):
        batch = f"batch_{n:04d}"
        audit = provenance / f"{method}_{batch}_acceptance.json"

        if not audit.is_file():
            missing_audits.append(batch)
            continue

        data = json.loads(audit.read_text())

        if (
            data.get("status") != "accepted"
            or data.get("adaptation_method") != method
            or data.get("accepted_case_count") != 250
        ):
            missing_audits.append(
                f"{batch}: invalid acceptance audit"
            )

    expected_ids = [
        f"synthetic_{i:06d}"
        for i in range(1, 10001)
    ]

    ok = (
        not missing_batches
        and not missing_audits
        and case_ids == expected_ids
        and len(set(case_ids)) == 10000
    )

    print()
    print(method)
    print("-" * len(method))
    print("accepted batches :", 40 - len(missing_batches))
    print("accepted cases   :", len(case_ids))
    print("unique case IDs  :", len(set(case_ids)))
    print("acceptance audits:", 40 - len(missing_audits))
    print("verdict          :", "PASS" if ok else "FAIL")

    if missing_batches:
        print("missing batches  :", missing_batches)

    if missing_audits:
        print("audit problems   :", missing_audits)

    if not ok:
        overall_ok = False

print()
print("=" * 70)
print("OVERALL VERDICT:", "PASS" if overall_ok else "FAIL")
print("=" * 70)

if not overall_ok:
    raise SystemExit(1)
PY

echo
echo "Finished: $(date)"
