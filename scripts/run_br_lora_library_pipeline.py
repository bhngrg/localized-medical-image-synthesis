#!/usr/bin/env python3
"""
End-to-end BR-LoRA frozen-library batch orchestrator.

This script assumes the 10,000-case library design is already frozen.

For one batch it:

1. runs the validated local batch-production script,
2. requires the local production audit and SHA-256 inventory,
3. transfers the completed batch to Falcon,
4. transfers the frozen batch manifest and execution manifest,
5. transfers the local audit/checksum records,
6. recomputes the Falcon SHA-256 inventory,
7. requires exact Mac/Falcon checksum equality,
8. optionally invokes the validated Falcon acceptance script,
9. optionally deletes the local staging batch only after acceptance passes.

It does NOT redesign cases or modify any frozen conditioning assignment.
"""

from __future__ import annotations

import argparse
import json
import shlex
import shutil
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

LOCAL_BATCH_RUNNER = (
    PROJECT_ROOT
    / "scripts/run_br_lora_library_batch.py"
)

DESIGN_BATCH_DIR = (
    PROJECT_ROOT
    / "downstream_evaluation/manifests/"
      "br_lora_library_design_10000/batches"
)

LOCAL_STAGING_ROOT = Path(
    "/Users/bhanugarg/archive/br_lora_library_staging"
)

REMOTE_HOST = "falcon"

REMOTE_LIBRARY_ROOT = (
    "/scratch/bhanug/br_lora_library"
)

REMOTE_REPO = (
    "/home/bhanug/localized-medical-image-synthesis"
)

EXPECTED_BATCH_SIZE = 250
EXPECTED_BATCH_FILE_COUNT = 1501

FIRST_BATCH = 2
LAST_BATCH = 40


class PipelineError(RuntimeError):
    """Raised when batch orchestration cannot continue safely."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate, transfer, and verify one frozen "
            "BR-LoRA library batch."
        )
    )

    parser.add_argument(
        "--batch",
        required=True,
        help="Batch identifier, e.g. batch_0005.",
    )

    parser.add_argument(
        "--accept",
        action="store_true",
        help=(
            "After checksum verification, invoke the Falcon "
            "acceptance script and promote the batch."
        ),
    )

    parser.add_argument(
        "--cleanup",
        action="store_true",
        help=(
            "Delete the local staging batch after successful "
            "Falcon acceptance. Requires --accept."
        ),
    )

    return parser.parse_args()


def parse_batch_id(value: str) -> tuple[str, int]:
    text = value.strip()

    if not text.startswith("batch_"):
        raise PipelineError(
            "--batch must use the form batch_0005."
        )

    suffix = text[len("batch_"):]

    if len(suffix) != 4 or not suffix.isdigit():
        raise PipelineError(
            "--batch must contain four numeric digits."
        )

    number = int(suffix)

    if not FIRST_BATCH <= number <= LAST_BATCH:
        raise PipelineError(
            f"Batch must be between batch_{FIRST_BATCH:04d} "
            f"and batch_{LAST_BATCH:04d}."
        )

    return f"batch_{number:04d}", number


def run(
    command: list[str],
    *,
    label: str,
    cwd: Path | None = None,
) -> None:
    print()
    print("=" * 78)
    print(label)
    print("=" * 78)
    print()
    print(
        " ".join(
            shlex.quote(part)
            for part in command
        )
    )
    print()

    subprocess.run(
        command,
        cwd=cwd,
        check=True,
    )


def require_file(
    path: Path,
    *,
    name: str,
) -> None:
    if not path.is_file():
        raise PipelineError(
            f"{name} is missing:\n{path}"
        )


def require_directory(
    path: Path,
    *,
    name: str,
) -> None:
    if not path.is_dir():
        raise PipelineError(
            f"{name} is missing:\n{path}"
        )


def rsync(
    sources: list[str],
    destination: str,
    *,
    label: str,
    progress: bool = False,
) -> None:
    command = [
        "rsync",
        "-avh",
        "--partial",
    ]

    if progress:
        command.append("--progress")

    command.extend(sources)

    command.append(
        f"{REMOTE_HOST}:{destination}"
    )

    run(
        command,
        cwd=PROJECT_ROOT,
        label=label,
    )


def ssh(
    remote_script: str,
    *,
    label: str,
) -> None:
    """
    Execute one multi-line Bash script on Falcon.

    The script is streamed over stdin rather than embedded in the SSH
    command line. This avoids shell-quoting problems and allows
    arbitrarily long remote scripts.
    """

    print()
    print("=" * 78)
    print(label)
    print("=" * 78)
    print()

    print("ssh", REMOTE_HOST, "bash -s")
    print()

    subprocess.run(
        [
            "ssh",
            REMOTE_HOST,
            "bash",
            "-s",
        ],
        input=remote_script,
        text=True,
        check=True,
    )


def main() -> None:
    args = parse_args()

    batch_id, batch_number = parse_batch_id(
        args.batch
    )

    if args.cleanup and not args.accept:
        raise PipelineError(
            "--cleanup requires --accept."
        )

    require_file(
        LOCAL_BATCH_RUNNER,
        name="Local batch runner",
    )

    frozen_manifest = (
        DESIGN_BATCH_DIR
        / f"{batch_id}_manifest.csv"
    )

    execution_manifest = (
        DESIGN_BATCH_DIR
        / f"{batch_id}_external_evaluation_manifest.csv"
    )

    require_file(
        frozen_manifest,
        name="Frozen batch manifest",
    )

    # The production runner will create/validate this if it does not
    # already exist.
    local_batch = (
        LOCAL_STAGING_ROOT
        / batch_id
    )

    production_audit = (
        LOCAL_STAGING_ROOT
        / f"br_lora_{batch_id}_production_audit.json"
    )

    mac_checksum = (
        LOCAL_STAGING_ROOT
        / f"br_lora_{batch_id}_mac_sha256.txt"
    )

    print()
    print("=" * 78)
    print("BR-LoRA FROZEN LIBRARY PIPELINE")
    print("=" * 78)
    print()
    print("Batch                    :", batch_id)
    print("Frozen design            :", frozen_manifest)
    print("Local staging            :", local_batch)
    print("Falcon library           :", REMOTE_LIBRARY_ROOT)
    print("Falcon acceptance        :", args.accept)
    print("Cleanup                  :", args.cleanup)

    # ------------------------------------------------------------
    # 1. Local production.
    # ------------------------------------------------------------

    local_command = [
        sys.executable,
        str(LOCAL_BATCH_RUNNER),
        "--batch",
        batch_id,
    ]

    # Resume only if a batch directory genuinely already exists.
    if local_batch.exists():
        print()
        print(
            "Existing local batch detected. "
            "Using strict resume/revalidation mode."
        )
        local_command.append("--resume")

    run(
        local_command,
        cwd=PROJECT_ROOT,
        label="STAGE 1 — LOCAL GENERATION + PRODUCTION AUDIT",
    )

    require_directory(
        local_batch,
        name="Completed local batch",
    )

    require_file(
        execution_manifest,
        name="Execution manifest",
    )

    require_file(
        production_audit,
        name="Production audit",
    )

    require_file(
        mac_checksum,
        name="Mac SHA-256 inventory",
    )

    with production_audit.open(
        "r",
        encoding="utf-8",
    ) as file:
        audit = json.load(file)

    if audit.get("status") != "pass":
        raise PipelineError(
            "Production audit does not report status=pass."
        )

    if audit.get("case_count") != EXPECTED_BATCH_SIZE:
        raise PipelineError(
            "Production audit does not report 250 cases."
        )

    # ------------------------------------------------------------
    # 2. Transfer batch.
    # ------------------------------------------------------------

    rsync(
        [
            str(local_batch) + "/",
        ],
        (
            f"{REMOTE_LIBRARY_ROOT}/"
            f"batches/{batch_id}/"
        ),
        label="STAGE 2 — TRANSFER BATCH TO FALCON",
        progress=True,
    )

    # ------------------------------------------------------------
    # 3. Transfer frozen/support manifests + audit records.
    # ------------------------------------------------------------

    rsync(
        [
            str(frozen_manifest),
            str(execution_manifest),
        ],
        (
            f"{REMOTE_LIBRARY_ROOT}/manifests/"
        ),
        label="STAGE 3 — TRANSFER MANIFESTS",
    )

    rsync(
        [
            str(production_audit),
            str(mac_checksum),
        ],
        (
            f"{REMOTE_LIBRARY_ROOT}/audits/"
        ),
        label="STAGE 3 — TRANSFER AUDIT RECORDS",
    )

    # ------------------------------------------------------------
    # 4. Recompute Falcon checksum and compare.
    # ------------------------------------------------------------

    remote_batch = (
        f"{REMOTE_LIBRARY_ROOT}/"
        f"batches/{batch_id}"
    )

    remote_mac_checksum = (
        f"{REMOTE_LIBRARY_ROOT}/audits/"
        f"br_lora_{batch_id}_mac_sha256.txt"
    )

    remote_falcon_checksum = (
        f"{REMOTE_LIBRARY_ROOT}/audits/"
        f"br_lora_{batch_id}_falcon_sha256.txt"
    )

    remote_verify = f"""
set -euo pipefail

BATCH={shlex.quote(remote_batch)}
MAC_HASH={shlex.quote(remote_mac_checksum)}
FALCON_HASH={shlex.quote(remote_falcon_checksum)}

test -d "$BATCH"
test -f "$MAC_HASH"

CASE_COUNT=$(find "$BATCH" \
  -mindepth 1 \
  -maxdepth 1 \
  -type d \
  -name 'synthetic_*' \
  | wc -l)

if [ "$CASE_COUNT" -ne {EXPECTED_BATCH_SIZE} ]; then
    echo "Expected {EXPECTED_BATCH_SIZE} case directories; observed $CASE_COUNT" >&2
    exit 1
fi

cd "$BATCH"

find . \
  -type f \
  -print0 \
  | sort -z \
  | xargs -0 sha256sum \
  > "$FALCON_HASH"

FILE_COUNT=$(wc -l < "$FALCON_HASH")

if [ "$FILE_COUNT" -ne {EXPECTED_BATCH_FILE_COUNT} ]; then
    echo "Expected {EXPECTED_BATCH_FILE_COUNT} files; observed $FILE_COUNT" >&2
    exit 1
fi

diff -u \
  "$MAC_HASH" \
  "$FALCON_HASH"

echo
echo "===== FALCON TRANSFER VERIFICATION ====="
echo "Case directories : $CASE_COUNT"
echo "Files checksummed: $FILE_COUNT"
echo "Mac/Falcon SHA256: IDENTICAL"
"""

    ssh(
        remote_verify,
        label="STAGE 4 — FALCON SHA-256 VERIFICATION",
    )

    # ------------------------------------------------------------
    # 5. Optional Falcon acceptance.
    # ------------------------------------------------------------

    if args.accept:
        remote_accept = f"""
set -euo pipefail

cd {shlex.quote(REMOTE_REPO)}

CONDA_BASE="$(conda info --base)"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate brats-nnunet

python \
  scripts/accept_br_lora_library_batch.py \
  --batch {shlex.quote(batch_id)}
"""

        ssh(
            remote_accept,
            label="STAGE 5 — FALCON ACCEPTANCE",
        )

    else:
        print()
        print("=" * 78)
        print("FALCON ACCEPTANCE NOT REQUESTED")
        print("=" * 78)
        print()
        print(
            "Transfer and checksum verification passed."
        )
        print(
            "Run again with --accept only after testing "
            "the Falcon-side acceptance path."
        )

    # ------------------------------------------------------------
    # 6. Optional cleanup.
    # ------------------------------------------------------------

    if args.cleanup:
        shutil.rmtree(
            local_batch
        )

        print()
        print("=" * 78)
        print("LOCAL CLEANUP COMPLETE")
        print("=" * 78)
        print()
        print("Deleted:", local_batch)

    print()
    print("=" * 78)
    print("PIPELINE STAGE COMPLETE")
    print("=" * 78)
    print()
    print("Batch                    :", batch_id)
    print("Local production         : PASS")
    print("Transfer                 : PASS")
    print("Mac/Falcon SHA-256       : IDENTICAL")
    print(
        "Falcon acceptance        :",
        "PASS" if args.accept else "NOT RUN",
    )
    print(
        "Local cleanup            :",
        "COMPLETE" if args.cleanup else "NOT RUN",
    )

    if batch_number < LAST_BATCH:
        print(
            "Next batch               :",
            f"batch_{batch_number + 1:04d}",
        )


if __name__ == "__main__":
    try:
        main()

    except (
        PipelineError,
        FileNotFoundError,
        OSError,
        subprocess.CalledProcessError,
        json.JSONDecodeError,
        ValueError,
    ) as exc:
        print()
        print("=" * 78, file=sys.stderr)
        print(
            "BR-LoRA FROZEN LIBRARY PIPELINE: FAILED",
            file=sys.stderr,
        )
        print("=" * 78, file=sys.stderr)
        print()
        print(exc, file=sys.stderr)
        print()
        print(
            "No cleanup was performed after failure.",
            file=sys.stderr,
        )
        sys.exit(1)
