#!/usr/bin/env python3
"""
Accept one completed BR-LoRA production batch into the library.

Acceptance requires:

1. the canonical frozen batch manifest,
2. the execution manifest,
3. a successful production audit,
4. an exact match to the production SHA-256 inventory,
5. exactly 250 complete case directories,
6. consistency between generated metadata and the frozen design, and
7. preservation of all existing master-library rows.

Only after all checks pass is the canonical master manifest extended.

The script never deletes batch data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_DESIGN_BATCH_DIR = (
    PROJECT_ROOT
    / "downstream_evaluation/manifests/"
      "br_lora_library_design_10000/batches"
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )

from src.config import (
    load_folders_config,
    resolve_path,
    save_folders_config,
)

EXPECTED_BATCH_SIZE = 250
FIRST_PRODUCTION_BATCH = 2
LAST_LIBRARY_BATCH = 40

EXPECTED_CASE_ARTIFACTS = (
    "posterior_samples.pt",
    "posterior_mean.pt",
    "posterior_variance.pt",
    "posterior_std.pt",
    "composite_mean.pt",
    "metadata.json",
)


class BatchAcceptanceError(
    RuntimeError
):
    """Raised when a completed batch cannot be accepted."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Verify and promote one completed BR-LoRA "
            "batch into the master library."
        )
    )

    parser.add_argument(
        "--batch",
        required=True,
        help="Batch identifier, for example batch_0003.",
    )

    parser.add_argument(
        "--design-batch-dir",
        type=Path,
        default=DEFAULT_DESIGN_BATCH_DIR,
    )

    parser.add_argument(
        "--staging-root",
        type=Path,
        default=None,
        help=(
            "BR-LoRA staging root. When supplied, the path "
            "is saved in the folders configuration for reuse."
        ),
    )

    parser.add_argument(
        "--library-root",
        type=Path,
        default=None,
        help=(
            "BR-LoRA library root. When supplied, the path "
            "is saved in the folders configuration for reuse."
        ),
    )

    parser.add_argument(
        "--folders-file",
        type=Path,
        default=Path("data/folders.yaml"),
        help="Machine-specific folders configuration file.",
    )

    return parser.parse_args()


def resolve_acceptance_paths(
    args: argparse.Namespace,
) -> tuple[Path, Path]:
    config = load_folders_config(
        args.folders_file
    )

    staging_root = resolve_path(
        key="br_lora_staging_root",
        cli_value=args.staging_root,
        config=config,
        selector=None,
    )

    library_root = resolve_path(
        key="br_lora_library_root",
        cli_value=args.library_root,
        config=config,
        selector=None,
    )

    save_folders_config(
        args.folders_file,
        config,
    )

    return (
        staging_root,
        library_root,
    )


def parse_batch_id(
    value: str,
) -> tuple[str, int]:
    text = value.strip()

    if not text.startswith("batch_"):
        raise BatchAcceptanceError(
            "--batch must use the form batch_0003."
        )

    suffix = text[len("batch_"):]

    if (
        len(suffix) != 4
        or not suffix.isdigit()
    ):
        raise BatchAcceptanceError(
            "--batch must contain four numeric digits."
        )

    number = int(suffix)

    if not (
        FIRST_PRODUCTION_BATCH
        <= number
        <= LAST_LIBRARY_BATCH
    ):
        raise BatchAcceptanceError(
            f"Batch number must lie between "
            f"{FIRST_PRODUCTION_BATCH:04d} and "
            f"{LAST_LIBRARY_BATCH:04d}."
        )

    return (
        f"batch_{number:04d}",
        number,
    )


def require_file(
    path: Path,
    *,
    name: str,
) -> Path:
    if not path.is_file():
        raise FileNotFoundError(
            f"{name} does not exist:\n{path}"
        )

    return path


def require_directory(
    path: Path,
    *,
    name: str,
) -> Path:
    if not path.is_dir():
        raise NotADirectoryError(
            f"{name} does not exist:\n{path}"
        )

    return path


def sha256_file(
    path: Path,
) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def load_checksum_inventory(
    path: Path,
) -> dict[str, str]:
    values: dict[str, str] = {}

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line_number, line in enumerate(
            file,
            start=1,
        ):
            line = line.rstrip("\n")

            if not line:
                continue

            try:
                digest, relative = line.split(
                    "  ",
                    1,
                )
            except ValueError as exc:
                raise BatchAcceptanceError(
                    f"Malformed checksum line "
                    f"{line_number} in {path}."
                ) from exc

            if (
                len(digest) != 64
                or any(
                    char not in "0123456789abcdef"
                    for char in digest.lower()
                )
            ):
                raise BatchAcceptanceError(
                    f"Invalid SHA-256 digest on line "
                    f"{line_number} in {path}."
                )

            if relative in values:
                raise BatchAcceptanceError(
                    f"Duplicate checksum path in {path}: "
                    f"{relative}"
                )

            values[relative] = digest.lower()

    return values



def compute_batch_checksums(
    batch_root: Path,
) -> dict[str, str]:
    files = sorted(
        path
        for path in batch_root.rglob("*")
        if path.is_file()
    )

    if not files:
        raise BatchAcceptanceError(
            "Cannot checksum an empty batch directory."
        )

    return {
        f"./{path.relative_to(batch_root).as_posix()}":
            sha256_file(path)
        for path in files
    }


def verify_existing_master(
    master: pd.DataFrame,
) -> None:
    required = {
        "library_index",
        "library_case_id",
        "batch_id",
        "source_case_id",
        "pair_key",
        "external_subject_name",
        "external_subject_numeric_id",
        "external_slice_index",
        "donor_h5_file",
        "donor_volume",
        "donor_slice_index",
        "mask_pixels",
        "image_relative_path",
        "label_container_relative_path",
        "metadata_relative_path",
        "label_tensor_key",
        "image_height",
        "image_width",
        "image_channel",
        "image_role",
        "label_role",
        "generator_checkpoint",
        "generator_git_commit",
        "posterior_samples",
    }

    missing = required - set(master.columns)

    if missing:
        raise BatchAcceptanceError(
            "Current master is missing required columns:\n"
            + "\n".join(sorted(missing))
        )

    if master.empty:
        raise BatchAcceptanceError(
            "Current master is unexpectedly empty."
        )

    if len(master) % EXPECTED_BATCH_SIZE != 0:
        raise BatchAcceptanceError(
            "Current master row count is not divisible by 250."
        )

    if master["library_index"].duplicated().any():
        raise BatchAcceptanceError(
            "Current master contains duplicate library_index."
        )

    if master["library_case_id"].duplicated().any():
        raise BatchAcceptanceError(
            "Current master contains duplicate library_case_id."
        )

    if master["pair_key"].duplicated().any():
        raise BatchAcceptanceError(
            "Current master contains duplicate pair_key."
        )

    if master["donor_h5_file"].duplicated().any():
        raise BatchAcceptanceError(
            "Current master contains duplicate donor_h5_file."
        )

    expected_indices = list(
        range(
            1,
            len(master) + 1,
        )
    )

    observed_indices = (
        master["library_index"]
        .astype(int)
        .tolist()
    )

    if observed_indices != expected_indices:
        raise BatchAcceptanceError(
            "Current master library_index is not contiguous."
        )

    expected_ids = [
        f"synthetic_{index:06d}"
        for index in expected_indices
    ]

    observed_ids = (
        master["library_case_id"]
        .astype(str)
        .tolist()
    )

    if observed_ids != expected_ids:
        raise BatchAcceptanceError(
            "Current master library_case_id sequence "
            "is not contiguous."
        )


def build_new_master_rows(
    *,
    batch_id: str,
    batch_number: int,
    design: pd.DataFrame,
    batch_root: Path,
    master_columns: list[str],
) -> pd.DataFrame:
    if len(design) != EXPECTED_BATCH_SIZE:
        raise BatchAcceptanceError(
            f"{batch_id} design must contain 250 rows."
        )

    if set(
        design["batch_id"].astype(str)
    ) != {batch_id}:
        raise BatchAcceptanceError(
            "Incoming design contains an unexpected batch_id."
        )

    expected_first = (
        (batch_number - 1)
        * EXPECTED_BATCH_SIZE
        + 1
    )

    expected_last = (
        batch_number
        * EXPECTED_BATCH_SIZE
    )

    expected_indices = list(
        range(
            expected_first,
            expected_last + 1,
        )
    )

    observed_indices = (
        design["library_index"]
        .astype(int)
        .tolist()
    )

    if observed_indices != expected_indices:
        raise BatchAcceptanceError(
            f"{batch_id} library indices are not exactly "
            f"{expected_first}-{expected_last}."
        )

    expected_ids = [
        f"synthetic_{index:06d}"
        for index in expected_indices
    ]

    observed_ids = (
        design["library_case_id"]
        .astype(str)
        .tolist()
    )

    if observed_ids != expected_ids:
        raise BatchAcceptanceError(
            f"{batch_id} library_case_id sequence is incorrect."
        )

    records = []

    for row in design.itertuples(index=False):
        case_id = str(row.library_case_id)

        case_dir = (
            batch_root
            / case_id
        )

        if not case_dir.is_dir():
            raise BatchAcceptanceError(
                f"{case_id}: missing case directory."
            )

        for artifact in EXPECTED_CASE_ARTIFACTS:
            artifact_path = (
                case_dir
                / artifact
            )

            if not artifact_path.is_file():
                raise BatchAcceptanceError(
                    f"{case_id}: missing {artifact}."
                )

        metadata_path = (
            case_dir
            / "metadata.json"
        )

        with metadata_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            metadata = json.load(file)

        expected_metadata = {
            "case_id":
                case_id,

            "external_subject_numeric_id":
                int(
                    row.external_subject_numeric_id
                ),

            "external_slice_index":
                int(
                    row.external_slice_index
                ),

            "external_modality":
                str(
                    row.external_modality
                ).lower(),

            "posterior_samples":
                100,

            "evaluation_seed":
                42,

            "resample_diffusion_noise":
                False,
        }

        for key, expected in expected_metadata.items():
            observed = metadata.get(key)

            if observed != expected:
                raise BatchAcceptanceError(
                    f"{case_id}: metadata {key} mismatch.\n"
                    f"Expected: {expected!r}\n"
                    f"Observed: {observed!r}"
                )

        checkpoint = metadata.get(
            "checkpoint"
        )

        git_commit = metadata.get(
            "git_commit"
        )

        if not checkpoint:
            raise BatchAcceptanceError(
                f"{case_id}: metadata does not record checkpoint."
            )

        record = {
            "library_index":
                int(row.library_index),

            "library_case_id":
                case_id,

            "batch_id":
                batch_id,

            "source_case_id":
                case_id,

            "pair_key":
                str(row.pair_key),

            "external_subject_name":
                str(row.external_subject_name),

            "external_subject_numeric_id":
                int(
                    row.external_subject_numeric_id
                ),

            "external_slice_index":
                int(
                    row.external_slice_index
                ),

            "donor_h5_file":
                str(row.donor_h5_file),

            "donor_volume":
                int(row.donor_volume),

            "donor_slice_index":
                int(row.donor_slice_index),

            "mask_pixels":
                int(row.donor_mask_pixels),

            "image_relative_path":
                str(
                    Path("batches")
                    / batch_id
                    / case_id
                    / "composite_mean.pt"
                ),

            "label_container_relative_path":
                str(
                    Path("batches")
                    / batch_id
                    / case_id
                    / "posterior_samples.pt"
                ),

            "metadata_relative_path":
                str(
                    Path("batches")
                    / batch_id
                    / case_id
                    / "metadata.json"
                ),

            "label_tensor_key":
                "transferred_mask",

            "image_height":
                240,

            "image_width":
                240,

            "image_channel":
                "FLAIR",

            "image_role":
                "synthetic",

            "label_role":
                "prescribed_whole_tumor_mask",

            "generator_checkpoint":
                str(checkpoint),

            "generator_git_commit":
                git_commit,

            "posterior_samples":
                100,
        }

        records.append(record)

    new_rows = pd.DataFrame(records)

    missing = [
        column
        for column in master_columns
        if column not in new_rows.columns
    ]

    if missing:
        raise BatchAcceptanceError(
            "Incoming rows cannot reproduce the master schema.\n"
            f"Missing: {missing}"
        )

    return new_rows[
        master_columns
    ].copy()


def verify_artifact_references(
    master: pd.DataFrame,
    *,
    library_root: Path,
) -> None:
    for row in master.itertuples(index=False):
        case_id = str(row.library_case_id)

        for column in (
            "image_relative_path",
            "label_container_relative_path",
            "metadata_relative_path",
        ):
            relative = getattr(
                row,
                column,
            )

            path = (
                library_root
                / str(relative)
            )

            if not path.is_file():
                raise BatchAcceptanceError(
                    f"{case_id}: missing master-referenced "
                    f"artifact for {column}:\n{path}"
                )


def atomic_write_csv(
    table: pd.DataFrame,
    path: Path,
) -> None:
    temporary = path.with_name(
        path.name
        + ".tmp"
    )

    table.to_csv(
        temporary,
        index=False,
    )

    os.replace(
        temporary,
        path,
    )


def main() -> None:
    args = parse_args()

    (
        staging_root,
        library_root,
    ) = resolve_acceptance_paths(
        args
    )

    design_batch_dir = require_directory(
        args.design_batch_dir,
        name="Frozen design batch directory",
    )

    staging_root = (
        staging_root
        .expanduser()
        .resolve()
    )

    library_root = (
        library_root
        .expanduser()
        .resolve()
    )

    batch_id, batch_number = parse_batch_id(
        args.batch
    )

    source_batch_root = require_directory(
        staging_root
        / batch_id,
        name="Completed staging batch directory",
    )

    design_path = require_file(
        design_batch_dir
        / f"{batch_id}_manifest.csv",
        name="Frozen batch manifest",
    )

    execution_path = require_file(
        staging_root
        / f"br_lora_{batch_id}_external_evaluation_manifest.csv",
        name="Execution manifest",
    )

    production_audit_path = require_file(
        staging_root
        / f"br_lora_{batch_id}_production_audit.json",
        name="Production audit",
    )

    checksum_path = require_file(
        staging_root
        / f"br_lora_{batch_id}_sha256.txt",
        name="Production checksum inventory",
    )

    master_path = require_file(
        library_root
        / "manifests"
        / "br_lora_library_manifest.csv",
        name="Current master library manifest",
    )

    print()
    print("=" * 78)
    print("BR-LoRA BATCH ACCEPTANCE")
    print("=" * 78)
    print()
    print("Batch                    :", batch_id)
    print("Staging batch            :", source_batch_root)
    print("Current master           :", master_path)

    # ------------------------------------------------------------
    # Batch-integrity acceptance.
    # ------------------------------------------------------------

    expected_checksums = load_checksum_inventory(
        checksum_path
    )

    observed_checksums = compute_batch_checksums(
        source_batch_root
    )

    if expected_checksums != observed_checksums:
        raise BatchAcceptanceError(
            "Current batch does not match the production "
            "checksum inventory."
        )

    if len(expected_checksums) != 1501:
        raise BatchAcceptanceError(
            "Expected exactly 1,501 batch files; "
            f"observed {len(expected_checksums)}."
        )

    print()
    print("===== BATCH INTEGRITY CHECK =====")
    print("Files                    :", len(expected_checksums))
    print("Production SHA-256       : MATCH")

    # ------------------------------------------------------------
    # Production-audit acceptance.
    # ------------------------------------------------------------

    with production_audit_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        production_audit = json.load(file)

    if production_audit.get(
        "status"
    ) != "pass":
        raise BatchAcceptanceError(
            "Production audit does not report status=pass."
        )

    if production_audit.get(
        "case_count"
    ) != EXPECTED_BATCH_SIZE:
        raise BatchAcceptanceError(
            "Production audit does not report 250 cases."
        )

    if production_audit.get(
        "problems"
    ) != 0:
        raise BatchAcceptanceError(
            "Production audit reports one or more problems."
        )

    if production_audit.get(
        "posterior_samples_per_case"
    ) != 100:
        raise BatchAcceptanceError(
            "Production audit does not report "
            "100 posterior samples per case."
        )

    if production_audit.get(
        "evaluation_seed"
    ) != 42:
        raise BatchAcceptanceError(
            "Production audit does not report seed 42."
        )

    observed_execution_sha = sha256_file(
        execution_path
    )

    expected_execution_sha = (
        production_audit.get(
            "execution_manifest_sha256"
        )
    )

    if (
        observed_execution_sha
        != expected_execution_sha
    ):
        raise BatchAcceptanceError(
            "Execution manifest hash does not match the "
            "production audit."
        )

    print()
    print("===== PRODUCTION AUDIT =====")
    print("Status                   : PASS")
    print("Cases                    : 250")
    print("Posterior samples        : 100")
    print("Evaluation seed          : 42")
    print("Execution manifest hash  : MATCH")

    # ------------------------------------------------------------
    # Master pre-state.
    # ------------------------------------------------------------

    master = pd.read_csv(
        master_path
    )

    verify_existing_master(
        master
    )

    expected_existing_rows = (
        (batch_number - 1)
        * EXPECTED_BATCH_SIZE
    )

    if len(master) != expected_existing_rows:
        raise BatchAcceptanceError(
            f"{batch_id} cannot be accepted out of order.\n"
            f"Expected current master rows: "
            f"{expected_existing_rows}\n"
            f"Observed current master rows: {len(master)}"
        )

    if batch_id in set(
        master["batch_id"].astype(str)
    ):
        raise BatchAcceptanceError(
            f"{batch_id} is already present in the master."
        )

    current_master_sha = sha256_file(
        master_path
    )

    print()
    print("===== MASTER PRE-STATE =====")
    print("Rows                     :", len(master))
    print(
        "Library IDs              :",
        master["library_case_id"].iloc[0],
        "to",
        master["library_case_id"].iloc[-1],
    )
    print(
        "Current master SHA-256   :",
        current_master_sha,
    )

    # ------------------------------------------------------------
    # Incoming design and generated outputs.
    # ------------------------------------------------------------

    design = pd.read_csv(
        design_path
    )

    new_rows = build_new_master_rows(
        batch_id=batch_id,
        batch_number=batch_number,
        design=design,
        batch_root=source_batch_root,
        master_columns=list(master.columns),
    )

    if set(
        new_rows["library_case_id"].astype(str)
    ) & set(
        master["library_case_id"].astype(str)
    ):
        raise BatchAcceptanceError(
            "Incoming library IDs overlap the existing master."
        )

    if set(
        new_rows["pair_key"].astype(str)
    ) & set(
        master["pair_key"].astype(str)
    ):
        raise BatchAcceptanceError(
            "Incoming pair keys overlap the existing master."
        )

    if set(
        new_rows["donor_h5_file"].astype(str)
    ) & set(
        master["donor_h5_file"].astype(str)
    ):
        raise BatchAcceptanceError(
            "Incoming donor slices overlap the existing master."
        )

    # ------------------------------------------------------------
    # Construct candidate new master.
    # ------------------------------------------------------------

    updated = pd.concat(
        [
            master,
            new_rows,
        ],
        ignore_index=True,
    )

    expected_new_rows = (
        batch_number
        * EXPECTED_BATCH_SIZE
    )

    if len(updated) != expected_new_rows:
        raise BatchAcceptanceError(
            "Candidate master has an unexpected number of rows."
        )

    verify_existing_master(
        updated
    )

    # Existing rows must remain exactly unchanged.
    try:
        pd.testing.assert_frame_equal(
            updated
            .iloc[:len(master)]
            .reset_index(drop=True),

            master
            .reset_index(drop=True),

            check_dtype=False,
            check_exact=True,
        )

    except AssertionError as exc:
        raise BatchAcceptanceError(
            "Candidate master modifies previously accepted rows."
        ) from exc

    expected_batch_counts = {
        f"batch_{number:04d}":
            EXPECTED_BATCH_SIZE
        for number in range(
            1,
            batch_number + 1,
        )
    }

    observed_batch_counts = (
        updated["batch_id"]
        .value_counts()
        .sort_index()
        .to_dict()
    )

    if (
        observed_batch_counts
        != expected_batch_counts
    ):
        raise BatchAcceptanceError(
            "Candidate master has unexpected batch counts.\n"
            f"Observed: {observed_batch_counts}"
        )

    for directory in (
        library_root / "batches",
        library_root / "manifests",
        library_root / "audits",
    ):
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    batch_root = (
        library_root
        / "batches"
        / batch_id
    )

    library_design_path = (
        library_root
        / "manifests"
        / design_path.name
    )

    library_execution_path = (
        library_root
        / "manifests"
        / execution_path.name
    )

    library_audit_path = (
        library_root
        / "audits"
        / production_audit_path.name
    )

    library_checksum_path = (
        library_root
        / "audits"
        / checksum_path.name
    )

    snapshot_path = (
        library_root
        / "manifests"
        / (
            "br_lora_library_manifest_"
            f"{len(master):05d}_pre_{batch_id}.csv"
        )
    )

    acceptance_path = (
        library_root
        / "audits"
        / f"br_lora_{batch_id}_acceptance.json"
    )

    master_hash_path = (
        library_root
        / "audits"
        / (
            "br_lora_library_manifest_"
            f"{expected_new_rows}_sha256.txt"
        )
    )

    destinations = (
        batch_root,
        library_design_path,
        library_execution_path,
        library_audit_path,
        library_checksum_path,
        snapshot_path,
        acceptance_path,
        master_hash_path,
    )

    existing = [
        path
        for path in destinations
        if path.exists()
    ]

    if existing:
        raise BatchAcceptanceError(
            "One or more library destinations already exist:\n"
            + "\n".join(
                str(path)
                for path in existing
            )
        )

    shutil.copytree(
        source_batch_root,
        batch_root,
    )

    if (
        compute_batch_checksums(
            batch_root
        )
        != expected_checksums
    ):
        raise BatchAcceptanceError(
            "Copied library batch does not match the "
            "production checksum inventory."
        )

    for source, destination in (
        (design_path, library_design_path),
        (execution_path, library_execution_path),
        (production_audit_path, library_audit_path),
        (checksum_path, library_checksum_path),
    ):
        shutil.copy2(
            source,
            destination,
        )

        if (
            sha256_file(source)
            != sha256_file(destination)
        ):
            raise BatchAcceptanceError(
                "Copied library artifact hash mismatch:\n"
                f"{destination}"
            )

    verify_artifact_references(
        updated,
        library_root=library_root,
    )

    # ------------------------------------------------------------
    # Preserve the exact pre-promotion master.
    # ------------------------------------------------------------

    shutil.copy2(
        master_path,
        snapshot_path,
    )

    if sha256_file(
        snapshot_path
    ) != current_master_sha:
        raise BatchAcceptanceError(
            "Pre-promotion snapshot hash does not match "
            "the current master."
        )

    # ------------------------------------------------------------
    # Promote atomically.
    # ------------------------------------------------------------

    atomic_write_csv(
        updated,
        master_path,
    )

    new_master_sha = sha256_file(
        master_path
    )

    # ------------------------------------------------------------
    # Re-read and independently validate written master.
    # ------------------------------------------------------------

    written = pd.read_csv(
        master_path
    )

    verify_existing_master(
        written
    )

    if len(written) != expected_new_rows:
        raise BatchAcceptanceError(
            "Written master row count is incorrect after promotion."
        )

    try:
        pd.testing.assert_frame_equal(
            written,
            updated,
            check_dtype=False,
            check_exact=True,
        )

    except AssertionError as exc:
        raise BatchAcceptanceError(
            "Written master differs from validated candidate master."
        ) from exc

    verify_artifact_references(
        written,
        library_root=library_root,
    )

    # ------------------------------------------------------------
    # Permanent acceptance audit.
    # ------------------------------------------------------------

    acceptance = {
        "status":
            "accepted",

        "batch_id":
            batch_id,

        "accepted_case_count":
            EXPECTED_BATCH_SIZE,

        "master_rows_before":
            int(len(master)),

        "master_rows_after":
            int(len(written)),

        "master_sha256_before":
            current_master_sha,

        "master_sha256_after":
            new_master_sha,

        "pre_promotion_snapshot":
            str(snapshot_path),

        "pre_promotion_snapshot_sha256":
            sha256_file(
                snapshot_path
            ),

        "frozen_batch_manifest":
            str(library_design_path),

        "frozen_batch_manifest_sha256":
            sha256_file(
                library_design_path
            ),

        "execution_manifest":
            str(library_execution_path),

        "execution_manifest_sha256":
            observed_execution_sha,

        "checksum_inventory":
            str(library_checksum_path),

        "checksum_inventory_sha256":
            sha256_file(
                library_checksum_path
            ),

        "checksum_inventory_file_count":
            int(
                len(expected_checksums)
            ),

        "batch_checksums_match":
            True,

        "production_audit":
            str(library_audit_path),

        "production_audit_sha256":
            sha256_file(
                library_audit_path
            ),

        "unique_library_case_ids":
            int(
                written[
                    "library_case_id"
                ].nunique()
            ),

        "unique_pair_keys":
            int(
                written[
                    "pair_key"
                ].nunique()
            ),

        "unique_donor_slices":
            int(
                written[
                    "donor_h5_file"
                ].nunique()
            ),

        "unique_external_bases":
            int(
                written[
                    [
                        "external_subject_numeric_id",
                        "external_slice_index",
                    ]
                ]
                .drop_duplicates()
                .shape[0]
            ),

        "batch_counts":
            {
                str(key):
                    int(value)
                for key, value in (
                    written[
                        "batch_id"
                    ]
                    .value_counts()
                    .sort_index()
                    .items()
                )
            },
    }

    with acceptance_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            acceptance,
            file,
            indent=2,
            sort_keys=True,
        )

        file.write("\n")

    with master_hash_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        file.write(
            f"{new_master_sha}  "
            "br_lora_library_manifest.csv\n"
        )

    print()
    print("=" * 78)
    print("BATCH ACCEPTANCE: PASS")
    print("=" * 78)
    print()
    print("Accepted batch           :", batch_id)
    print("Accepted cases           :", EXPECTED_BATCH_SIZE)
    print("Master rows before       :", len(master))
    print("Master rows after        :", len(written))
    print(
        "Library IDs              :",
        written["library_case_id"].iloc[0],
        "to",
        written["library_case_id"].iloc[-1],
    )
    print(
        "Unique pair keys         :",
        written["pair_key"].nunique(),
    )
    print(
        "Unique donor slices      :",
        written["donor_h5_file"].nunique(),
    )
    print(
        "Unique external bases    :",
        written[
            [
                "external_subject_numeric_id",
                "external_slice_index",
            ]
        ]
        .drop_duplicates()
        .shape[0],
    )
    print(
        "Master SHA-256           :",
        new_master_sha,
    )
    print(
        "Acceptance audit         :",
        acceptance_path,
    )
    print(
        "Pre-promotion snapshot   :",
        snapshot_path,
    )
    print()
    print("Staging batch was preserved after acceptance.")


if __name__ == "__main__":
    try:
        main()

    except (
        BatchAcceptanceError,
        FileNotFoundError,
        NotADirectoryError,
        OSError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        print(
            "\nBR-LoRA BATCH ACCEPTANCE FAILED",
            file=sys.stderr,
        )

        print(
            exc,
            file=sys.stderr,
        )

        sys.exit(1)
