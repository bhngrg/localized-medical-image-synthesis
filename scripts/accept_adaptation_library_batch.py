#!/usr/bin/env python3
"""
Accept one completed deterministic-PEFT production batch into its library.

Acceptance requires:

1. the canonical frozen BR-LoRA design batch,
2. the deterministic execution manifest,
3. a successful deterministic production audit,
4. an exact match to the production SHA-256 inventory,
5. exactly 250 complete canonical library_case_id directories,
6. consistency between generated artifacts and production provenance, and
7. a destination that does not already contain the accepted batch.

The frozen BR-LoRA 10,000-case design remains the canonical library manifest.
No separate deterministic master manifest is constructed.

All accepted deterministic case directories use library_case_id. The legacy
source_case_id field in batch_0001 is retained only as frozen-design
provenance.

The script never deletes staging data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]

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
from src.inference.adaptation import (
    SUPPORTED_ADAPTATION_METHODS,
)


DEFAULT_DESIGN_BATCH_DIR = (
    PROJECT_ROOT
    / "downstream_evaluation"
    / "manifests"
    / "br_lora_library_design_10000"
    / "batches"
)

EXPECTED_BATCH_SIZE = 250
FIRST_LIBRARY_BATCH = 1
LAST_LIBRARY_BATCH = 40
EXPECTED_EVALUATION_SEED = 42
EXPECTED_BATCH_FILE_COUNT = 751

EVALUATION_NAME = (
    "deterministic_adaptation_external_evaluation"
)

EXPECTED_CASE_ARTIFACTS = (
    "prediction.pt",
    "synthesis_payload.pt",
    "metadata.json",
)

REQUIRED_PAYLOAD_KEYS = {
    "evaluation_name",
    "case_id",
    "adaptation_method",
    "checkpoint",
    "evaluation_seed",
    "case_seed",
    "reference_br_lora_payload",
    "prediction",
    "base_image",
    "transferred_mask",
    "known",
    "donor_image",
    "donor_patch",
    "donor_condition",
    "timestep",
    "diffusion_noise",
    "x_t",
    "reference_x_t_max_abs_difference",
}


class AdaptationBatchAcceptanceError(
    RuntimeError
):
    """Raised when a deterministic PEFT batch cannot be accepted."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Verify and promote one completed deterministic PEFT "
            "batch into its method-specific library."
        )
    )

    parser.add_argument(
        "--batch",
        required=True,
        help="Batch identifier, for example batch_0003.",
    )

    parser.add_argument(
        "--method",
        required=True,
        choices=SUPPORTED_ADAPTATION_METHODS,
    )

    parser.add_argument(
        "--staging-root",
        type=Path,
        default=None,
        help=(
            "Method-specific deterministic production staging root. "
            "Overrides the method-specific *_staging_root value "
            "in --folders-file."
        ),
    )

    parser.add_argument(
        "--library-root",
        type=Path,
        default=None,
        help=(
            "Accepted deterministic library root. Overrides the "
            "method-specific *_library_root value in --folders-file."
        ),
    )

    parser.add_argument(
        "--design-batch-dir",
        type=Path,
        default=DEFAULT_DESIGN_BATCH_DIR,
    )

    parser.add_argument(
        "--folders-file",
        type=Path,
        default=Path(
            "data/folders.yaml"
        ),
        help="Machine-specific folders configuration file.",
    )

    return parser.parse_args()


def parse_batch_id(
    value: str,
) -> tuple[str, int]:
    text = value.strip()

    if not text.startswith(
        "batch_"
    ):
        raise AdaptationBatchAcceptanceError(
            "--batch must use the form batch_0003."
        )

    suffix = text[
        len(
            "batch_"
        ):
    ]

    if (
        len(
            suffix
        )
        != 4
        or not suffix.isdigit()
    ):
        raise AdaptationBatchAcceptanceError(
            "--batch must contain four numeric digits."
        )

    number = int(
        suffix
    )

    if not (
        FIRST_LIBRARY_BATCH
        <= number
        <= LAST_LIBRARY_BATCH
    ):
        raise AdaptationBatchAcceptanceError(
            f"Batch number must lie between "
            f"{FIRST_LIBRARY_BATCH:04d} and "
            f"{LAST_LIBRARY_BATCH:04d}."
        )

    return (
        text,
        number,
    )


def resolve_acceptance_paths(
    args: argparse.Namespace,
) -> tuple[
    Path,
    Path,
]:
    config = load_folders_config(
        args.folders_file
    )

    staging_root = resolve_path(
        key=f"{args.method}_staging_root",
        cli_value=args.staging_root,
        config=config,
        selector=None,
    )

    library_root = resolve_path(
        key=f"{args.method}_library_root",
        cli_value=args.library_root,
        config=config,
        selector=None,
    )

    save_folders_config(
        args.folders_file,
        config,
    )

    return (
        Path(
            staging_root
        )
        .expanduser()
        .resolve(),
        Path(
            library_root
        )
        .expanduser()
        .resolve(),
    )


def require_file(
    path: Path,
    *,
    name: str,
) -> Path:
    resolved = (
        path
        .expanduser()
        .resolve()
    )

    if not resolved.is_file():
        raise FileNotFoundError(
            f"{name} does not exist:\n{resolved}"
        )

    return resolved


def require_directory(
    path: Path,
    *,
    name: str,
) -> Path:
    resolved = (
        path
        .expanduser()
        .resolve()
    )

    if not resolved.is_dir():
        raise NotADirectoryError(
            f"{name} does not exist:\n{resolved}"
        )

    return resolved


def sha256_file(
    path: Path,
) -> str:
    digest = hashlib.sha256()

    with path.open(
        "rb"
    ) as file:
        for chunk in iter(
            lambda: file.read(
                1024 * 1024
            ),
            b"",
        ):
            digest.update(
                chunk
            )

    return digest.hexdigest()


def load_checksum_inventory(
    path: Path,
) -> dict[str, str]:
    values: dict[
        str,
        str,
    ] = {}

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line_number, raw_line in enumerate(
            file,
            start=1,
        ):
            line = raw_line.rstrip(
                "\n"
            )

            if not line:
                continue

            parts = line.split(
                "  ",
                1,
            )

            if len(
                parts
            ) != 2:
                raise AdaptationBatchAcceptanceError(
                    "Malformed checksum inventory line "
                    f"{line_number} in {path}."
                )

            digest, relative = parts

            if (
                len(
                    digest
                )
                != 64
                or any(
                    character
                    not in "0123456789abcdefABCDEF"
                    for character in digest
                )
            ):
                raise AdaptationBatchAcceptanceError(
                    "Invalid SHA-256 digest on line "
                    f"{line_number} in {path}."
                )

            if relative in values:
                raise AdaptationBatchAcceptanceError(
                    f"Duplicate checksum path in {path}: "
                    f"{relative}"
                )

            values[
                relative
            ] = digest.lower()

    return values


def compute_batch_checksums(
    batch_root: Path,
) -> dict[str, str]:
    files = sorted(
        path
        for path in batch_root.rglob(
            "*"
        )
        if path.is_file()
    )

    if not files:
        raise AdaptationBatchAcceptanceError(
            "Cannot checksum an empty batch directory."
        )

    return {
        (
            "./"
            + path.relative_to(
                batch_root
            ).as_posix()
        ):
            sha256_file(
                path
            )
        for path in files
    }


def validate_design(
    *,
    design: pd.DataFrame,
    batch_id: str,
    batch_number: int,
) -> list[str]:
    required = {
        "library_index",
        "library_case_id",
        "batch_id",
        "source_case_id",
        "external_subject_numeric_id",
        "external_slice_index",
        "external_modality",
        "donor_h5_file",
    }

    missing = (
        required
        - set(
            design.columns
        )
    )

    if missing:
        raise AdaptationBatchAcceptanceError(
            "Frozen batch manifest is missing required columns:\n"
            + "\n".join(
                sorted(
                    missing
                )
            )
        )

    if len(
        design
    ) != EXPECTED_BATCH_SIZE:
        raise AdaptationBatchAcceptanceError(
            f"{batch_id} must contain exactly "
            f"{EXPECTED_BATCH_SIZE} rows."
        )

    if set(
        design[
            "batch_id"
        ].astype(
            str
        )
    ) != {
        batch_id
    }:
        raise AdaptationBatchAcceptanceError(
            "Frozen design contains an unexpected batch_id."
        )

    expected_first = (
        (
            batch_number
            - 1
        )
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
        design[
            "library_index"
        ]
        .astype(
            int
        )
        .tolist()
    )

    if (
        observed_indices
        != expected_indices
    ):
        raise AdaptationBatchAcceptanceError(
            "Frozen design library_index sequence is incorrect."
        )

    expected_case_ids = [
        f"synthetic_{index:06d}"
        for index in expected_indices
    ]

    observed_case_ids = (
        design[
            "library_case_id"
        ]
        .astype(
            str
        )
        .tolist()
    )

    if (
        observed_case_ids
        != expected_case_ids
    ):
        raise AdaptationBatchAcceptanceError(
            "Frozen design library_case_id sequence is incorrect."
        )

    if batch_number == 1:
        if not design[
            "source_case_id"
        ].notna().all():
            raise AdaptationBatchAcceptanceError(
                "batch_0001 must contain source_case_id "
                "for all 250 frozen-design rows."
            )

    else:
        if design[
            "source_case_id"
        ].notna().any():
            raise AdaptationBatchAcceptanceError(
                f"{batch_id} unexpectedly contains source_case_id values."
            )

    return expected_case_ids


def validate_execution_manifest(
    *,
    execution: pd.DataFrame,
    design: pd.DataFrame,
    expected_case_ids: list[str],
) -> None:
    expected_columns = [
        "case_id",
        "external_subject_numeric_id",
        "external_slice_index",
        "external_modality",
        "donor_h5_path",
    ]

    if list(
        execution.columns
    ) != expected_columns:
        raise AdaptationBatchAcceptanceError(
            "Execution manifest does not have the exact "
            "five-column production contract."
        )

    if len(
        execution
    ) != EXPECTED_BATCH_SIZE:
        raise AdaptationBatchAcceptanceError(
            "Execution manifest must contain exactly 250 rows."
        )

    observed_case_ids = (
        execution[
            "case_id"
        ]
        .astype(
            str
        )
        .tolist()
    )

    if (
        observed_case_ids
        != expected_case_ids
    ):
        raise AdaptationBatchAcceptanceError(
            "Execution case_id sequence does not exactly match "
            "the canonical library_case_id sequence."
        )

    for column in (
        "external_subject_numeric_id",
        "external_slice_index",
    ):
        observed = (
            execution[
                column
            ]
            .astype(
                int
            )
            .tolist()
        )

        expected = (
            design[
                column
            ]
            .astype(
                int
            )
            .tolist()
        )

        if (
            observed
            != expected
        ):
            raise AdaptationBatchAcceptanceError(
                f"Execution manifest {column} does not match "
                "the frozen design."
            )

    observed_modalities = (
        execution[
            "external_modality"
        ]
        .astype(
            str
        )
        .str.lower()
        .tolist()
    )

    expected_modalities = (
        design[
            "external_modality"
        ]
        .astype(
            str
        )
        .str.lower()
        .tolist()
    )

    if (
        observed_modalities
        != expected_modalities
    ):
        raise AdaptationBatchAcceptanceError(
            "Execution manifest external_modality does not "
            "match the frozen design."
        )

    observed_donor_names = [
        Path(
            value
        ).name
        for value in execution[
            "donor_h5_path"
        ].astype(
            str
        )
    ]

    expected_donor_names = (
        design[
            "donor_h5_file"
        ]
        .astype(
            str
        )
        .tolist()
    )

    if (
        observed_donor_names
        != expected_donor_names
    ):
        raise AdaptationBatchAcceptanceError(
            "Execution manifest donor H5 sequence does not "
            "match the frozen design."
        )


def validate_production_audit(
    *,
    audit: dict[str, object],
    method: str,
    execution_path: Path,
    source_batch_root: Path,
) -> tuple[
    Path,
    Path,
]:
    expected = {
        "status":
            "pass",
        "adaptation_method":
            method,
        "case_count":
            EXPECTED_BATCH_SIZE,
        "evaluation_seed":
            EXPECTED_EVALUATION_SEED,
        "problems":
            0,
    }

    for key, value in expected.items():
        if audit.get(
            key
        ) != value:
            raise AdaptationBatchAcceptanceError(
                f"Production audit {key} mismatch.\n"
                f"Expected: {value!r}\n"
                f"Observed: {audit.get(key)!r}"
            )

    observed_execution_sha = sha256_file(
        execution_path
    )

    if audit.get(
        "execution_manifest_sha256"
    ) != observed_execution_sha:
        raise AdaptationBatchAcceptanceError(
            "Production audit execution-manifest SHA-256 mismatch."
        )

    checkpoint_text = audit.get(
        "checkpoint"
    )

    checkpoint_sha = audit.get(
        "checkpoint_sha256"
    )

    if (
        not isinstance(
            checkpoint_text,
            str,
        )
        or not checkpoint_text
        or not isinstance(
            checkpoint_sha,
            str,
        )
        or not checkpoint_sha
    ):
        raise AdaptationBatchAcceptanceError(
            "Production audit does not contain valid checkpoint provenance."
        )

    checkpoint = require_file(
        Path(
            checkpoint_text
        ),
        name="Production checkpoint",
    )

    if sha256_file(
        checkpoint
    ) != checkpoint_sha:
        raise AdaptationBatchAcceptanceError(
            "Current production checkpoint does not match "
            "the SHA-256 recorded by the production audit."
        )

    summary_text = audit.get(
        "evaluation_summary"
    )

    summary_sha = audit.get(
        "evaluation_summary_sha256"
    )

    if (
        not isinstance(
            summary_text,
            str,
        )
        or not summary_text
        or not isinstance(
            summary_sha,
            str,
        )
        or not summary_sha
    ):
        raise AdaptationBatchAcceptanceError(
            "Production audit does not contain valid "
            "evaluation-summary provenance."
        )

    summary_path = require_file(
        Path(
            summary_text
        ),
        name="Production evaluation summary",
    )

    expected_summary_path = (
        source_batch_root
        / "evaluation_summary.json"
    ).resolve()

    if (
        summary_path
        != expected_summary_path
    ):
        raise AdaptationBatchAcceptanceError(
            "Production audit evaluation_summary does not point "
            "to the staging batch evaluation_summary.json."
        )

    if sha256_file(
        summary_path
    ) != summary_sha:
        raise AdaptationBatchAcceptanceError(
            "Production evaluation summary does not match "
            "the SHA-256 recorded by the production audit."
        )

    return (
        checkpoint,
        summary_path,
    )


def validate_summary(
    *,
    summary: dict[str, object],
    method: str,
    execution_path: Path,
    checkpoint: Path,
) -> Path:
    expected = {
        "evaluation_name":
            EVALUATION_NAME,
        "adaptation_method":
            method,
        "checkpoint":
            str(
                checkpoint
            ),
        "evaluation_manifest":
            str(
                execution_path
            ),
        "evaluation_seed":
            EXPECTED_EVALUATION_SEED,
        "cases":
            EXPECTED_BATCH_SIZE,
        "completed_cases":
            EXPECTED_BATCH_SIZE,
        "fixed_reference_diffusion_noise":
            True,
    }

    for key, value in expected.items():
        if summary.get(
            key
        ) != value:
            raise AdaptationBatchAcceptanceError(
                f"Evaluation summary {key} mismatch.\n"
                f"Expected: {value!r}\n"
                f"Observed: {summary.get(key)!r}"
            )

    reference_root = summary.get(
        "reference_br_lora_batch_root"
    )

    if (
        not isinstance(
            reference_root,
            str,
        )
        or not reference_root
    ):
        raise AdaptationBatchAcceptanceError(
            "Evaluation summary does not contain a valid "
            "reference_br_lora_batch_root."
        )

    return require_directory(
        Path(
            reference_root
        ),
        name="Accepted BR-LoRA reference batch",
    )


def validate_case_artifacts(
    *,
    source_batch_root: Path,
    design: pd.DataFrame,
    expected_case_ids: list[str],
    method: str,
    checkpoint: Path,
    execution_path: Path,
    reference_batch_root: Path,
) -> None:
    actual_case_dirs = sorted(
        path.name
        for path in source_batch_root.iterdir()
        if path.is_dir()
    )

    if (
        actual_case_dirs
        != sorted(
            expected_case_ids
        )
    ):
        raise AdaptationBatchAcceptanceError(
            "Staging case-directory set does not exactly match "
            "the canonical library_case_id set."
        )

    design_by_case = (
        design
        .set_index(
            "library_case_id",
            drop=False,
        )
    )

    for case_id in expected_case_ids:
        case_dir = (
            source_batch_root
            / case_id
        )

        for artifact in EXPECTED_CASE_ARTIFACTS:
            artifact_path = (
                case_dir
                / artifact
            )

            if not artifact_path.is_file():
                raise AdaptationBatchAcceptanceError(
                    f"{case_id}: missing {artifact}."
                )

        prediction_path = (
            case_dir
            / "prediction.pt"
        )

        prediction = torch.load(
            prediction_path,
            map_location="cpu",
            weights_only=False,
        )

        if not isinstance(
            prediction,
            torch.Tensor,
        ):
            raise AdaptationBatchAcceptanceError(
                f"{case_id}: prediction.pt is not a tensor."
            )

        if tuple(
            prediction.shape
        ) != (
            1,
            240,
            240,
        ):
            raise AdaptationBatchAcceptanceError(
                f"{case_id}: prediction.pt has unexpected shape "
                f"{tuple(prediction.shape)}."
            )

        if not torch.isfinite(
            prediction
        ).all():
            raise AdaptationBatchAcceptanceError(
                f"{case_id}: prediction.pt contains non-finite values."
            )

        payload_path = (
            case_dir
            / "synthesis_payload.pt"
        )

        payload = torch.load(
            payload_path,
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )

        if not isinstance(
            payload,
            dict,
        ):
            raise AdaptationBatchAcceptanceError(
                f"{case_id}: synthesis_payload.pt is not a dictionary."
            )

        missing_payload = (
            REQUIRED_PAYLOAD_KEYS
            - set(
                payload
            )
        )

        if missing_payload:
            raise AdaptationBatchAcceptanceError(
                f"{case_id}: payload missing keys "
                f"{sorted(missing_payload)}."
            )

        expected_payload = {
            "evaluation_name":
                EVALUATION_NAME,
            "case_id":
                case_id,
            "adaptation_method":
                method,
            "checkpoint":
                str(
                    checkpoint
                ),
            "evaluation_seed":
                EXPECTED_EVALUATION_SEED,
        }

        for key, value in expected_payload.items():
            if payload.get(
                key
            ) != value:
                raise AdaptationBatchAcceptanceError(
                    f"{case_id}: payload {key} mismatch."
                )

        payload_prediction = payload.get(
            "prediction"
        )

        if not isinstance(
            payload_prediction,
            torch.Tensor,
        ):
            raise AdaptationBatchAcceptanceError(
                f"{case_id}: payload prediction is not a tensor."
            )

        if not torch.equal(
            prediction,
            payload_prediction,
        ):
            raise AdaptationBatchAcceptanceError(
                f"{case_id}: prediction.pt does not exactly match "
                "payload prediction."
            )

        metadata_path = (
            case_dir
            / "metadata.json"
        )

        try:
            metadata = json.loads(
                metadata_path.read_text(
                    encoding="utf-8"
                )
            )
        except Exception as exc:
            raise AdaptationBatchAcceptanceError(
                f"{case_id}: unreadable metadata.json."
            ) from exc

        row = design_by_case.loc[
            case_id
        ]

        expected_metadata = {
            "evaluation_name":
                EVALUATION_NAME,
            "case_id":
                case_id,
            "adaptation_method":
                method,
            "checkpoint":
                str(
                    checkpoint
                ),
            "evaluation_manifest":
                str(
                    execution_path
                ),
            "reference_br_lora_batch_root":
                str(
                    reference_batch_root
                ),
            "evaluation_seed":
                EXPECTED_EVALUATION_SEED,
            "external_subject_numeric_id":
                int(
                    row[
                        "external_subject_numeric_id"
                    ]
                ),
            "external_slice_index":
                int(
                    row[
                        "external_slice_index"
                    ]
                ),
            "external_modality":
                str(
                    row[
                        "external_modality"
                    ]
                ).lower(),
            "fixed_reference_diffusion_noise":
                True,
            "prediction_artifact":
                "prediction.pt",
            "synthesis_payload_artifact":
                "synthesis_payload.pt",
        }

        for key, value in expected_metadata.items():
            if metadata.get(
                key
            ) != value:
                raise AdaptationBatchAcceptanceError(
                    f"{case_id}: metadata {key} mismatch.\n"
                    f"Expected: {value!r}\n"
                    f"Observed: {metadata.get(key)!r}"
                )

        expected_reference_payload = (
            reference_batch_root
            / case_id
            / "posterior_samples.pt"
        )

        if not expected_reference_payload.is_file():
            raise AdaptationBatchAcceptanceError(
                f"{case_id}: accepted BR-LoRA reference payload "
                "does not exist."
            )

        expected_reference_text = str(
            expected_reference_payload
        )

        if payload.get(
            "reference_br_lora_payload"
        ) != expected_reference_text:
            raise AdaptationBatchAcceptanceError(
                f"{case_id}: payload BR-LoRA reference mismatch."
            )

        if metadata.get(
            "reference_br_lora_payload"
        ) != expected_reference_text:
            raise AdaptationBatchAcceptanceError(
                f"{case_id}: metadata BR-LoRA reference mismatch."
            )


def main() -> None:
    args = parse_args()

    batch_id, batch_number = parse_batch_id(
        args.batch
    )

    (
        staging_root,
        library_root,
    ) = resolve_acceptance_paths(
        args
    )

    staging_root = require_directory(
        staging_root,
        name="Deterministic PEFT staging root",
    )

    design_batch_dir = require_directory(
        args.design_batch_dir,
        name="Frozen design batch directory",
    )

    source_batch_root = require_directory(
        staging_root
        / batch_id,
        name="Completed deterministic staging batch",
    )

    design_path = require_file(
        design_batch_dir
        / f"{batch_id}_manifest.csv",
        name="Frozen batch manifest",
    )

    execution_path = require_file(
        staging_root
        / (
            f"{args.method}_{batch_id}_"
            "external_evaluation_manifest.csv"
        ),
        name="Deterministic execution manifest",
    )

    production_audit_path = require_file(
        staging_root
        / f"{args.method}_{batch_id}_production_audit.json",
        name="Deterministic production audit",
    )

    checksum_path = require_file(
        staging_root
        / f"{args.method}_{batch_id}_sha256.txt",
        name="Deterministic production checksum inventory",
    )

    print()
    print("=" * 78)
    print("DETERMINISTIC PEFT BATCH ACCEPTANCE")
    print("=" * 78)
    print()
    print("Method                   :", args.method)
    print("Batch                    :", batch_id)
    print("Staging batch            :", source_batch_root)
    print("Library root             :", library_root)

    design = pd.read_csv(
        design_path
    )

    expected_case_ids = validate_design(
        design=design,
        batch_id=batch_id,
        batch_number=batch_number,
    )

    execution = pd.read_csv(
        execution_path
    )

    validate_execution_manifest(
        execution=execution,
        design=design,
        expected_case_ids=expected_case_ids,
    )

    expected_checksums = load_checksum_inventory(
        checksum_path
    )

    observed_checksums = compute_batch_checksums(
        source_batch_root
    )

    if (
        expected_checksums
        != observed_checksums
    ):
        raise AdaptationBatchAcceptanceError(
            "Current staging batch does not match the "
            "production checksum inventory."
        )

    if len(
        expected_checksums
    ) != EXPECTED_BATCH_FILE_COUNT:
        raise AdaptationBatchAcceptanceError(
            f"Expected exactly {EXPECTED_BATCH_FILE_COUNT} batch files; "
            f"observed {len(expected_checksums)}."
        )

    try:
        production_audit = json.loads(
            production_audit_path.read_text(
                encoding="utf-8"
            )
        )
    except Exception as exc:
        raise AdaptationBatchAcceptanceError(
            "Could not read deterministic production audit."
        ) from exc

    if not isinstance(
        production_audit,
        dict,
    ):
        raise AdaptationBatchAcceptanceError(
            "Deterministic production audit must contain a JSON object."
        )

    checkpoint, summary_path = validate_production_audit(
        audit=production_audit,
        method=args.method,
        execution_path=execution_path,
        source_batch_root=source_batch_root,
    )

    try:
        summary = json.loads(
            summary_path.read_text(
                encoding="utf-8"
            )
        )
    except Exception as exc:
        raise AdaptationBatchAcceptanceError(
            "Could not read deterministic evaluation summary."
        ) from exc

    if not isinstance(
        summary,
        dict,
    ):
        raise AdaptationBatchAcceptanceError(
            "Deterministic evaluation summary must contain a JSON object."
        )

    reference_batch_root = validate_summary(
        summary=summary,
        method=args.method,
        execution_path=execution_path,
        checkpoint=checkpoint,
    )

    expected_reference_name = batch_id

    if (
        reference_batch_root.name
        != expected_reference_name
    ):
        raise AdaptationBatchAcceptanceError(
            "Evaluation summary points to the wrong BR-LoRA "
            "reference batch.\n"
            f"Expected batch: {batch_id}\n"
            f"Observed: {reference_batch_root}"
        )

    validate_case_artifacts(
        source_batch_root=source_batch_root,
        design=design,
        expected_case_ids=expected_case_ids,
        method=args.method,
        checkpoint=checkpoint,
        execution_path=execution_path,
        reference_batch_root=reference_batch_root,
    )

    provenance_root = (
        library_root
        / "_provenance"
    )

    accepted_batch_root = (
        library_root
        / batch_id
    )

    accepted_design_path = (
        provenance_root
        / f"{batch_id}_manifest.csv"
    )

    accepted_execution_path = (
        provenance_root
        / execution_path.name
    )

    accepted_production_audit_path = (
        provenance_root
        / production_audit_path.name
    )

    accepted_checksum_path = (
        provenance_root
        / checksum_path.name
    )

    acceptance_path = (
        provenance_root
        / f"{args.method}_{batch_id}_acceptance.json"
    )

    destinations = (
        accepted_batch_root,
        accepted_design_path,
        accepted_execution_path,
        accepted_production_audit_path,
        accepted_checksum_path,
        acceptance_path,
    )

    existing = [
        path
        for path in destinations
        if path.exists()
    ]

    if existing:
        raise AdaptationBatchAcceptanceError(
            "One or more deterministic-library destinations "
            "already exist:\n"
            + "\n".join(
                str(
                    path
                )
                for path in existing
            )
        )

    library_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    provenance_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    shutil.copytree(
        source_batch_root,
        accepted_batch_root,
    )

    if (
        compute_batch_checksums(
            accepted_batch_root
        )
        != expected_checksums
    ):
        raise AdaptationBatchAcceptanceError(
            "Copied deterministic library batch does not match "
            "the production checksum inventory."
        )

    for source, destination in (
        (
            design_path,
            accepted_design_path,
        ),
        (
            execution_path,
            accepted_execution_path,
        ),
        (
            production_audit_path,
            accepted_production_audit_path,
        ),
        (
            checksum_path,
            accepted_checksum_path,
        ),
    ):
        shutil.copy2(
            source,
            destination,
        )

        if sha256_file(
            source
        ) != sha256_file(
            destination
        ):
            raise AdaptationBatchAcceptanceError(
                "Copied deterministic provenance artifact "
                "hash mismatch:\n"
                f"{destination}"
            )

    acceptance = {
        "status":
            "accepted",
        "adaptation_method":
            args.method,
        "batch_id":
            batch_id,
        "accepted_at_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),
        "accepted_case_count":
            EXPECTED_BATCH_SIZE,
        "first_library_index":
            int(
                design[
                    "library_index"
                ].iloc[
                    0
                ]
            ),
        "last_library_index":
            int(
                design[
                    "library_index"
                ].iloc[
                    -1
                ]
            ),
        "first_library_case_id":
            expected_case_ids[
                0
            ],
        "last_library_case_id":
            expected_case_ids[
                -1
            ],
        "canonical_case_directory_identifier":
            "library_case_id",
        "source_case_id_role":
            "frozen_design_provenance_only",
        "accepted_batch_root":
            str(
                accepted_batch_root
            ),
        "frozen_batch_manifest":
            str(
                accepted_design_path
            ),
        "frozen_batch_manifest_sha256":
            sha256_file(
                accepted_design_path
            ),
        "execution_manifest":
            str(
                accepted_execution_path
            ),
        "execution_manifest_sha256":
            sha256_file(
                accepted_execution_path
            ),
        "production_audit":
            str(
                accepted_production_audit_path
            ),
        "production_audit_sha256":
            sha256_file(
                accepted_production_audit_path
            ),
        "checksum_inventory":
            str(
                accepted_checksum_path
            ),
        "checksum_inventory_sha256":
            sha256_file(
                accepted_checksum_path
            ),
        "checksum_inventory_file_count":
            int(
                len(
                    expected_checksums
                )
            ),
        "batch_checksums_match":
            True,
        "checkpoint":
            str(
                checkpoint
            ),
        "checkpoint_sha256":
            sha256_file(
                checkpoint
            ),
        "reference_br_lora_batch_root":
            str(
                reference_batch_root
            ),
        "evaluation_seed":
            EXPECTED_EVALUATION_SEED,
        "fixed_reference_diffusion_noise":
            True,
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
        file.write(
            "\n"
        )

    print()
    print("=" * 78)
    print("DETERMINISTIC PEFT BATCH ACCEPTANCE: PASS")
    print("=" * 78)
    print()
    print("Method                   :", args.method)
    print("Accepted batch           :", batch_id)
    print("Accepted cases           :", EXPECTED_BATCH_SIZE)
    print(
        "Library IDs              :",
        expected_case_ids[
            0
        ],
        "to",
        expected_case_ids[
            -1
        ],
    )
    print("Files checksummed        :", len(expected_checksums))
    print("Accepted batch root      :", accepted_batch_root)
    print("Acceptance audit         :", acceptance_path)
    print()
    print(
        "Staging data were preserved and were not deleted."
    )


if __name__ == "__main__":
    try:
        main()

    except (
        AdaptationBatchAcceptanceError,
        FileNotFoundError,
        NotADirectoryError,
        OSError,
        ValueError,
    ) as exc:
        print(
            file=sys.stderr,
        )
        print(
            "=" * 78,
            file=sys.stderr,
        )
        print(
            "DETERMINISTIC PEFT BATCH ACCEPTANCE: FAIL",
            file=sys.stderr,
        )
        print(
            "=" * 78,
            file=sys.stderr,
        )
        print(
            str(
                exc
            ),
            file=sys.stderr,
        )
        raise SystemExit(
            1
        )
