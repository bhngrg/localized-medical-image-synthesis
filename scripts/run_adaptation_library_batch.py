#!/usr/bin/env python3

"""
Run one frozen deterministic-PEFT synthetic-library production batch.

The same frozen 10,000-case conditioning design used by the accepted BR-LoRA
library is reused unchanged. For each deterministic PEFT method, batches
0001-0040 contain exactly 250 cases.

Batch 0001 preserves the accepted legacy source_case_id directory names.
Batches 0002-0040 use library_case_id.

This wrapper:

1. validates the frozen design batch,
2. validates that the checkpoint method matches --method,
3. constructs the fixed five-column external execution manifest,
4. validates donor H5 references,
5. optionally stops after preparation,
6. invokes scripts/evaluate_adaptation_external.py,
7. audits every generated deterministic case,
8. validates evaluation_summary.json, and
9. writes a SHA-256 inventory of the completed batch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(
            PROJECT_ROOT
        ),
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

DEFAULT_EVALUATION_SEED = 42

EXPECTED_BATCH_SIZE = 250
FIRST_LIBRARY_BATCH = 1
LAST_LIBRARY_BATCH = 40

EVALUATION_NAME = (
    "deterministic_adaptation_external_evaluation"
)

REQUIRED_CASE_ARTIFACTS = (
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


class AdaptationBatchProductionError(
    RuntimeError
):
    """Raised when deterministic library production fails validation."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare, run, and audit one frozen 250-case "
            "deterministic PEFT library batch."
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
        "--checkpoint",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--design-batch-dir",
        type=Path,
        default=DEFAULT_DESIGN_BATCH_DIR,
    )

    parser.add_argument(
        "--h5-root",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--validation-dataset",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--reference-br-lora-library-root",
        type=Path,
        default=None,
        help=(
            "Accepted BR-LoRA library batches root. "
            "Defaults to br_lora_library_root in folders YAML."
        ),
    )

    parser.add_argument(
        "--staging-root",
        type=Path,
        required=True,
        help=(
            "Method-specific deterministic library batches root, for "
            "example /scratch/.../regional_lora/batches."
        ),
    )

    parser.add_argument(
        "--folders-file",
        type=Path,
        default=Path(
            "data/folders.yaml"
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_EVALUATION_SEED,
    )

    parser.add_argument(
        "--device",
        choices=(
            "auto",
            "cpu",
            "mps",
            "cuda",
        ),
        default="cuda",
    )

    parser.add_argument(
        "--prepare-only",
        action="store_true",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
    )

    return parser.parse_args()


def resolve_existing_file(
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


def resolve_existing_directory(
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


def resolve_production_paths(
    args: argparse.Namespace,
) -> tuple[
    Path,
    Path,
    Path,
]:
    config = load_folders_config(
        args.folders_file
    )

    h5_root = resolve_path(
        key="h5_root",
        cli_value=args.h5_root,
        config=config,
        selector=None,
    )

    validation_dataset = resolve_path(
        key="yaml_validation_dataset_path",
        cli_value=args.validation_dataset,
        config=config,
        selector=None,
    )

    reference_root = resolve_path(
        key="br_lora_library_root",
        cli_value=args.reference_br_lora_library_root,
        config=config,
        selector=None,
    )

    save_folders_config(
        args.folders_file,
        config,
    )

    return (
        Path(
            h5_root
        ),
        Path(
            validation_dataset
        ),
        Path(
            reference_root
        ),
    )


def parse_batch_id(
    value: str,
) -> tuple[
    str,
    int,
]:
    text = value.strip()

    if not text.startswith(
        "batch_"
    ):
        raise AdaptationBatchProductionError(
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
        raise AdaptationBatchProductionError(
            "--batch must use four numeric digits."
        )

    number = int(
        suffix
    )

    if not (
        FIRST_LIBRARY_BATCH
        <= number
        <= LAST_LIBRARY_BATCH
    ):
        raise AdaptationBatchProductionError(
            "Batch number must lie between "
            f"{FIRST_LIBRARY_BATCH:04d} and "
            f"{LAST_LIBRARY_BATCH:04d}."
        )

    return (
        f"batch_{number:04d}",
        number,
    )


def sha256_file(
    path: Path,
) -> str:
    digest = hashlib.sha256()

    with path.open(
        "rb"
    ) as file:
        for block in iter(
            lambda: file.read(
                1024
                * 1024
            ),
            b"",
        ):
            digest.update(
                block
            )

    return digest.hexdigest()


def validate_checkpoint_method(
    *,
    checkpoint: Path,
    method: str,
) -> None:
    payload = torch.load(
        checkpoint,
        map_location="cpu",
        weights_only=False,
    )

    if not isinstance(
        payload,
        dict,
    ):
        raise AdaptationBatchProductionError(
            "Deterministic PEFT checkpoint must contain a dictionary."
        )

    observed = payload.get(
        "adaptation_method"
    )

    if observed != method:
        raise AdaptationBatchProductionError(
            "Checkpoint adaptation method does not match --method.\n"
            f"Requested:  {method!r}\n"
            f"Checkpoint: {observed!r}"
        )


def case_directory_name(
    row: pd.Series,
) -> str:
    source_case_id = row[
        "source_case_id"
    ]

    if pd.notna(
        source_case_id
    ):
        return str(
            source_case_id
        )

    return str(
        row[
            "library_case_id"
        ]
    )


def prepare_execution_manifest(
    *,
    batch_id: str,
    batch_number: int,
    design_manifest_path: Path,
    execution_manifest_path: Path,
    h5_root: Path,
) -> pd.DataFrame:
    table = pd.read_csv(
        design_manifest_path
    )

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
            table.columns
        )
    )

    if missing:
        raise AdaptationBatchProductionError(
            "Frozen batch manifest is missing required columns:\n"
            + "\n".join(
                sorted(
                    missing
                )
            )
        )

    if len(
        table
    ) != EXPECTED_BATCH_SIZE:
        raise AdaptationBatchProductionError(
            f"{batch_id} must contain exactly "
            f"{EXPECTED_BATCH_SIZE} rows; "
            f"observed {len(table)}."
        )

    if set(
        table[
            "batch_id"
        ].astype(
            str
        )
    ) != {
        batch_id
    }:
        raise AdaptationBatchProductionError(
            "Frozen batch manifest contains an unexpected batch_id."
        )

    expected_first_index = (
        (
            batch_number
            - 1
        )
        * EXPECTED_BATCH_SIZE
        + 1
    )

    expected_last_index = (
        batch_number
        * EXPECTED_BATCH_SIZE
    )

    observed_indices = (
        table[
            "library_index"
        ]
        .astype(
            int
        )
        .tolist()
    )

    expected_indices = list(
        range(
            expected_first_index,
            expected_last_index
            + 1,
        )
    )

    if (
        observed_indices
        != expected_indices
    ):
        raise AdaptationBatchProductionError(
            "Frozen batch library_index sequence is incorrect."
        )

    expected_library_ids = [
        f"synthetic_{index:06d}"
        for index in expected_indices
    ]

    if (
        table[
            "library_case_id"
        ]
        .astype(
            str
        )
        .tolist()
        != expected_library_ids
    ):
        raise AdaptationBatchProductionError(
            "Frozen batch library_case_id sequence is incorrect."
        )

    if batch_number == 1:
        if not table[
            "source_case_id"
        ].notna().all():
            raise AdaptationBatchProductionError(
                "batch_0001 must contain source_case_id for all 250 cases."
            )
    else:
        if table[
            "source_case_id"
        ].notna().any():
            raise AdaptationBatchProductionError(
                f"{batch_id} unexpectedly contains source_case_id values."
            )

    case_ids = [
        case_directory_name(
            row
        )
        for _,
        row in table.iterrows()
    ]

    if len(
        set(
            case_ids
        )
    ) != EXPECTED_BATCH_SIZE:
        raise AdaptationBatchProductionError(
            "Execution case identifiers are not unique."
        )

    execution = pd.DataFrame(
        {
            "case_id":
                case_ids,

            "external_subject_numeric_id":
                table[
                    "external_subject_numeric_id"
                ].astype(
                    int
                ),

            "external_slice_index":
                table[
                    "external_slice_index"
                ].astype(
                    int
                ),

            "external_modality":
                (
                    table[
                        "external_modality"
                    ]
                    .astype(
                        str
                    )
                    .str.lower()
                ),

            "donor_h5_path":
                table[
                    "donor_h5_file"
                ]
                .astype(
                    str
                )
                .map(
                    lambda name: str(
                        (
                            h5_root
                            / name
                        ).resolve()
                    )
                ),
        }
    )

    missing_donors = [
        value
        for value in execution[
            "donor_h5_path"
        ]
        if not Path(
            value
        ).is_file()
    ]

    if missing_donors:
        raise AdaptationBatchProductionError(
            "Frozen batch references inaccessible donor H5 files.\n"
            + "\n".join(
                missing_donors[
                    :10
                ]
            )
        )

    execution_manifest_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    execution.to_csv(
        execution_manifest_path,
        index=False,
    )

    return execution


def validate_reference_batch(
    *,
    execution: pd.DataFrame,
    reference_batch_root: Path,
) -> None:
    missing = []

    for case_id in execution[
        "case_id"
    ].astype(
        str
    ):
        path = (
            reference_batch_root
            / case_id
            / "posterior_samples.pt"
        )

        if not path.is_file():
            missing.append(
                str(
                    path
                )
            )

    if missing:
        raise AdaptationBatchProductionError(
            "Accepted BR-LoRA reference batch is incomplete.\n"
            + "\n".join(
                missing[
                    :10
                ]
            )
        )


def run_inference(
    *,
    checkpoint: Path,
    validation_dataset: Path,
    execution_manifest: Path,
    reference_batch_root: Path,
    seed: int,
    output_dir: Path,
    device: str,
    resume: bool,
) -> None:
    script = (
        PROJECT_ROOT
        / "scripts"
        / "evaluate_adaptation_external.py"
    )

    if not script.is_file():
        raise FileNotFoundError(
            f"Deterministic evaluator not found:\n{script}"
        )

    command = [
        sys.executable,
        str(
            script
        ),
        "--checkpoint",
        str(
            checkpoint
        ),
        "--validation-dataset",
        str(
            validation_dataset
        ),
        "--evaluation-manifest",
        str(
            execution_manifest
        ),
        "--reference-br-lora-batch-root",
        str(
            reference_batch_root
        ),
        "--seed",
        str(
            seed
        ),
        "--output-dir",
        str(
            output_dir
        ),
        "--device",
        device,
    ]

    if resume:
        command.append(
            "--resume"
        )

    print()
    print("=" * 78)
    print("LAUNCHING DETERMINISTIC PEFT INFERENCE ENGINE")
    print("=" * 78)
    print(
        " ".join(
            command
        )
    )
    print()

    subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        check=True,
    )


def audit_completed_batch(
    *,
    execution: pd.DataFrame,
    execution_manifest_path: Path,
    checkpoint: Path,
    method: str,
    validation_dataset: Path,
    reference_batch_root: Path,
    batch_root: Path,
    seed: int,
) -> dict[str, object]:
    problems: list[
        str
    ] = []

    expected_case_ids = (
        execution[
            "case_id"
        ]
        .astype(
            str
        )
        .tolist()
    )

    actual_case_dirs = sorted(
        path.name
        for path in batch_root.iterdir()
        if (
            path.is_dir()
        )
    )

    if (
        actual_case_dirs
        != sorted(
            expected_case_ids
        )
    ):
        problems.append(
            "Output case-directory set does not exactly match "
            "the execution manifest."
        )

    expected_checkpoint = str(
        checkpoint
    )

    expected_manifest = str(
        execution_manifest_path
    )

    for case_id in expected_case_ids:
        case_dir = (
            batch_root
            / case_id
        )

        if not case_dir.is_dir():
            problems.append(
                f"{case_id}: missing case directory"
            )
            continue

        for artifact in REQUIRED_CASE_ARTIFACTS:
            if not (
                case_dir
                / artifact
            ).is_file():
                problems.append(
                    f"{case_id}: missing {artifact}"
                )

        payload_path = (
            case_dir
            / "synthesis_payload.pt"
        )

        if payload_path.is_file():
            try:
                payload = torch.load(
                    payload_path,
                    map_location="cpu",
                    weights_only=False,
                    mmap=True,
                )
            except Exception as exc:
                problems.append(
                    f"{case_id}: unreadable synthesis_payload.pt: {exc}"
                )
                payload = None

            if isinstance(
                payload,
                dict,
            ):
                missing_keys = (
                    REQUIRED_PAYLOAD_KEYS
                    - set(
                        payload
                    )
                )

                if missing_keys:
                    problems.append(
                        f"{case_id}: payload missing keys "
                        f"{sorted(missing_keys)}"
                    )

                expected_payload = {
                    "evaluation_name":
                        EVALUATION_NAME,
                    "case_id":
                        case_id,
                    "adaptation_method":
                        method,
                    "checkpoint":
                        expected_checkpoint,
                    "evaluation_seed":
                        int(
                            seed
                        ),
                }

                for key, expected in expected_payload.items():
                    if payload.get(
                        key
                    ) != expected:
                        problems.append(
                            f"{case_id}: payload {key} mismatch"
                        )

                prediction = payload.get(
                    "prediction"
                )

                if not isinstance(
                    prediction,
                    torch.Tensor,
                ):
                    problems.append(
                        f"{case_id}: prediction is not a tensor"
                    )

                elif tuple(
                    prediction.shape
                ) != (
                    1,
                    240,
                    240,
                ):
                    problems.append(
                        f"{case_id}: prediction shape "
                        f"{tuple(prediction.shape)}"
                    )

                elif not torch.isfinite(
                    prediction
                ).all():
                    problems.append(
                        f"{case_id}: prediction contains non-finite values"
                    )

            elif payload is not None:
                problems.append(
                    f"{case_id}: synthesis_payload.pt is not a dict"
                )

        metadata_path = (
            case_dir
            / "metadata.json"
        )

        if metadata_path.is_file():
            try:
                metadata = json.loads(
                    metadata_path.read_text(
                        encoding="utf-8"
                    )
                )
            except Exception as exc:
                problems.append(
                    f"{case_id}: unreadable metadata.json: {exc}"
                )
                metadata = {}

            expected_metadata = {
                "evaluation_name":
                    EVALUATION_NAME,
                "case_id":
                    case_id,
                "adaptation_method":
                    method,
                "checkpoint":
                    expected_checkpoint,
                "evaluation_manifest":
                    expected_manifest,
                "reference_br_lora_batch_root":
                    str(
                        reference_batch_root
                    ),
                "evaluation_seed":
                    int(
                        seed
                    ),
            }

            for key, expected in expected_metadata.items():
                if metadata.get(
                    key
                ) != expected:
                    problems.append(
                        f"{case_id}: metadata {key} mismatch"
                    )

    summary_path = (
        batch_root
        / "evaluation_summary.json"
    )

    if not summary_path.is_file():
        problems.append(
            "Missing evaluation_summary.json."
        )
        summary = {}

    else:
        try:
            summary = json.loads(
                summary_path.read_text(
                    encoding="utf-8"
                )
            )
        except Exception as exc:
            problems.append(
                f"evaluation_summary.json unreadable: {exc}"
            )
            summary = {}

    expected_summary = {
        "evaluation_name":
            EVALUATION_NAME,
        "adaptation_method":
            method,
        "checkpoint":
            expected_checkpoint,
        "evaluation_manifest":
            expected_manifest,
        "validation_dataset":
            str(
                validation_dataset
            ),
        "reference_br_lora_batch_root":
            str(
                reference_batch_root
            ),
        "evaluation_seed":
            int(
                seed
            ),
        "cases":
            EXPECTED_BATCH_SIZE,
        "completed_cases":
            EXPECTED_BATCH_SIZE,
        "fixed_reference_diffusion_noise":
            True,
    }

    for key, expected in expected_summary.items():
        if summary.get(
            key
        ) != expected:
            problems.append(
                "evaluation_summary.json "
                f"{key} expected {expected!r}, "
                f"observed {summary.get(key)!r}"
            )

    if problems:
        print()
        print("=" * 78)
        print("PRODUCTION AUDIT: FAIL")
        print("=" * 78)

        for problem in problems[
            :30
        ]:
            print(
                " -",
                problem,
            )

        raise AdaptationBatchProductionError(
            f"Production audit found {len(problems)} problem(s)."
        )

    return {
        "status":
            "pass",
        "adaptation_method":
            method,
        "audited_at_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),
        "case_count":
            EXPECTED_BATCH_SIZE,
        "evaluation_seed":
            int(
                seed
            ),
        "execution_manifest":
            str(
                execution_manifest_path
            ),
        "execution_manifest_sha256":
            sha256_file(
                execution_manifest_path
            ),
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
        "evaluation_summary":
            str(
                summary_path
            ),
        "evaluation_summary_sha256":
            sha256_file(
                summary_path
            ),
        "problems":
            0,
    }


def write_batch_checksum_inventory(
    *,
    batch_root: Path,
    output_path: Path,
) -> int:
    files = sorted(
        path
        for path in batch_root.rglob(
            "*"
        )
        if path.is_file()
    )

    if not files:
        raise AdaptationBatchProductionError(
            "Cannot checksum an empty batch directory."
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        for path in files:
            relative = path.relative_to(
                batch_root
            )

            file.write(
                f"{sha256_file(path)}  ./{relative.as_posix()}\n"
            )

    return len(
        files
    )


def main() -> None:
    args = parse_args()

    batch_id, batch_number = parse_batch_id(
        args.batch
    )

    (
        h5_root,
        validation_dataset,
        reference_library_root,
    ) = resolve_production_paths(
        args
    )

    design_batch_dir = resolve_existing_directory(
        args.design_batch_dir,
        name="Frozen design batch directory",
    )

    h5_root = resolve_existing_directory(
        h5_root,
        name="BraTS H5 root",
    )

    validation_dataset = resolve_existing_file(
        validation_dataset,
        name="Validation dataset specification",
    )

    reference_library_root = resolve_existing_directory(
        reference_library_root,
        name="Accepted BR-LoRA library root",
    )

    checkpoint = resolve_existing_file(
        args.checkpoint,
        name="Deterministic PEFT checkpoint",
    )

    validate_checkpoint_method(
        checkpoint=checkpoint,
        method=args.method,
    )

    design_manifest_path = resolve_existing_file(
        design_batch_dir
        / f"{batch_id}_manifest.csv",
        name="Frozen batch manifest",
    )

    staging_root = (
        args.staging_root
        .expanduser()
        .resolve()
    )

    staging_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    execution_manifest_path = (
        staging_root
        / f"{args.method}_{batch_id}_external_evaluation_manifest.csv"
    )

    batch_root = (
        staging_root
        / batch_id
    )

    audit_path = (
        staging_root
        / f"{args.method}_{batch_id}_production_audit.json"
    )

    checksum_path = (
        staging_root
        / f"{args.method}_{batch_id}_sha256.txt"
    )

    reference_batch_root = (
        reference_library_root
        / "batches"
        / batch_id
    )

    reference_batch_root = resolve_existing_directory(
        reference_batch_root,
        name="Accepted BR-LoRA reference batch",
    )

    print()
    print("=" * 78)
    print("DETERMINISTIC PEFT LIBRARY BATCH PRODUCTION")
    print("=" * 78)
    print("Method                   :", args.method)
    print("Batch                    :", batch_id)
    print("Frozen design manifest   :", design_manifest_path)
    print("Execution manifest       :", execution_manifest_path)
    print("Checkpoint               :", checkpoint)
    print("Evaluation seed          :", args.seed)
    print("Device                   :", args.device)
    print("Reference BR-LoRA batch  :", reference_batch_root)
    print("Output                   :", batch_root)
    print("Prepare only             :", args.prepare_only)
    print("Resume                   :", args.resume)

    execution = prepare_execution_manifest(
        batch_id=batch_id,
        batch_number=batch_number,
        design_manifest_path=design_manifest_path,
        execution_manifest_path=execution_manifest_path,
        h5_root=h5_root,
    )

    validate_reference_batch(
        execution=execution,
        reference_batch_root=reference_batch_root,
    )

    print()
    print("===== EXECUTION MANIFEST: PASS =====")
    print("Cases                    :", len(execution))
    print(
        "Case IDs                 :",
        execution[
            "case_id"
        ].iloc[
            0
        ],
        "to",
        execution[
            "case_id"
        ].iloc[
            -1
        ],
    )
    print(
        "Unique donor slices      :",
        execution[
            "donor_h5_path"
        ].nunique(),
    )
    print(
        "Manifest SHA-256         :",
        sha256_file(
            execution_manifest_path
        ),
    )

    if args.prepare_only:
        print()
        print("=" * 78)
        print("PREPARATION COMPLETE — INFERENCE NOT STARTED")
        print("=" * 78)
        return

    if (
        batch_root.exists()
        and not args.resume
    ):
        raise AdaptationBatchProductionError(
            "Batch output directory already exists.\n"
            f"{batch_root}\n\n"
            "Use --resume only for an interrupted matching run."
        )

    run_inference(
        checkpoint=checkpoint,
        validation_dataset=validation_dataset,
        execution_manifest=execution_manifest_path,
        reference_batch_root=reference_batch_root,
        seed=args.seed,
        output_dir=batch_root,
        device=args.device,
        resume=args.resume,
    )

    audit = audit_completed_batch(
        execution=execution,
        execution_manifest_path=execution_manifest_path,
        checkpoint=checkpoint,
        method=args.method,
        validation_dataset=validation_dataset,
        reference_batch_root=reference_batch_root,
        batch_root=batch_root,
        seed=args.seed,
    )

    with audit_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            audit,
            file,
            indent=2,
            sort_keys=True,
        )
        file.write(
            "\n"
        )

    file_count = write_batch_checksum_inventory(
        batch_root=batch_root,
        output_path=checksum_path,
    )

    print()
    print("=" * 78)
    print("PRODUCTION AUDIT: PASS")
    print("=" * 78)
    print("Cases audited            :", EXPECTED_BATCH_SIZE)
    print("Problems                 :", 0)
    print("Files checksummed        :", file_count)
    print("Production audit         :", audit_path)
    print("Checksum inventory       :", checksum_path)

    print()
    print("=" * 78)
    print("BATCH PRODUCTION COMPLETE")
    print("=" * 78)


if __name__ == "__main__":
    try:
        main()

    except (
        AdaptationBatchProductionError,
        FileNotFoundError,
        NotADirectoryError,
        OSError,
        subprocess.CalledProcessError,
        ValueError,
        KeyError,
        TypeError,
    ) as exc:
        print(
            "\nDETERMINISTIC PEFT LIBRARY BATCH PRODUCTION FAILED",
            file=sys.stderr,
        )
        print(
            exc,
            file=sys.stderr,
        )
        sys.exit(
            1
        )
