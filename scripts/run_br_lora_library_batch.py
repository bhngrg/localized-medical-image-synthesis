#!/usr/bin/env python3
"""
Run one frozen BR-LoRA synthetic-library production batch.

This script orchestrates, but does not reimplement, the validated external
BR-LoRA inference workflow.

For one frozen 250-case batch it:

1. validates the canonical library-design batch manifest,
2. deterministically constructs the five-column external execution manifest,
3. validates all referenced donor H5 files,
4. optionally stops after preparation,
5. invokes scripts/evaluate_br_lora_external.py,
6. independently audits all generated case artifacts,
7. validates evaluation_summary.json, and
8. writes a SHA-256 inventory of every file in the completed batch.

This script intentionally does NOT:

- select or redesign conditioning pairs,
- modify the frozen 10,000-case design,
- promote completed batches into the library,
- update the master library manifest,
- delete staging output.

Those operations remain separate acceptance/archive steps.
"""

from __future__ import annotations

import argparse
import csv
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
        str(PROJECT_ROOT),
    )

from src.config import (
    load_folders_config,
    resolve_path,
    save_folders_config,
)


DEFAULT_DESIGN_BATCH_DIR = (
    PROJECT_ROOT
    / "downstream_evaluation/manifests/"
      "br_lora_library_design_10000/batches"
)

DEFAULT_CHECKPOINT = (
    PROJECT_ROOT
    / "checkpoints/br_lora_full_train/final.pt"
)

DEFAULT_POSTERIOR_SAMPLES = 100
DEFAULT_EVALUATION_SEED = 42
DEFAULT_DEVICE = "mps"

EXPECTED_BATCH_SIZE = 250
FIRST_PRODUCTION_BATCH = 2
LAST_LIBRARY_BATCH = 40

EVALUATION_NAME = "br_lora_external_evaluation"

REQUIRED_CASE_ARTIFACTS = (
    "posterior_samples.pt",
    "posterior_mean.pt",
    "posterior_variance.pt",
    "posterior_std.pt",
    "composite_mean.pt",
    "metadata.json",
)

REQUIRED_POSTERIOR_KEYS = {
    "evaluation_name",
    "case_id",
    "checkpoint",
    "evaluation_seed",
    "case_seed",
    "posterior_samples",
    "resample_diffusion_noise",
    "prediction_samples",
    "base_image",
    "transferred_mask",
    "known",
    "donor_image",
    "donor_patch",
    "donor_condition",
    "timestep",
    "diffusion_noise",
    "x_t",
}

TENSOR_CONTRACT = {
    "prediction_samples":
        (100, 1, 1, 240, 240),
    "base_image":
        (1, 240, 240),
    "transferred_mask":
        (1, 240, 240),
    "known":
        (1, 240, 240),
    "donor_image":
        (1, 240, 240),
    "donor_patch":
        (1, 240, 240),
    "donor_condition":
        (4,),
    "timestep":
        (1,),
    "diffusion_noise":
        (1, 1, 240, 240),
    "x_t":
        (1, 1, 240, 240),
}


class BatchProductionError(
    RuntimeError
):
    """Raised when a frozen production batch fails validation."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare, run, and audit one frozen "
            "250-case BR-LoRA library batch."
        )
    )

    parser.add_argument(
        "--batch",
        required=True,
        help=(
            "Batch identifier, for example batch_0003."
        ),
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
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
    )

    parser.add_argument(
        "--staging-root",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--folders-file",
        type=Path,
        default=Path("data/folders.yaml"),
        help="Machine-specific folders configuration file.",
    )

    parser.add_argument(
        "--posterior-samples",
        type=int,
        default=DEFAULT_POSTERIOR_SAMPLES,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_EVALUATION_SEED,
    )

    parser.add_argument(
        "--device",
        choices=[
            "auto",
            "cpu",
            "mps",
            "cuda",
        ],
        default=DEFAULT_DEVICE,
    )

    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help=(
            "Create and validate the execution manifest, "
            "then stop before inference."
        ),
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Pass --resume to evaluate_br_lora_external.py."
        ),
    )

    return parser.parse_args()


def resolve_production_paths(
    args: argparse.Namespace,
) -> tuple[Path, Path, Path]:
    """Resolve and persist machine-specific production paths."""
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

    staging_root = resolve_path(
        key="br_lora_staging_root",
        cli_value=args.staging_root,
        config=config,
        selector=None,
    )

    save_folders_config(
        args.folders_file,
        config,
    )

    return (
        h5_root,
        validation_dataset,
        staging_root,
    )


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
        raise BatchProductionError(
            "--batch must use the form batch_0003."
        )

    suffix = text[
        len("batch_"):
    ]

    if (
        len(suffix) != 4
        or not suffix.isdigit()
    ):
        raise BatchProductionError(
            "--batch must use four numeric digits, "
            "for example batch_0003."
        )

    number = int(
        suffix
    )

    if not (
        FIRST_PRODUCTION_BATCH
        <= number
        <= LAST_LIBRARY_BATCH
    ):
        raise BatchProductionError(
            "Production batch number must lie between "
            f"{FIRST_PRODUCTION_BATCH:04d} and "
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
                1024 * 1024
            ),
            b"",
        ):
            digest.update(
                block
            )

    return digest.hexdigest()


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
        raise BatchProductionError(
            "Frozen batch manifest is missing required columns:\n"
            + "\n".join(
                sorted(
                    missing
                )
            )
        )

    if len(table) != EXPECTED_BATCH_SIZE:
        raise BatchProductionError(
            f"{batch_id} must contain exactly "
            f"{EXPECTED_BATCH_SIZE} rows; "
            f"observed {len(table)}."
        )

    if set(
        table[
            "batch_id"
        ].astype(str)
    ) != {
        batch_id
    }:
        raise BatchProductionError(
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

    expected_indices = list(
        range(
            expected_first_index,
            expected_last_index + 1,
        )
    )

    observed_indices = (
        table[
            "library_index"
        ]
        .astype(int)
        .tolist()
    )

    if observed_indices != expected_indices:
        raise BatchProductionError(
            "Frozen batch library_index sequence is incorrect.\n"
            f"Expected: {expected_first_index}-"
            f"{expected_last_index}"
        )

    expected_case_ids = [
        f"synthetic_{index:06d}"
        for index in expected_indices
    ]

    observed_case_ids = (
        table[
            "library_case_id"
        ]
        .astype(str)
        .tolist()
    )

    if observed_case_ids != expected_case_ids:
        raise BatchProductionError(
            "Frozen batch library_case_id sequence does not "
            "match its library_index sequence."
        )

    if table[
        "library_case_id"
    ].duplicated().any():
        raise BatchProductionError(
            "Frozen batch contains duplicate library_case_id values."
        )

    if table[
        "donor_h5_file"
    ].duplicated().any():
        raise BatchProductionError(
            "Frozen batch contains duplicate donor slices."
        )

    execution = pd.DataFrame(
        {
            "case_id":
                table[
                    "library_case_id"
                ].astype(str),

            "external_subject_numeric_id":
                table[
                    "external_subject_numeric_id"
                ].astype(int),

            "external_slice_index":
                table[
                    "external_slice_index"
                ].astype(int),

            "external_modality":
                (
                    table[
                        "external_modality"
                    ]
                    .astype(str)
                    .str.lower()
                ),

            "donor_h5_path":
                table[
                    "donor_h5_file"
                ]
                .astype(str)
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
        raise BatchProductionError(
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


def validate_execution_manifest_with_loader(
    execution_manifest_path: Path,
) -> None:
    """
    Validate through the same loader used by the inference engine.

    Import is local so running this script from the repository root
    follows the same project import contract as other scripts.
    """

    if str(
        PROJECT_ROOT
    ) not in sys.path:
        sys.path.insert(
            0,
            str(
                PROJECT_ROOT
            ),
        )

    from src.inference.external_manifest import (
        load_external_evaluation_manifest,
    )

    cases = load_external_evaluation_manifest(
        execution_manifest_path
    )

    if len(cases) != EXPECTED_BATCH_SIZE:
        raise BatchProductionError(
            "Execution-manifest loader did not return 250 cases."
        )

    case_ids = [
        case.case_id
        for case in cases
    ]

    if len(
        set(
            case_ids
        )
    ) != EXPECTED_BATCH_SIZE:
        raise BatchProductionError(
            "Execution-manifest loader returned duplicate case IDs."
        )


def run_inference(
    *,
    checkpoint: Path,
    validation_dataset: Path,
    execution_manifest: Path,
    posterior_samples: int,
    seed: int,
    output_dir: Path,
    device: str,
    resume: bool,
) -> None:
    script = (
        PROJECT_ROOT
        / "scripts/evaluate_br_lora_external.py"
    )

    if not script.is_file():
        raise FileNotFoundError(
            f"Inference script not found:\n{script}"
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
        "--posterior-samples",
        str(
            posterior_samples
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
    print("LAUNCHING VALIDATED BR-LoRA INFERENCE ENGINE")
    print("=" * 78)
    print()
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
    validation_dataset: Path,
    batch_root: Path,
    posterior_samples: int,
    seed: int,
) -> dict[str, object]:
    problems: list[str] = []

    expected_case_ids = (
        execution[
            "case_id"
        ]
        .astype(str)
        .tolist()
    )

    actual_case_dirs = sorted(
        path.name
        for path in batch_root.glob(
            "synthetic_*"
        )
        if path.is_dir()
    )

    if actual_case_dirs != sorted(
        expected_case_ids
    ):
        problems.append(
            "Output case-directory set does not exactly match "
            "the execution manifest."
        )

    summary_path = (
        batch_root
        / "evaluation_summary.json"
    )

    if not summary_path.is_file():
        problems.append(
            "Missing evaluation_summary.json."
        )

    expected_checkpoint = str(
        checkpoint
    )

    expected_manifest = str(
        execution_manifest_path
    )

    for row in execution.itertuples(
        index=False
    ):
        case_id = str(
            row.case_id
        )

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

        metadata_path = (
            case_dir
            / "metadata.json"
        )

        if metadata_path.is_file():
            try:
                with metadata_path.open(
                    "r",
                    encoding="utf-8",
                ) as file:
                    metadata = json.load(
                        file
                    )
            except (
                OSError,
                json.JSONDecodeError,
            ) as exc:
                problems.append(
                    f"{case_id}: unreadable metadata.json: {exc}"
                )
                metadata = {}

            expected_metadata = {
                "evaluation_name":
                    EVALUATION_NAME,

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

                "donor_h5_path":
                    str(
                        Path(
                            row.donor_h5_path
                        ).resolve()
                    ),

                "checkpoint":
                    expected_checkpoint,

                "validation_dataset":
                    str(
                        validation_dataset
                    ),

                "evaluation_manifest":
                    expected_manifest,

                "evaluation_seed":
                    int(
                        seed
                    ),

                "posterior_samples":
                    int(
                        posterior_samples
                    ),

                "resample_diffusion_noise":
                    False,
            }

            for key, expected in (
                expected_metadata.items()
            ):
                observed = metadata.get(
                    key
                )

                if observed != expected:
                    problems.append(
                        f"{case_id}: metadata {key} "
                        f"expected {expected!r}, "
                        f"observed {observed!r}"
                    )

        posterior_path = (
            case_dir
            / "posterior_samples.pt"
        )

        if posterior_path.is_file():
            try:
                payload = torch.load(
                    posterior_path,
                    map_location="cpu",
                    weights_only=False,
                )
            except Exception as exc:
                problems.append(
                    f"{case_id}: posterior_samples.pt "
                    f"could not be loaded: {exc}"
                )
                payload = None

            if not isinstance(
                payload,
                dict,
            ):
                problems.append(
                    f"{case_id}: posterior_samples.pt "
                    "is not a dictionary."
                )

            else:
                missing_keys = (
                    REQUIRED_POSTERIOR_KEYS
                    - set(
                        payload
                    )
                )

                if missing_keys:
                    problems.append(
                        f"{case_id}: posterior_samples.pt "
                        f"missing keys {sorted(missing_keys)}"
                    )

                expected_payload = {
                    "evaluation_name":
                        EVALUATION_NAME,
                    "case_id":
                        case_id,
                    "checkpoint":
                        expected_checkpoint,
                    "evaluation_seed":
                        int(
                            seed
                        ),
                    "posterior_samples":
                        int(
                            posterior_samples
                        ),
                    "resample_diffusion_noise":
                        False,
                }

                for key, expected in (
                    expected_payload.items()
                ):
                    observed = payload.get(
                        key
                    )

                    if observed != expected:
                        problems.append(
                            f"{case_id}: posterior payload {key} "
                            f"expected {expected!r}, "
                            f"observed {observed!r}"
                        )

                tensor_contract = dict(
                    TENSOR_CONTRACT
                )

                tensor_contract[
                    "prediction_samples"
                ] = (
                    posterior_samples,
                    1,
                    1,
                    240,
                    240,
                )

                for key, expected_shape in (
                    tensor_contract.items()
                ):
                    value = payload.get(
                        key
                    )

                    if not isinstance(
                        value,
                        torch.Tensor,
                    ):
                        problems.append(
                            f"{case_id}: {key} is not a tensor."
                        )
                        continue

                    if tuple(
                        value.shape
                    ) != expected_shape:
                        problems.append(
                            f"{case_id}: {key} shape "
                            f"{tuple(value.shape)} != "
                            f"{expected_shape}"
                        )

                    if not torch.isfinite(
                        value
                    ).all():
                        problems.append(
                            f"{case_id}: {key} contains "
                            "non-finite values."
                        )

                mask = payload.get(
                    "transferred_mask"
                )

                if isinstance(
                    mask,
                    torch.Tensor,
                ):
                    unique_values = torch.unique(
                        mask
                    )

                    if not torch.all(
                        (
                            unique_values
                            == 0
                        )
                        | (
                            unique_values
                            == 1
                        )
                    ):
                        problems.append(
                            f"{case_id}: transferred_mask "
                            "is not binary."
                        )

    summary = {}

    if summary_path.is_file():
        try:
            with summary_path.open(
                "r",
                encoding="utf-8",
            ) as file:
                summary = json.load(
                    file
                )
        except (
            OSError,
            json.JSONDecodeError,
        ) as exc:
            problems.append(
                f"evaluation_summary.json unreadable: {exc}"
            )

    expected_summary = {
        "evaluation_name":
            EVALUATION_NAME,

        "checkpoint":
            expected_checkpoint,

        "validation_dataset":
            str(
                validation_dataset
            ),

        "evaluation_manifest":
            expected_manifest,

        "seed":
            int(
                seed
            ),

        "posterior_samples_per_case":
            int(
                posterior_samples
            ),

        "fixed_diffusion_noise":
            True,

        "case_count":
            EXPECTED_BATCH_SIZE,

        "completed_case_count":
            EXPECTED_BATCH_SIZE,
    }

    for key, expected in (
        expected_summary.items()
    ):
        observed = summary.get(
            key
        )

        if observed != expected:
            problems.append(
                "evaluation_summary.json "
                f"{key} expected {expected!r}, "
                f"observed {observed!r}"
            )

    if problems:
        print()
        print("=" * 78)
        print("PRODUCTION AUDIT: FAIL")
        print("=" * 78)
        print(
            f"Problems: {len(problems)}"
        )
        print()

        for item in problems[
            :30
        ]:
            print(
                " -",
                item,
            )

        raise BatchProductionError(
            f"Production audit found "
            f"{len(problems)} problem(s)."
        )

    return {
        "status":
            "pass",

        "audited_at_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "case_count":
            EXPECTED_BATCH_SIZE,

        "required_artifacts_per_case":
            len(
                REQUIRED_CASE_ARTIFACTS
            ),

        "posterior_samples_per_case":
            int(
                posterior_samples
            ),

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
        raise BatchProductionError(
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

            digest = sha256_file(
                path
            )

            file.write(
                f"{digest}  ./{relative.as_posix()}\n"
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
        staging_root,
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

    checkpoint = resolve_existing_file(
        args.checkpoint,
        name="BR-LoRA checkpoint",
    )

    if args.posterior_samples <= 1:
        raise BatchProductionError(
            "--posterior-samples must be greater than one."
        )

    design_manifest_path = (
        design_batch_dir
        / f"{batch_id}_manifest.csv"
    )

    design_manifest_path = resolve_existing_file(
        design_manifest_path,
        name="Frozen batch manifest",
    )

    staging_root = (
        staging_root
        .expanduser()
        .resolve()
    )

    execution_manifest_path = (
        staging_root
        / f"br_lora_{batch_id}_external_evaluation_manifest.csv"
    )

    staging_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    batch_root = (
        staging_root
        / batch_id
    )

    audit_path = (
        staging_root
        / f"br_lora_{batch_id}_production_audit.json"
    )

    checksum_path = (
        staging_root
        / f"br_lora_{batch_id}_sha256.txt"
    )

    print()
    print("=" * 78)
    print("BR-LoRA LIBRARY BATCH PRODUCTION")
    print("=" * 78)
    print()
    print(
        "Batch                    :",
        batch_id,
    )
    print(
        "Frozen design manifest   :",
        design_manifest_path,
    )
    print(
        "Execution manifest       :",
        execution_manifest_path,
    )
    print(
        "Checkpoint               :",
        checkpoint,
    )
    print(
        "Posterior samples        :",
        args.posterior_samples,
    )
    print(
        "Evaluation seed          :",
        args.seed,
    )
    print(
        "Device                   :",
        args.device,
    )
    print(
        "Output                   :",
        batch_root,
    )
    print(
        "Prepare only             :",
        args.prepare_only,
    )
    print(
        "Resume                   :",
        args.resume,
    )

    execution = prepare_execution_manifest(
        batch_id=batch_id,
        batch_number=batch_number,
        design_manifest_path=design_manifest_path,
        execution_manifest_path=execution_manifest_path,
        h5_root=h5_root,
    )

    validate_execution_manifest_with_loader(
        execution_manifest_path
    )

    print()
    print("===== EXECUTION MANIFEST: PASS =====")
    print(
        "Cases                    :",
        len(
            execution
        ),
    )
    print(
        "Case IDs                 :",
        execution[
            "case_id"
        ].iloc[0],
        "to",
        execution[
            "case_id"
        ].iloc[-1],
    )
    print(
        "External subjects        :",
        execution[
            "external_subject_numeric_id"
        ].nunique(),
    )
    print(
        "Unique base slices       :",
        execution[
            [
                "external_subject_numeric_id",
                "external_slice_index",
            ]
        ]
        .drop_duplicates()
        .shape[0],
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
        raise BatchProductionError(
            "Batch output directory already exists.\n"
            f"{batch_root}\n\n"
            "Use --resume only for an interrupted run whose "
            "existing cases satisfy the inference engine's strict "
            "resume checks."
        )

    run_inference(
        checkpoint=checkpoint,
        validation_dataset=validation_dataset,
        execution_manifest=execution_manifest_path,
        posterior_samples=args.posterior_samples,
        seed=args.seed,
        output_dir=batch_root,
        device=args.device,
        resume=args.resume,
    )

    audit = audit_completed_batch(
        execution=execution,
        execution_manifest_path=execution_manifest_path,
        checkpoint=checkpoint,
        validation_dataset=validation_dataset,
        batch_root=batch_root,
        posterior_samples=args.posterior_samples,
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
    print(
        "Cases audited            :",
        EXPECTED_BATCH_SIZE,
    )
    print(
        "Problems                 :",
        0,
    )
    print(
        "Files checksummed        :",
        file_count,
    )
    print(
        "Production audit         :",
        audit_path,
    )
    print(
        "Checksum inventory       :",
        checksum_path,
    )

    print()
    print("=" * 78)
    print("BATCH PRODUCTION COMPLETE")
    print("=" * 78)
    print()
    print(
        "Staging output was preserved."
    )


if __name__ == "__main__":
    try:
        main()

    except (
        BatchProductionError,
        FileNotFoundError,
        NotADirectoryError,
        OSError,
        subprocess.CalledProcessError,
        ValueError,
    ) as exc:
        print(
            "\nBR-LoRA LIBRARY BATCH PRODUCTION FAILED",
            file=sys.stderr,
        )
        print(
            exc,
            file=sys.stderr,
        )
        sys.exit(
            1
        )
