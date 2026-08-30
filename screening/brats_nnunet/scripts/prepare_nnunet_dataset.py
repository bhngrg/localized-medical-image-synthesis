#!/usr/bin/env python3

"""
Prepare BraTS 2020 for nnU-Net external-cohort screening.

This script constructs an nnU-Net v2 dataset from the already registered
BraTS 2020 training and validation NIfTI releases.

Scientific purpose
------------------
The resulting nnU-Net model will be used only to identify validation slices
without predicted tumor involvement for construction of an external BR-LoRA
base-image cohort.

The segmentation target is binary whole tumor:

    0 -> background
    1 -> any original BraTS tumor label (1, 2, or 4)

The script does not modify the registered source datasets and does not repeat
their full registration scans.

Output
------
nnUNet_raw/
└── Dataset500_BraTS2020Screening/
    ├── imagesTr/
    ├── labelsTr/
    ├── imagesTs/
    └── dataset.json
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import nibabel as nib
import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[3]

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


SUPPORTED_SCHEMA_VERSION = 1

TRAINING_DATASET_ID = "brats2020_training"
VALIDATION_DATASET_ID = "brats2020_validation"

DEFAULT_NNUNET_DATASET_ID = 500
DEFAULT_NNUNET_DATASET_NAME = "BraTS2020Screening"

EXPECTED_TRAINING_SUBJECTS = 369
EXPECTED_VALIDATION_SUBJECTS = 125

EXPECTED_VOLUME_SHAPE = (
    240,
    240,
    155,
)

CHANNELS = (
    (
        0,
        "flair",
    ),
    (
        1,
        "t1",
    ),
    (
        2,
        "t1ce",
    ),
    (
        3,
        "t2",
    ),
)

EXPECTED_BRATS_LABELS = {
    0,
    1,
    2,
    4,
}


class DatasetPreparationError(
    ValueError
):
    """Raised when the screening dataset cannot be prepared."""


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Prepare registered BraTS 2020 training and validation releases "
            "for nnU-Net external-cohort screening."
        )
    )

    parser.add_argument(
        "--folders-file",
        type=Path,
        default=Path("data/folders.yaml"),
        help=(
            "Machine-specific path configuration YAML. "
            "Default: data/folders.yaml."
        ),
    )

    parser.add_argument(
        "--training-dataset",
        type=Path,
        default=None,
        help=(
            "Registered BraTS 2020 training dataset.yaml. Overrides "
            "yaml_dataset_path in --folders-file."
        ),
    )

    parser.add_argument(
        "--validation-dataset",
        type=Path,
        default=None,
        help=(
            "Registered BraTS 2020 validation_dataset.yaml. Overrides "
            "yaml_validation_dataset_path in --folders-file."
        ),
    )

    parser.add_argument(
        "--nnunet-raw",
        type=Path,
        default=None,
        help=(
            "nnU-Net raw-data root. If omitted, uses "
            "<nnunet_archive_root>/nnUNet_raw. The DatasetXXX_Name "
            "directory will be created beneath this path."
        ),
    )

    parser.add_argument(
        "--dataset-id",
        type=int,
        default=DEFAULT_NNUNET_DATASET_ID,
        help=(
            "Three-digit nnU-Net dataset identifier. "
            f"Default: {DEFAULT_NNUNET_DATASET_ID}."
        ),
    )

    parser.add_argument(
        "--dataset-name",
        type=str,
        default=DEFAULT_NNUNET_DATASET_NAME,
        help=(
            "nnU-Net dataset name. "
            f"Default: {DEFAULT_NNUNET_DATASET_NAME}."
        ),
    )

    parser.add_argument(
        "--limit-training",
        type=int,
        default=None,
        help=(
            "Optional development limit on the number of registered "
            "training subjects prepared. Omit for the complete training "
            "release."
        ),
    )

    parser.add_argument(
        "--limit-validation",
        type=int,
        default=None,
        help=(
            "Optional development limit on the number of registered "
            "validation subjects prepared. Omit for the complete validation "
            "release."
        ),
    )

    parser.add_argument(
        "--validate-only",
        action="store_true",
        help=(
            "Validate dataset specifications and preparation settings, then "
            "exit without creating an nnU-Net dataset."
        ),
    )

    return parser.parse_args()


def load_yaml(
    path: Path,
    *,
    name: str,
) -> dict:
    """Load one registered dataset specification."""

    resolved = (
        path
        .expanduser()
        .resolve()
    )

    if not resolved.is_file():
        raise FileNotFoundError(
            f"{name} not found:\n{resolved}"
        )

    try:
        with resolved.open(
            "r",
            encoding="utf-8",
        ) as file:
            payload = yaml.safe_load(
                file
            )

    except (
        OSError,
        yaml.YAMLError,
    ) as exc:
        raise DatasetPreparationError(
            f"{name} could not be read.\n\n"
            f"File:\n{resolved}\n\n"
            f"Error:\n{exc}"
        ) from exc

    if not isinstance(
        payload,
        dict,
    ):
        raise DatasetPreparationError(
            f"{name} must contain a top-level YAML mapping."
        )

    return payload


def require_mapping(
    parent: dict,
    key: str,
    *,
    source_name: str,
) -> dict:
    """Return one required YAML mapping."""

    value = parent.get(
        key
    )

    if not isinstance(
        value,
        dict,
    ):
        raise DatasetPreparationError(
            f"{source_name} is missing required mapping: {key}"
        )

    return value


def validate_common_specification(
    specification: dict,
    *,
    expected_dataset_id: str,
    expected_subject_count: int,
    source_name: str,
) -> dict:
    """Validate fields shared by registered training/validation datasets."""

    if specification.get(
        "schema_version"
    ) != SUPPORTED_SCHEMA_VERSION:
        raise DatasetPreparationError(
            f"{source_name} has an unsupported schema version."
        )

    dataset = require_mapping(
        specification,
        "dataset",
        source_name=source_name,
    )

    subjects = require_mapping(
        specification,
        "subjects",
        source_name=source_name,
    )

    volumes = require_mapping(
        specification,
        "volumes",
        source_name=source_name,
    )

    modalities = require_mapping(
        specification,
        "modalities",
        source_name=source_name,
    )

    validation = require_mapping(
        specification,
        "validation",
        source_name=source_name,
    )

    observed_dataset_id = dataset.get(
        "id"
    )

    if observed_dataset_id != expected_dataset_id:
        raise DatasetPreparationError(
            f"{source_name} has an unexpected dataset id.\n\n"
            f"Expected: {expected_dataset_id}\n"
            f"Observed: {observed_dataset_id}"
        )

    if validation.get(
        "status"
    ) != "passed":
        raise DatasetPreparationError(
            f"{source_name} does not record successful registration."
        )

    raw_root_value = dataset.get(
        "raw_data_root"
    )

    if not isinstance(
        raw_root_value,
        str,
    ) or not raw_root_value:
        raise DatasetPreparationError(
            f"{source_name} does not contain dataset.raw_data_root."
        )

    raw_root = (
        Path(
            raw_root_value
        )
        .expanduser()
        .resolve()
    )

    if not raw_root.is_dir():
        raise FileNotFoundError(
            f"Registered source dataset root is unavailable:\n{raw_root}"
        )

    subject_count = int(
        subjects.get(
            "count"
        )
    )

    if subject_count != expected_subject_count:
        raise DatasetPreparationError(
            f"{source_name} contains an unexpected subject count.\n\n"
            f"Expected: {expected_subject_count}\n"
            f"Observed: {subject_count}"
        )

    first_numeric_id = int(
        subjects.get(
            "first_numeric_id"
        )
    )

    last_numeric_id = int(
        subjects.get(
            "last_numeric_id"
        )
    )

    id_pattern = subjects.get(
        "id_pattern"
    )

    if not isinstance(
        id_pattern,
        str,
    ) or not id_pattern:
        raise DatasetPreparationError(
            f"{source_name} subjects.id_pattern is invalid."
        )

    volume_shape = tuple(
        int(
            value
        )
        for value in volumes.get(
            "shape",
            ()
        )
    )

    if volume_shape != EXPECTED_VOLUME_SHAPE:
        raise DatasetPreparationError(
            f"{source_name} has an unexpected registered volume shape.\n\n"
            f"Expected: {EXPECTED_VOLUME_SHAPE}\n"
            f"Observed: {volume_shape}"
        )

    modality_files = modalities.get(
        "files"
    )

    if not isinstance(
        modality_files,
        dict,
    ):
        raise DatasetPreparationError(
            f"{source_name} modalities.files must be a mapping."
        )

    for _, modality in CHANNELS:
        pattern = modality_files.get(
            modality
        )

        if not isinstance(
            pattern,
            str,
        ) or not pattern:
            raise DatasetPreparationError(
                f"{source_name} is missing modality pattern: {modality}"
            )

    return {
        "raw_root": raw_root,
        "subject_count": subject_count,
        "first_numeric_id": first_numeric_id,
        "last_numeric_id": last_numeric_id,
        "id_pattern": id_pattern,
        "modality_files": modality_files,
    }


def validate_training_specification(
    specification: dict,
) -> dict:
    """Validate registered BraTS training metadata."""

    registered = validate_common_specification(
        specification,
        expected_dataset_id=TRAINING_DATASET_ID,
        expected_subject_count=EXPECTED_TRAINING_SUBJECTS,
        source_name="Training dataset specification",
    )

    segmentation = require_mapping(
        specification,
        "segmentation",
        source_name="Training dataset specification",
    )

    pattern = segmentation.get(
        "file_pattern"
    )

    if not isinstance(
        pattern,
        str,
    ) or not pattern:
        raise DatasetPreparationError(
            "Training dataset specification is missing "
            "segmentation.file_pattern."
        )

    exceptions = segmentation.get(
        "filename_exceptions",
        {},
    )

    if not isinstance(
        exceptions,
        dict,
    ):
        raise DatasetPreparationError(
            "segmentation.filename_exceptions must be a mapping."
        )

    registered[
        "segmentation_pattern"
    ] = pattern

    registered[
        "segmentation_filename_exceptions"
    ] = exceptions

    return registered


def validate_validation_specification(
    specification: dict,
) -> dict:
    """Validate registered BraTS validation metadata."""

    registered = validate_common_specification(
        specification,
        expected_dataset_id=VALIDATION_DATASET_ID,
        expected_subject_count=EXPECTED_VALIDATION_SUBJECTS,
        source_name="Validation dataset specification",
    )

    segmentation = require_mapping(
        specification,
        "segmentation",
        source_name="Validation dataset specification",
    )

    if segmentation.get(
        "available"
    ) is not False:
        raise DatasetPreparationError(
            "BraTS validation specification must record "
            "segmentation.available=false."
        )

    return registered


def format_subject_name(
    *,
    id_pattern: str,
    numeric_id: int,
) -> str:
    """Format one registered BraTS subject name."""

    try:
        return id_pattern.format(
            id=numeric_id
        )

    except (
        KeyError,
        ValueError,
    ) as exc:
        raise DatasetPreparationError(
            "Registered subject id pattern could not be formatted."
        ) from exc


def resolve_pattern_path(
    *,
    subject_dir: Path,
    subject_name: str,
    pattern: str,
) -> Path:
    """Resolve one registered wildcard filename pattern."""

    filename = pattern.replace(
        "*",
        subject_name,
        1,
    )

    path = (
        subject_dir
        / filename
    )

    if not path.is_file():
        raise FileNotFoundError(
            "Registered source file is unavailable.\n\n"
            f"Expected:\n{path}"
        )

    return path


def resolve_segmentation_path(
    *,
    subject_dir: Path,
    subject_name: str,
    pattern: str,
    exceptions: dict,
) -> Path:
    """Resolve one registered BraTS training segmentation."""

    if subject_name in exceptions:
        filename = exceptions[
            subject_name
        ]

    else:
        filename = pattern.replace(
            "*",
            subject_name,
            1,
        )

    path = (
        subject_dir
        / filename
    )

    if not path.is_file():
        raise FileNotFoundError(
            "Registered training segmentation is unavailable.\n\n"
            f"Expected:\n{path}"
        )

    return path


def load_nifti(
    path: Path,
) -> nib.Nifti1Image:
    """Load and minimally validate one registered NIfTI image."""

    try:
        image = nib.load(
            str(
                path
            )
        )

    except Exception as exc:
        raise DatasetPreparationError(
            "NIfTI image could not be loaded.\n\n"
            f"File:\n{path}\n\n"
            f"Error:\n{exc}"
        ) from exc

    if tuple(
        image.shape
    ) != EXPECTED_VOLUME_SHAPE:
        raise DatasetPreparationError(
            "NIfTI image has an unexpected shape.\n\n"
            f"File:\n{path}\n"
            f"Expected: {EXPECTED_VOLUME_SHAPE}\n"
            f"Observed: {image.shape}"
        )

    return image


def save_image_as_nifti_gz(
    *,
    source_path: Path,
    output_path: Path,
) -> None:
    """Save one source MRI volume as compressed NIfTI without resampling."""

    if output_path.exists():
        raise DatasetPreparationError(
            f"Refusing to overwrite existing file:\n{output_path}"
        )

    image = load_nifti(
        source_path
    )

    try:
        nib.save(
            image,
            str(
                output_path
            ),
        )

    except Exception as exc:
        raise DatasetPreparationError(
            "MRI volume could not be written.\n\n"
            f"Source:\n{source_path}\n\n"
            f"Output:\n{output_path}\n\n"
            f"Error:\n{exc}"
        ) from exc


def save_binary_whole_tumor_label(
    *,
    source_path: Path,
    output_path: Path,
) -> None:
    """Convert one BraTS segmentation to binary whole tumor."""

    if output_path.exists():
        raise DatasetPreparationError(
            f"Refusing to overwrite existing file:\n{output_path}"
        )

    image = load_nifti(
        source_path
    )

    segmentation = np.asanyarray(
        image.dataobj
    )

    unique_labels = {
        int(
            value
        )
        for value in np.unique(
            segmentation
        )
    }

    unexpected = (
        unique_labels
        - EXPECTED_BRATS_LABELS
    )

    if unexpected:
        raise DatasetPreparationError(
            "Training segmentation contains unexpected BraTS labels.\n\n"
            f"File:\n{source_path}\n"
            f"Observed labels: {sorted(unique_labels)}\n"
            f"Unexpected: {sorted(unexpected)}"
        )

    binary = (
        segmentation
        > 0
    ).astype(
        np.uint8
    )

    binary_image = nib.Nifti1Image(
        binary,
        image.affine,
        image.header.copy(),
    )

    binary_image.set_data_dtype(
        np.uint8
    )

    try:
        nib.save(
            binary_image,
            str(
                output_path
            ),
        )

    except Exception as exc:
        raise DatasetPreparationError(
            "Binary whole-tumor label could not be written.\n\n"
            f"Source:\n{source_path}\n\n"
            f"Output:\n{output_path}\n\n"
            f"Error:\n{exc}"
        ) from exc


def validate_dataset_id(
    dataset_id: int,
) -> None:
    """Validate nnU-Net dataset identifier."""

    if (
        isinstance(
            dataset_id,
            bool,
        )
        or not isinstance(
            dataset_id,
            int,
        )
        or not (
            1
            <= dataset_id
            <= 999
        )
    ):
        raise DatasetPreparationError(
            "dataset-id must be an integer from 1 through 999."
        )


def validate_dataset_name(
    dataset_name: str,
) -> str:
    """Validate simple nnU-Net dataset name."""

    normalized = dataset_name.strip()

    if not normalized:
        raise DatasetPreparationError(
            "dataset-name must not be empty."
        )

    if not all(
        character.isalnum()
        or character == "_"
        for character in normalized
    ):
        raise DatasetPreparationError(
            "dataset-name may contain only letters, numbers, and underscores."
        )

    return normalized


def resolve_case_limit(
    *,
    requested: int | None,
    available: int,
    name: str,
) -> int:
    """
    Resolve an optional development case limit.

    When no limit is supplied, all registered subjects are retained.
    """

    if requested is None:
        return available

    if (
        isinstance(
            requested,
            bool,
        )
        or not isinstance(
            requested,
            int,
        )
        or requested <= 0
    ):
        raise DatasetPreparationError(
            f"{name} must be a positive integer when provided."
        )

    if requested > available:
        raise DatasetPreparationError(
            f"{name} exceeds the registered subject count.\n\n"
            f"Requested: {requested}\n"
            f"Available: {available}"
        )

    return requested


def create_output_structure(
    *,
    nnunet_raw: Path,
    dataset_id: int,
    dataset_name: str,
) -> dict:
    """Create one empty nnU-Net raw dataset directory."""

    root = (
        nnunet_raw
        .expanduser()
        .resolve()
    )

    if not root.exists():
        raise FileNotFoundError(
            "nnUNet raw-data root does not exist.\n\n"
            f"Path:\n{root}"
        )

    if not root.is_dir():
        raise DatasetPreparationError(
            f"nnUNet raw-data root is not a directory:\n{root}"
        )

    dataset_dir = (
        root
        / (
            f"Dataset{dataset_id:03d}_"
            f"{dataset_name}"
        )
    )

    if dataset_dir.exists():
        raise DatasetPreparationError(
            "Refusing to overwrite an existing nnU-Net dataset.\n\n"
            f"Dataset directory:\n{dataset_dir}"
        )

    images_tr = (
        dataset_dir
        / "imagesTr"
    )

    labels_tr = (
        dataset_dir
        / "labelsTr"
    )

    images_ts = (
        dataset_dir
        / "imagesTs"
    )

    images_tr.mkdir(
        parents=True,
        exist_ok=False,
    )

    labels_tr.mkdir(
        exist_ok=False,
    )

    images_ts.mkdir(
        exist_ok=False,
    )

    return {
        "dataset_dir": dataset_dir,
        "images_tr": images_tr,
        "labels_tr": labels_tr,
        "images_ts": images_ts,
    }


def prepare_training_cases(
    *,
    registered: dict,
    images_tr: Path,
    labels_tr: Path,
    case_count: int,
) -> int:
    """Prepare the selected registered BraTS 2020 training cases."""

    written_cases = 0

    numeric_ids = range(
        registered[
            "first_numeric_id"
        ],
        registered[
            "first_numeric_id"
        ] + case_count,
    )

    for index, numeric_id in enumerate(
        numeric_ids,
        start=1,
    ):
        subject_name = format_subject_name(
            id_pattern=registered[
                "id_pattern"
            ],
            numeric_id=numeric_id,
        )

        subject_dir = (
            registered[
                "raw_root"
            ]
            / subject_name
        )

        if not subject_dir.is_dir():
            raise FileNotFoundError(
                f"Training subject directory unavailable:\n{subject_dir}"
            )

        for channel_index, modality in CHANNELS:
            source = resolve_pattern_path(
                subject_dir=subject_dir,
                subject_name=subject_name,
                pattern=registered[
                    "modality_files"
                ][
                    modality
                ],
            )

            output = (
                images_tr
                / (
                    f"{subject_name}_"
                    f"{channel_index:04d}.nii.gz"
                )
            )

            save_image_as_nifti_gz(
                source_path=source,
                output_path=output,
            )

        segmentation_source = resolve_segmentation_path(
            subject_dir=subject_dir,
            subject_name=subject_name,
            pattern=registered[
                "segmentation_pattern"
            ],
            exceptions=registered[
                "segmentation_filename_exceptions"
            ],
        )

        segmentation_output = (
            labels_tr
            / (
                f"{subject_name}.nii.gz"
            )
        )

        save_binary_whole_tumor_label(
            source_path=segmentation_source,
            output_path=segmentation_output,
        )

        written_cases += 1

        if (
            index % 25 == 0
            or index == case_count
        ):
            print(
                f"  Prepared training cases: "
                f"{index:,} / "
                f"{case_count:,}",
                flush=True,
            )

    return written_cases


def prepare_validation_cases(
    *,
    registered: dict,
    images_ts: Path,
    case_count: int,
) -> int:
    """Prepare the selected registered BraTS 2020 validation cases."""

    written_cases = 0

    numeric_ids = range(
        registered[
            "first_numeric_id"
        ],
        registered[
            "first_numeric_id"
        ] + case_count,
    )

    for index, numeric_id in enumerate(
        numeric_ids,
        start=1,
    ):
        subject_name = format_subject_name(
            id_pattern=registered[
                "id_pattern"
            ],
            numeric_id=numeric_id,
        )

        subject_dir = (
            registered[
                "raw_root"
            ]
            / subject_name
        )

        if not subject_dir.is_dir():
            raise FileNotFoundError(
                f"Validation subject directory unavailable:\n{subject_dir}"
            )

        for channel_index, modality in CHANNELS:
            source = resolve_pattern_path(
                subject_dir=subject_dir,
                subject_name=subject_name,
                pattern=registered[
                    "modality_files"
                ][
                    modality
                ],
            )

            output = (
                images_ts
                / (
                    f"{subject_name}_"
                    f"{channel_index:04d}.nii.gz"
                )
            )

            save_image_as_nifti_gz(
                source_path=source,
                output_path=output,
            )

        written_cases += 1

        if (
            index % 25 == 0
            or index == case_count
        ):
            print(
                f"  Prepared validation cases: "
                f"{index:,} / "
                f"{case_count:,}",
                flush=True,
            )

    return written_cases


def write_dataset_json(
    *,
    dataset_dir: Path,
    training_case_count: int,
    dataset_name: str,
) -> Path:
    """Write nnU-Net v2 dataset.json for binary whole-tumor screening."""

    path = (
        dataset_dir
        / "dataset.json"
    )

    if path.exists():
        raise DatasetPreparationError(
            f"Refusing to overwrite existing dataset.json:\n{path}"
        )

    payload = {
        "channel_names": {
            "0": "FLAIR",
            "1": "T1",
            "2": "T1ce",
            "3": "T2",
        },

        "labels": {
            "background": 0,
            "whole_tumor": 1,
        },

        "numTraining": (
            training_case_count
        ),

        "file_ending": ".nii.gz",

        "name": (
            dataset_name
        ),

        "description": (
            "BraTS 2020 binary whole-tumor segmentation dataset used "
            "only for external-cohort screening in the localized medical "
            "image synthesis project."
        ),
    }

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            payload,
            file,
            indent=2,
            sort_keys=False,
        )

        file.write(
            "\n"
        )

    return path


def verify_output_counts(
    *,
    outputs: dict,
    training_cases: int,
    validation_cases: int,
) -> None:
    """Verify expected nnU-Net raw dataset file counts."""

    image_tr_count = len(
        list(
            outputs[
                "images_tr"
            ].glob(
                "*.nii.gz"
            )
        )
    )

    label_tr_count = len(
        list(
            outputs[
                "labels_tr"
            ].glob(
                "*.nii.gz"
            )
        )
    )

    image_ts_count = len(
        list(
            outputs[
                "images_ts"
            ].glob(
                "*.nii.gz"
            )
        )
    )

    expected_image_tr = (
        training_cases
        * len(
            CHANNELS
        )
    )

    expected_image_ts = (
        validation_cases
        * len(
            CHANNELS
        )
    )

    if image_tr_count != expected_image_tr:
        raise RuntimeError(
            "Unexpected imagesTr file count.\n\n"
            f"Expected: {expected_image_tr}\n"
            f"Observed: {image_tr_count}"
        )

    if label_tr_count != training_cases:
        raise RuntimeError(
            "Unexpected labelsTr file count.\n\n"
            f"Expected: {training_cases}\n"
            f"Observed: {label_tr_count}"
        )

    if image_ts_count != expected_image_ts:
        raise RuntimeError(
            "Unexpected imagesTs file count.\n\n"
            f"Expected: {expected_image_ts}\n"
            f"Observed: {image_ts_count}"
        )


def resolve_git_commit() -> str | None:
    """Return the current Git commit when available."""

    project_root = Path(
        __file__
    ).resolve().parents[
        3
    ]

    try:
        result = subprocess.run(
            [
                "git",
                "rev-parse",
                "HEAD",
            ],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        )

    except (
        OSError,
        subprocess.CalledProcessError,
    ):
        return None

    commit = result.stdout.strip()

    if not commit:
        return None

    return commit


def resolve_git_worktree_clean() -> bool | None:
    """Return whether the repository working tree is clean when available."""

    project_root = Path(
        __file__
    ).resolve().parents[
        3
    ]

    try:
        result = subprocess.run(
            [
                "git",
                "status",
                "--porcelain",
            ],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        )

    except (
        OSError,
        subprocess.CalledProcessError,
    ):
        return None

    return not bool(
        result.stdout.strip()
    )


def sha256_file(
    path: Path,
) -> str:
    """Compute SHA-256 for one file."""

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


def write_preparation_summary(
    *,
    dataset_dir: Path,
    training_dataset_path: Path,
    validation_dataset_path: Path,
    dataset_id: int,
    dataset_name: str,
    registered_training: int,
    selected_training: int,
    registered_validation: int,
    selected_validation: int,
) -> Path:
    """Write lightweight provenance for one prepared nnU-Net dataset."""

    path = (
        dataset_dir
        / "preparation_summary.json"
    )

    if path.exists():
        raise DatasetPreparationError(
            f"Refusing to overwrite preparation summary:\n{path}"
        )

    payload = {
        "dataset_id": dataset_id,
        "dataset_name": dataset_name,

        "training_dataset_specification": str(
            training_dataset_path
            .expanduser()
            .resolve()
        ),

        "validation_dataset_specification": str(
            validation_dataset_path
            .expanduser()
            .resolve()
        ),

        "registered_training_subjects": (
            registered_training
        ),

        "selected_training_subjects": (
            selected_training
        ),

        "registered_validation_subjects": (
            registered_validation
        ),

        "selected_validation_subjects": (
            selected_validation
        ),

        "channel_mapping": {
            "0000": "flair",
            "0001": "t1",
            "0002": "t1ce",
            "0003": "t2",
        },

        "segmentation_target": (
            "binary whole tumor"
        ),

        "segmentation_mapping": {
            "0": 0,
            "1": 1,
            "2": 1,
            "4": 1,
        },

        "expected_training_image_files": (
            selected_training
            * len(
                CHANNELS
            )
        ),

        "expected_training_label_files": (
            selected_training
        ),

        "expected_validation_image_files": (
            selected_validation
            * len(
                CHANNELS
            )
        ),

        "git_commit": resolve_git_commit(),

        "git_worktree_clean": (
            resolve_git_worktree_clean()
        ),

        "preparation_script_sha256": sha256_file(
            Path(
                __file__
            ).resolve()
        ),

        "prepared_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
    }

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            payload,
            file,
            indent=2,
            sort_keys=True,
        )

        file.write(
            "\n"
        )

    return path


def validate_prepared_dataset(
    *,
    outputs: dict,
    training_cases: int,
    validation_cases: int,
) -> dict:
    """
    Validate the completed nnU-Net raw dataset.

    MRI volumes are checked through their NIfTI headers and geometry.
    Training labels are additionally loaded to verify binary whole-tumor
    contents.
    """

    verify_output_counts(
        outputs=outputs,
        training_cases=training_cases,
        validation_cases=validation_cases,
    )

    training_images = sorted(
        outputs[
            "images_tr"
        ].glob(
            "*.nii.gz"
        )
    )

    training_labels = sorted(
        outputs[
            "labels_tr"
        ].glob(
            "*.nii.gz"
        )
    )

    validation_images = sorted(
        outputs[
            "images_ts"
        ].glob(
            "*.nii.gz"
        )
    )

    image_files = (
        training_images
        + validation_images
    )

    for image_path in image_files:
        image = nib.load(
            str(
                image_path
            )
        )

        if tuple(
            image.shape
        ) != EXPECTED_VOLUME_SHAPE:
            raise RuntimeError(
                "Prepared MRI volume has an unexpected shape.\n\n"
                f"File:\n{image_path}\n"
                f"Expected: {EXPECTED_VOLUME_SHAPE}\n"
                f"Observed: {image.shape}"
            )

        affine = np.asarray(
            image.affine
        )

        if affine.shape != (
            4,
            4,
        ):
            raise RuntimeError(
                "Prepared MRI volume has an invalid affine shape.\n\n"
                f"File:\n{image_path}\n"
                f"Observed: {affine.shape}"
            )

        if not np.isfinite(
            affine
        ).all():
            raise RuntimeError(
                "Prepared MRI volume contains a non-finite affine.\n\n"
                f"File:\n{image_path}"
            )

        zooms = image.header.get_zooms()[
            :3
        ]

        if (
            len(
                zooms
            )
            != 3
            or not np.isfinite(
                zooms
            ).all()
            or any(
                float(
                    zoom
                )
                <= 0
                for zoom in zooms
            )
        ):
            raise RuntimeError(
                "Prepared MRI volume has invalid spatial voxel sizes.\n\n"
                f"File:\n{image_path}\n"
                f"Observed: {zooms}"
            )

    total_positive_tumor_voxels = 0

    for label_path in training_labels:
        label_image = nib.load(
            str(
                label_path
            )
        )

        if tuple(
            label_image.shape
        ) != EXPECTED_VOLUME_SHAPE:
            raise RuntimeError(
                "Prepared training label has an unexpected shape.\n\n"
                f"File:\n{label_path}\n"
                f"Expected: {EXPECTED_VOLUME_SHAPE}\n"
                f"Observed: {label_image.shape}"
            )

        label = np.asanyarray(
            label_image.dataobj
        )

        unique_labels = {
            int(
                value
            )
            for value in np.unique(
                label
            )
        }

        if not unique_labels.issubset(
            {
                0,
                1,
            }
        ):
            raise RuntimeError(
                "Prepared training label is not binary.\n\n"
                f"File:\n{label_path}\n"
                f"Observed labels: {sorted(unique_labels)}"
            )

        positive_voxels = int(
            np.count_nonzero(
                label
            )
        )

        if positive_voxels <= 0:
            raise RuntimeError(
                "Prepared BraTS training label contains no tumor voxels.\n\n"
                f"File:\n{label_path}"
            )

        total_positive_tumor_voxels += (
            positive_voxels
        )

    dataset_json_path = (
        outputs[
            "dataset_dir"
        ]
        / "dataset.json"
    )

    with dataset_json_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        dataset_json = json.load(
            file
        )

    if dataset_json.get(
        "numTraining"
    ) != training_cases:
        raise RuntimeError(
            "dataset.json numTraining does not match the prepared "
            "training-case count."
        )

    if dataset_json.get(
        "file_ending"
    ) != ".nii.gz":
        raise RuntimeError(
            "dataset.json contains an unexpected file_ending."
        )

    if dataset_json.get(
        "labels"
    ) != {
        "background": 0,
        "whole_tumor": 1,
    }:
        raise RuntimeError(
            "dataset.json contains an unexpected label contract."
        )

    return {
        "training_image_files": len(
            training_images
        ),
        "training_label_files": len(
            training_labels
        ),
        "validation_image_files": len(
            validation_images
        ),
        "total_mri_files_checked": len(
            image_files
        ),
        "training_labels_checked": len(
            training_labels
        ),
        "total_positive_tumor_voxels": (
            total_positive_tumor_voxels
        ),
    }


def main() -> None:
    """Prepare the nnU-Net screening dataset."""

    args = parse_args()

    validate_dataset_id(
        args.dataset_id
    )

    dataset_name = validate_dataset_name(
        args.dataset_name
    )

    folders_config = load_folders_config(
        args.folders_file
    )

    training_dataset_path = resolve_path(
        key="yaml_dataset_path",
        cli_value=args.training_dataset,
        config=folders_config,
        selector=None,
    ).expanduser().resolve()

    validation_dataset_path = resolve_path(
        key="yaml_validation_dataset_path",
        cli_value=args.validation_dataset,
        config=folders_config,
        selector=None,
    ).expanduser().resolve()

    if args.nnunet_raw is not None:
        nnunet_raw = (
            args.nnunet_raw
            .expanduser()
            .resolve()
        )
    else:
        nnunet_archive_root = resolve_path(
            key="nnunet_archive_root",
            cli_value=None,
            config=folders_config,
            selector=None,
        ).expanduser().resolve()

        nnunet_raw = (
            nnunet_archive_root
            / "nnUNet_raw"
        )

    save_folders_config(
        args.folders_file,
        folders_config,
    )

    training_specification = load_yaml(
        training_dataset_path,
        name="Training dataset specification",
    )

    validation_specification = load_yaml(
        validation_dataset_path,
        name="Validation dataset specification",
    )

    training = validate_training_specification(
        training_specification
    )

    validation = validate_validation_specification(
        validation_specification
    )

    training_case_limit = resolve_case_limit(
        requested=args.limit_training,
        available=training[
            "subject_count"
        ],
        name="limit-training",
    )

    validation_case_limit = resolve_case_limit(
        requested=args.limit_validation,
        available=validation[
            "subject_count"
        ],
        name="limit-validation",
    )

    if args.validate_only:
        print()
        print("=" * 78)
        print("BraTS 2020 nnU-Net SCREENING DATASET VALIDATION")
        print("=" * 78)

        print(
            "Training dataset root    :",
            training[
                "raw_root"
            ],
        )

        print(
            "Validation dataset root  :",
            validation[
                "raw_root"
            ],
        )

        print(
            "Registered training      :",
            training[
                "subject_count"
            ],
        )

        print(
            "Selected training        :",
            training_case_limit,
        )

        print(
            "Registered validation    :",
            validation[
                "subject_count"
            ],
        )

        print(
            "Selected validation      :",
            validation_case_limit,
        )

        print(
            "Dataset ID               :",
            args.dataset_id,
        )

        print(
            "Dataset name             :",
            dataset_name,
        )

        print(
            "MRI channels             :",
            len(
                CHANNELS
            ),
        )

        print(
            "Segmentation target      : binary whole tumor",
        )

        print()
        print("VALIDATION ONLY: PASS")
        print(
            "No nnU-Net dataset directories or files were created."
        )
        print("=" * 78)

        return

    outputs = create_output_structure(
        nnunet_raw=nnunet_raw,
        dataset_id=args.dataset_id,
        dataset_name=dataset_name,
    )

    print()
    print("=" * 78)
    print("BraTS 2020 nnU-Net SCREENING DATASET PREPARATION")
    print("=" * 78)

    print(
        "Dataset                  :",
        outputs[
            "dataset_dir"
        ].name,
    )

    print(
        "Training subjects        :",
        training[
            "subject_count"
        ],
    )

    print(
        "Validation subjects      :",
        validation[
            "subject_count"
        ],
    )

    print(
        "MRI channels             :",
        len(
            CHANNELS
        ),
    )

    print(
        "Segmentation target      : binary whole tumor",
    )

    print(
        "Output                   :",
        outputs[
            "dataset_dir"
        ],
    )

    print()
    print("Preparing training cases...")

    training_cases = prepare_training_cases(
        registered=training,
        images_tr=outputs[
            "images_tr"
        ],
        labels_tr=outputs[
            "labels_tr"
        ],
        case_count=training_case_limit,
    )

    print()
    print("Preparing validation cases...")

    validation_cases = prepare_validation_cases(
        registered=validation,
        images_ts=outputs[
            "images_ts"
        ],
        case_count=validation_case_limit,
    )

    dataset_json = write_dataset_json(
        dataset_dir=outputs[
            "dataset_dir"
        ],
        training_case_count=training_cases,
        dataset_name=dataset_name,
    )

    validation_result = validate_prepared_dataset(
        outputs=outputs,
        training_cases=training_cases,
        validation_cases=validation_cases,
    )

    preparation_summary = write_preparation_summary(
        dataset_dir=outputs[
            "dataset_dir"
        ],
        training_dataset_path=args.training_dataset,
        validation_dataset_path=args.validation_dataset,
        dataset_id=args.dataset_id,
        dataset_name=dataset_name,
        registered_training=training[
            "subject_count"
        ],
        selected_training=training_cases,
        registered_validation=validation[
            "subject_count"
        ],
        selected_validation=validation_cases,
    )

    print()
    print("=" * 78)
    print("nnU-Net SCREENING DATASET PREPARATION: PASS")
    print("=" * 78)

    print(
        "Training cases           :",
        training_cases,
    )

    print(
        "Training image files     :",
        training_cases
        * len(
            CHANNELS
        ),
    )

    print(
        "Training label files     :",
        training_cases,
    )

    print(
        "Validation cases         :",
        validation_cases,
    )

    print(
        "Validation image files   :",
        validation_cases
        * len(
            CHANNELS
        ),
    )

    print(
        "dataset.json             :",
        dataset_json,
    )

    print(
        "Preparation summary      :",
        preparation_summary,
    )

    print(
        "Dataset directory        :",
        outputs[
            "dataset_dir"
        ],
    )

    print("=" * 78)


if __name__ == "__main__":
    try:
        main()

    except (
        DatasetPreparationError,
        FileNotFoundError,
        KeyError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        print(
            "\nBraTS nnU-Net SCREENING DATASET PREPARATION FAILED",
            file=sys.stderr,
        )

        print(
            exc,
            file=sys.stderr,
        )

        sys.exit(
            1
        )
