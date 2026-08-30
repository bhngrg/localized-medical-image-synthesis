#!/usr/bin/env python3
"""
Screen BraTS 2020 validation slices using predicted nnU-Net whole-tumor masks.

This script converts subject-level binary tumor predictions into an auditable
slice-level screening table.

The primary tumor-free definition mirrors the historical composition pipeline:

    predicted_tumor_pixels == 0

Additional quantities such as distance to the nearest predicted tumor slice
and contiguous tumor-free run length are retained only as descriptive/audit
variables. They do not alter the primary tumor-free candidate definition.

Expected nnU-Net prediction naming
----------------------------------
BraTS20_Validation_001.nii.gz
BraTS20_Validation_002.nii.gz
...
BraTS20_Validation_125.nii.gz

Primary outputs
---------------
validation_slice_screening.csv
validation_slice_screening_summary.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any

import nibabel as nib
import numpy as np
import pandas as pd
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


EXPECTED_PREDICTION_SUFFIX = ".nii.gz"

OUTPUT_CSV_NAME = (
    "validation_slice_screening.csv"
)

OUTPUT_SUMMARY_NAME = (
    "validation_slice_screening_summary.json"
)


class ValidationScreeningError(
    RuntimeError
):
    """Raised when validation-slice screening fails."""


@dataclass(
    frozen=True
)
class RegisteredValidationDataset:
    """Minimal registered-validation dataset contract."""

    specification_path: Path
    dataset_root: Path
    subject_count: int
    first_numeric_id: int
    id_pattern: str
    modality_files: dict[str, str]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Screen BraTS validation slices using "
            "binary nnU-Net tumor predictions."
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
        "--validation-dataset",
        type=Path,
        default=None,
        help=(
            "Registered BraTS validation dataset YAML specification. "
            "Overrides yaml_validation_dataset_path in --folders-file."
        ),
    )

    parser.add_argument(
        "--prediction-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing one nnU-Net prediction NIfTI per "
            "validation subject. If omitted, uses "
            "<nnunet_run_root>/validation_predictions_l40s_normal_q."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory for screening CSV and summary JSON. If omitted, "
            "uses <nnunet_run_root>/validation_slice_screening."
        ),
    )

    parser.add_argument(
        "--reference-modality",
        default="flair",
        help=(
            "Validation MRI modality used to "
            "verify prediction geometry. "
            "Default: flair."
        ),
    )

    parser.add_argument(
        "--validate-only",
        action="store_true",
        help=(
            "Validate the registered cohort and "
            "prediction contract without writing "
            "screening outputs."
        ),
    )

    return parser.parse_args()


def sha256_file(
    path: Path,
) -> str:
    """Return the SHA-256 hash of one file."""

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


def resolve_git_commit() -> str | None:
    """Return current Git commit when available."""

    try:
        result = subprocess.run(
            [
                "git",
                "rev-parse",
                "HEAD",
            ],
            check=True,
            capture_output=True,
            text=True,
        )

    except (
        subprocess.CalledProcessError,
        FileNotFoundError,
    ):
        return None

    value = result.stdout.strip()

    return value or None


def resolve_git_worktree_clean() -> bool | None:
    """Return whether the current Git worktree is clean."""

    try:
        result = subprocess.run(
            [
                "git",
                "status",
                "--porcelain",
            ],
            check=True,
            capture_output=True,
            text=True,
        )

    except (
        subprocess.CalledProcessError,
        FileNotFoundError,
    ):
        return None

    return (
        result.stdout.strip()
        == ""
    )


def require_positive_integer(
    value: Any,
    *,
    name: str,
) -> int:
    """Validate one positive integer."""

    if isinstance(
        value,
        bool,
    ):
        raise ValidationScreeningError(
            f"`{name}` must be a positive integer."
        )

    try:
        integer = int(
            value
        )

    except (
        TypeError,
        ValueError,
    ) as exc:
        raise ValidationScreeningError(
            f"`{name}` must be a positive integer."
        ) from exc

    if integer <= 0:
        raise ValidationScreeningError(
            f"`{name}` must be positive."
        )

    return integer


def load_validation_specification(
    path: Path,
) -> RegisteredValidationDataset:
    """Load the registered validation dataset specification."""

    path = path.expanduser().resolve()

    if not path.is_file():
        raise FileNotFoundError(
            "Validation dataset specification "
            f"does not exist:\n{path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        payload = yaml.safe_load(
            file
        )

    if not isinstance(
        payload,
        dict,
    ):
        raise ValidationScreeningError(
            "Validation dataset specification "
            "must contain a YAML mapping."
        )

    dataset_root_value = (
        payload.get(
            "dataset_root"
        )
        or payload.get(
            "raw_root"
        )
        or payload.get(
            "root"
        )
    )

    if dataset_root_value is None:
        # Registered YAML currently lives inside
        # the raw validation dataset root.
        dataset_root = path.parent

    else:
        dataset_root = Path(
            str(
                dataset_root_value
            )
        ).expanduser()

        if not dataset_root.is_absolute():
            dataset_root = (
                path.parent
                / dataset_root
            )

        dataset_root = (
            dataset_root.resolve()
        )

    if not dataset_root.is_dir():
        raise ValidationScreeningError(
            "Validation dataset root "
            f"does not exist:\n{dataset_root}"
        )

    subject_count = (
        payload.get(
            "subject_count"
        )
        or payload.get(
            "registered_subjects"
        )
        or payload.get(
            "registered_subject_count"
        )
        or payload.get(
            "num_subjects"
        )
    )

    if subject_count is None:
        # Known registered BraTS 2020
        # validation cohort size.
        subject_count = 125

    subject_count = require_positive_integer(
        subject_count,
        name="subject_count",
    )

    first_numeric_id = int(
        payload.get(
            "first_numeric_id",
            1,
        )
    )

    id_pattern = str(
        payload.get(
            "id_pattern",
            "BraTS20_Validation_{numeric_id:03d}",
        )
    )

    modality_payload = (
        payload.get(
            "modality_files"
        )
        or payload.get(
            "modalities"
        )
        or {}
    )

    modality_files: dict[str, str] = {}

    if isinstance(
        modality_payload,
        dict,
    ):
        for key, value in (
            modality_payload.items()
        ):
            if isinstance(
                value,
                str,
            ):
                modality_files[
                    str(
                        key
                    ).lower()
                ] = value

    return RegisteredValidationDataset(
        specification_path=path,
        dataset_root=dataset_root,
        subject_count=subject_count,
        first_numeric_id=first_numeric_id,
        id_pattern=id_pattern,
        modality_files=modality_files,
    )


def format_subject_name(
    *,
    id_pattern: str,
    numeric_id: int,
) -> str:
    """Format one registered validation subject name."""

    formats = (
        {
            "numeric_id":
                numeric_id,
        },
        {
            "id":
                numeric_id,
        },
        {
            "subject_id":
                numeric_id,
        },
    )

    for values in formats:
        try:
            return id_pattern.format(
                **values
            )
        except KeyError:
            continue

    raise ValidationScreeningError(
        "Unable to format validation subject "
        f"name from id_pattern={id_pattern!r}."
    )


def prediction_filename(
    subject_name: str,
) -> str:
    """Return expected nnU-Net prediction filename."""

    return (
        subject_name
        + EXPECTED_PREDICTION_SUFFIX
    )


def resolve_reference_modality(
    *,
    dataset: RegisteredValidationDataset,
    subject_name: str,
    modality: str,
) -> Path:
    """Resolve one registered validation MRI modality."""

    subject_dir = (
        dataset.dataset_root
        / subject_name
    )

    if not subject_dir.is_dir():
        raise ValidationScreeningError(
            "Validation subject directory "
            f"does not exist:\n{subject_dir}"
        )

    modality = modality.lower()

    pattern = dataset.modality_files.get(
        modality
    )

    candidates: list[Path] = []

    if pattern is not None:
        formatted = pattern.format(
            subject_name=subject_name,
            subject=subject_name,
        )

        candidates.append(
            subject_dir
            / formatted
        )

    candidates.extend(
        [
            subject_dir
            / (
                f"{subject_name}_"
                f"{modality}.nii"
            ),
            subject_dir
            / (
                f"{subject_name}_"
                f"{modality}.nii.gz"
            ),
        ]
    )

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    matches = sorted(
        subject_dir.glob(
            f"*_{modality}.nii*"
        )
    )

    if len(
        matches
    ) == 1:
        return matches[
            0
        ].resolve()

    raise ValidationScreeningError(
        "Unable to resolve reference "
        f"modality `{modality}` for "
        f"{subject_name}."
    )


def load_nifti_array(
    path: Path,
) -> tuple[
    np.ndarray,
    np.ndarray,
]:
    """Load one NIfTI image and affine."""

    if not path.is_file():
        raise FileNotFoundError(
            f"NIfTI file unavailable:\n{path}"
        )

    image = nib.load(
        str(
            path
        )
    )

    array = np.asarray(
        image.dataobj
    )

    if array.ndim != 3:
        raise ValidationScreeningError(
            f"Expected a 3-D NIfTI volume, "
            f"observed shape {array.shape}:\n"
            f"{path}"
        )

    if not np.isfinite(
        array
    ).all():
        raise ValidationScreeningError(
            "NIfTI volume contains "
            f"non-finite values:\n{path}"
        )

    affine = np.asarray(
        image.affine,
        dtype=np.float64,
    )

    return (
        array,
        affine,
    )


def validate_binary_prediction(
    prediction: np.ndarray,
    *,
    path: Path,
) -> np.ndarray:
    """Validate and return one binary whole-tumor mask."""

    unique = np.unique(
        prediction
    )

    allowed = np.array(
        [
            0,
            1,
        ]
    )

    if not np.isin(
        unique,
        allowed,
    ).all():
        raise ValidationScreeningError(
            "Prediction must be binary with "
            "labels {0, 1}.\n"
            f"Observed labels: {unique.tolist()}\n"
            f"Prediction: {path}"
        )

    return prediction.astype(
        np.uint8,
        copy=False,
    )


def nearest_positive_slice_distances(
    tumor_pixels: np.ndarray,
) -> np.ndarray:
    """
    Compute distance to the nearest tumor-positive slice.

    Tumor-positive slices receive distance zero.

    If an entire subject has no predicted tumor,
    all distances are returned as -1.
    """

    tumor_pixels = np.asarray(
        tumor_pixels
    )

    positive = np.flatnonzero(
        tumor_pixels > 0
    )

    slice_count = int(
        tumor_pixels.shape[
            0
        ]
    )

    if positive.size == 0:
        return np.full(
            slice_count,
            -1,
            dtype=np.int64,
        )

    indices = np.arange(
        slice_count,
        dtype=np.int64,
    )

    distances = np.min(
        np.abs(
            indices[
                :,
                None,
            ]
            - positive[
                None,
                :,
            ]
        ),
        axis=1,
    )

    return distances.astype(
        np.int64,
        copy=False,
    )


def contiguous_zero_run_lengths(
    tumor_pixels: np.ndarray,
) -> np.ndarray:
    """
    Return contiguous tumor-free run length for every slice.

    Tumor-positive slices receive zero.
    """

    tumor_free = (
        np.asarray(
            tumor_pixels
        )
        == 0
    )

    result = np.zeros(
        tumor_free.shape[
            0
        ],
        dtype=np.int64,
    )

    start = 0
    slice_count = int(
        tumor_free.shape[
            0
        ]
    )

    while start < slice_count:
        if not tumor_free[
            start
        ]:
            start += 1
            continue

        end = start

        while (
            end < slice_count
            and tumor_free[
                end
            ]
        ):
            end += 1

        run_length = (
            end
            - start
        )

        result[
            start:end
        ] = run_length

        start = end

    return result


def validate_prediction_directory(
    *,
    dataset: RegisteredValidationDataset,
    prediction_dir: Path,
) -> list[
    tuple[
        int,
        str,
        Path,
    ]
]:
    """Validate exact prediction coverage for the registered cohort."""

    prediction_dir = (
        prediction_dir
        .expanduser()
        .resolve()
    )

    if not prediction_dir.is_dir():
        raise ValidationScreeningError(
            "Prediction directory "
            f"does not exist:\n{prediction_dir}"
        )

    expected: list[
        tuple[
            int,
            str,
            Path,
        ]
    ] = []

    expected_names: set[str] = set()

    for offset in range(
        dataset.subject_count
    ):
        numeric_id = (
            dataset.first_numeric_id
            + offset
        )

        subject_name = format_subject_name(
            id_pattern=dataset.id_pattern,
            numeric_id=numeric_id,
        )

        path = (
            prediction_dir
            / prediction_filename(
                subject_name
            )
        )

        expected.append(
            (
                numeric_id,
                subject_name,
                path,
            )
        )

        expected_names.add(
            path.name
        )

    missing = [
        str(
            path
        )
        for (
            _numeric_id,
            _subject_name,
            path,
        ) in expected
        if not path.is_file()
    ]

    if missing:
        preview = "\n".join(
            missing[
                :10
            ]
        )

        raise ValidationScreeningError(
            "Missing expected nnU-Net "
            f"prediction(s): {len(missing)}\n"
            f"{preview}"
        )

    observed = {
        path.name
        for path in prediction_dir.glob(
            "*.nii.gz"
        )
        if path.is_file()
    }

    unexpected = sorted(
        observed
        - expected_names
    )

    if unexpected:
        preview = "\n".join(
            unexpected[
                :10
            ]
        )

        raise ValidationScreeningError(
            "Unexpected prediction file(s) "
            f"detected: {len(unexpected)}\n"
            f"{preview}"
        )

    return expected


def screen_subject(
    *,
    numeric_id: int,
    subject_name: str,
    prediction_path: Path,
    dataset: RegisteredValidationDataset,
    reference_modality: str,
) -> list[dict[str, Any]]:
    """Create slice-level screening rows for one subject."""

    prediction, prediction_affine = (
        load_nifti_array(
            prediction_path
        )
    )

    prediction = (
        validate_binary_prediction(
            prediction,
            path=prediction_path,
        )
    )

    reference_path = (
        resolve_reference_modality(
            dataset=dataset,
            subject_name=subject_name,
            modality=reference_modality,
        )
    )

    reference, reference_affine = (
        load_nifti_array(
            reference_path
        )
    )

    if (
        prediction.shape
        != reference.shape
    ):
        raise ValidationScreeningError(
            "Prediction/reference shape "
            f"mismatch for {subject_name}.\n"
            f"Prediction: {prediction.shape}\n"
            f"Reference : {reference.shape}"
        )

    if not np.allclose(
        prediction_affine,
        reference_affine,
        rtol=1e-5,
        atol=1e-5,
    ):
        raise ValidationScreeningError(
            "Prediction/reference affine "
            f"mismatch for {subject_name}."
        )

    # Historical 2-D composition uses axial
    # slices indexed along the third NIfTI axis.
    slice_count = int(
        prediction.shape[
            2
        ]
    )

    tumor_pixels = np.asarray(
        [
            int(
                prediction[
                    :,
                    :,
                    slice_index,
                ].sum(
                    dtype=np.int64
                )
            )
            for slice_index in range(
                slice_count
            )
        ],
        dtype=np.int64,
    )

    slice_area = int(
        prediction.shape[
            0
        ]
        * prediction.shape[
            1
        ]
    )

    tumor_fraction = (
        tumor_pixels.astype(
            np.float64
        )
        / float(
            slice_area
        )
    )

    nearest_distance = (
        nearest_positive_slice_distances(
            tumor_pixels
        )
    )

    zero_run_length = (
        contiguous_zero_run_lengths(
            tumor_pixels
        )
    )

    subject_has_predicted_tumor = bool(
        np.any(
            tumor_pixels > 0
        )
    )

    rows: list[
        dict[str, Any]
    ] = []

    for slice_index in range(
        slice_count
    ):
        pixel_count = int(
            tumor_pixels[
                slice_index
            ]
        )

        tumor_present = (
            pixel_count > 0
        )

        tumor_free = (
            pixel_count == 0
        )

        distance_value = int(
            nearest_distance[
                slice_index
            ]
        )

        rows.append(
            {
                "subject":
                    subject_name,

                "subject_numeric_id":
                    numeric_id,

                "slice_index":
                    slice_index,

                "predicted_tumor_pixels":
                    pixel_count,

                "predicted_tumor_fraction":
                    float(
                        tumor_fraction[
                            slice_index
                        ]
                    ),

                "predicted_tumor_present":
                    tumor_present,

                "tumor_free_candidate":
                    tumor_free,

                "nearest_tumor_slice_distance":
                    (
                        distance_value
                        if distance_value
                        >= 0
                        else np.nan
                    ),

                "contiguous_tumor_free_run_length":
                    int(
                        zero_run_length[
                            slice_index
                        ]
                    ),

                "subject_has_predicted_tumor":
                    subject_has_predicted_tumor,

                "prediction_path":
                    str(
                        prediction_path
                    ),

                "reference_modality":
                    reference_modality,

                "reference_image_path":
                    str(
                        reference_path
                    ),
            }
        )

    return rows


def refuse_existing_outputs(
    *,
    output_dir: Path,
) -> None:
    """Refuse to overwrite existing screening outputs."""

    csv_path = (
        output_dir
        / OUTPUT_CSV_NAME
    )

    summary_path = (
        output_dir
        / OUTPUT_SUMMARY_NAME
    )

    existing = [
        path
        for path in (
            csv_path,
            summary_path,
        )
        if path.exists()
    ]

    if existing:
        raise ValidationScreeningError(
            "Refusing to overwrite "
            "existing output(s):\n"
            + "\n".join(
                str(
                    path
                )
                for path in existing
            )
        )


def build_summary(
    *,
    table: pd.DataFrame,
    dataset: RegisteredValidationDataset,
    prediction_dir: Path,
    reference_modality: str,
    script_path: Path,
) -> dict[str, Any]:
    """Build screening provenance and descriptive summary."""

    subject_candidate_counts = (
        table.groupby(
            "subject"
        )[
            "tumor_free_candidate"
        ]
        .sum()
        .astype(
            int
        )
    )

    tumor_free_count = int(
        table[
            "tumor_free_candidate"
        ].sum()
    )

    tumor_positive_count = int(
        table[
            "predicted_tumor_present"
        ].sum()
    )

    subjects_with_candidates = int(
        (
            subject_candidate_counts
            > 0
        ).sum()
    )

    subjects_with_predicted_tumor = int(
        table.loc[
            :,
            [
                "subject",
                "subject_has_predicted_tumor",
            ],
        ]
        .drop_duplicates(
            subset=[
                "subject"
            ]
        )[
            "subject_has_predicted_tumor"
        ]
        .sum()
    )

    return {
        "screening_definition":
            (
                "tumor_free_candidate = "
                "predicted_tumor_pixels == 0"
            ),

        "screening_definition_source":
            (
                "Mirrors historical composition "
                "candidate discovery using zero "
                "whole-tumor pixels."
            ),

        "registered_validation_subjects":
            dataset.subject_count,

        "subjects_screened":
            int(
                table[
                    "subject"
                ].nunique()
            ),

        "subjects_with_predicted_tumor":
            subjects_with_predicted_tumor,

        "subjects_with_tumor_free_candidates":
            subjects_with_candidates,

        "total_slices_screened":
            int(
                len(
                    table
                )
            ),

        "tumor_positive_slices":
            tumor_positive_count,

        "tumor_free_candidate_slices":
            tumor_free_count,

        "tumor_free_candidate_fraction":
            float(
                tumor_free_count
                / len(
                    table
                )
            ),

        "tumor_free_candidates_per_subject":
            {
                "minimum":
                    int(
                        subject_candidate_counts.min()
                    ),

                "median":
                    float(
                        subject_candidate_counts.median()
                    ),

                "maximum":
                    int(
                        subject_candidate_counts.max()
                    ),

                "mean":
                    float(
                        subject_candidate_counts.mean()
                    ),
            },

        "reference_modality":
            reference_modality,

        "validation_dataset_specification":
            str(
                dataset.specification_path
            ),

        "validation_dataset_root":
            str(
                dataset.dataset_root
            ),

        "prediction_directory":
            str(
                prediction_dir
            ),

        "prediction_file_count":
            dataset.subject_count,

        "git_commit":
            resolve_git_commit(),

        "git_worktree_clean":
            resolve_git_worktree_clean(),

        "screening_script":
            str(
                script_path
            ),

        "screening_script_sha256":
            sha256_file(
                script_path
            ),

        "screened_at_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),
    }


def print_validation_banner(
    *,
    dataset: RegisteredValidationDataset,
    prediction_dir: Path,
    reference_modality: str,
    validate_only: bool,
) -> None:
    """Print screening configuration."""

    print()
    print(
        "=" * 78
    )

    print(
        "BraTS 2020 VALIDATION SLICE SCREENING"
    )

    print(
        "=" * 78
    )

    print(
        f"Validation dataset       : "
        f"{dataset.specification_path}"
    )

    print(
        f"Validation root          : "
        f"{dataset.dataset_root}"
    )

    print(
        f"Registered subjects      : "
        f"{dataset.subject_count}"
    )

    print(
        f"Prediction directory     : "
        f"{prediction_dir}"
    )

    print(
        f"Reference modality       : "
        f"{reference_modality}"
    )

    print(
        "Tumor-free rule          : "
        "predicted tumor pixels == 0"
    )

    print(
        f"Validate only            : "
        f"{validate_only}"
    )

    print(
        "=" * 78
    )


def main() -> None:
    """Run validation slice screening."""

    args = parse_args()

    folders_config = load_folders_config(
        args.folders_file
    )

    validation_dataset_path = resolve_path(
        key="yaml_validation_dataset_path",
        cli_value=args.validation_dataset,
        config=folders_config,
        selector=None,
    )

    nnunet_run_root = None

    if (
        args.prediction_dir is None
        or args.output_dir is None
    ):
        nnunet_run_root = resolve_path(
            key="nnunet_run_root",
            cli_value=None,
            config=folders_config,
            selector=None,
        ).expanduser().resolve()

    if args.prediction_dir is not None:
        prediction_dir = (
            args.prediction_dir
            .expanduser()
            .resolve()
        )
    else:
        prediction_dir = (
            nnunet_run_root
            / "validation_predictions_l40s_normal_q"
        )

    if args.output_dir is not None:
        output_dir = (
            args.output_dir
            .expanduser()
            .resolve()
        )
    else:
        output_dir = (
            nnunet_run_root
            / "validation_slice_screening"
        )

    save_folders_config(
        args.folders_file,
        folders_config,
    )

    dataset = load_validation_specification(
        validation_dataset_path
    )

    reference_modality = (
        str(
            args.reference_modality
        )
        .strip()
        .lower()
    )

    if not reference_modality:
        raise ValidationScreeningError(
            "`--reference-modality` "
            "must not be empty."
        )

    print_validation_banner(
        dataset=dataset,
        prediction_dir=prediction_dir,
        reference_modality=reference_modality,
        validate_only=args.validate_only,
    )

    expected_predictions = (
        validate_prediction_directory(
            dataset=dataset,
            prediction_dir=prediction_dir,
        )
    )

    print()
    print(
        "Prediction coverage      : PASS"
    )

    print(
        f"Prediction files         : "
        f"{len(expected_predictions)}"
    )

    rows: list[
        dict[str, Any]
    ] = []

    for index, (
        numeric_id,
        subject_name,
        prediction_path,
    ) in enumerate(
        expected_predictions,
        start=1,
    ):
        subject_rows = (
            screen_subject(
                numeric_id=numeric_id,
                subject_name=subject_name,
                prediction_path=prediction_path,
                dataset=dataset,
                reference_modality=reference_modality,
            )
        )

        rows.extend(
            subject_rows
        )

        if (
            index % 25 == 0
            or index
            == len(
                expected_predictions
            )
        ):
            print(
                f"  Screened subjects: "
                f"{index:,} / "
                f"{len(expected_predictions):,}",
                flush=True,
            )

    if not rows:
        raise ValidationScreeningError(
            "No validation slices "
            "were screened."
        )

    table = pd.DataFrame(
        rows
    )

    expected_subject_count = (
        dataset.subject_count
    )

    observed_subject_count = int(
        table[
            "subject"
        ].nunique()
    )

    if (
        observed_subject_count
        != expected_subject_count
    ):
        raise ValidationScreeningError(
            "Unexpected number of "
            "screened subjects.\n"
            f"Expected: "
            f"{expected_subject_count}\n"
            f"Observed: "
            f"{observed_subject_count}"
        )

    candidate_count = int(
        table[
            "tumor_free_candidate"
        ].sum()
    )

    positive_count = int(
        table[
            "predicted_tumor_present"
        ].sum()
    )

    print()
    print(
        f"Total slices screened    : "
        f"{len(table):,}"
    )

    print(
        f"Tumor-positive slices    : "
        f"{positive_count:,}"
    )

    print(
        f"Tumor-free candidates    : "
        f"{candidate_count:,}"
    )

    if args.validate_only:
        print()
        print(
            "=" * 78
        )

        print(
            "VALIDATION ONLY: PASS"
        )

        print(
            "Prediction coverage, binary masks, "
            "geometry, and slice-level screening "
            "logic were validated."
        )

        print(
            "No screening outputs were written."
        )

        print(
            "=" * 78
        )

        return

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    refuse_existing_outputs(
        output_dir=output_dir
    )

    csv_path = (
        output_dir
        / OUTPUT_CSV_NAME
    )

    summary_path = (
        output_dir
        / OUTPUT_SUMMARY_NAME
    )

    table = table.sort_values(
        [
            "subject_numeric_id",
            "slice_index",
        ],
        kind="stable",
    ).reset_index(
        drop=True
    )

    table.to_csv(
        csv_path,
        index=False,
    )

    script_path = Path(
        __file__
    ).expanduser().resolve()

    summary = build_summary(
        table=table,
        dataset=dataset,
        prediction_dir=prediction_dir,
        reference_modality=reference_modality,
        script_path=script_path,
    )

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
            sort_keys=True,
        )

        file.write(
            "\n"
        )

    print()
    print(
        "=" * 78
    )

    print(
        "VALIDATION SLICE SCREENING: PASS"
    )

    print(
        "=" * 78
    )

    print(
        f"Screening table          : "
        f"{csv_path}"
    )

    print(
        f"Screening summary        : "
        f"{summary_path}"
    )

    print(
        f"Tumor-free candidates    : "
        f"{candidate_count:,}"
    )

    print(
        "=" * 78
    )


if __name__ == "__main__":
    main()