"""
Registered BraTS 2020 validation-dataset loading utilities.

The official BraTS 2020 validation release contains four MRI modalities but
does not contain segmentation masks. These utilities consume the previously
validated ``validation_dataset.yaml`` specification and do not repeat the
full raw-dataset registration scan.

Validation slices are converted to the same image representation used by the
historical training H5 pipeline:

1. load the registered NIfTI modality as float64,
2. extract one axial slice,
3. apply historical all-pixel per-slice z-score standardization,
4. apply the notebook-equivalent percentile normalization,
5. return a float32 tensor with shape ``[1, H, W]``.

No tumor-free status is inferred by this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import torch
import yaml

from .preprocessing import (
    normalize_image_channel,
    standardize_nifti_slice,
)


SUPPORTED_SCHEMA_VERSION = 1
SUPPORTED_DATASET_ID = "brats2020_validation"

EXPECTED_VOLUME_SHAPE = (
    240,
    240,
    155,
)

VALIDATION_MODALITIES = (
    "flair",
    "t1",
    "t1ce",
    "t2",
)


class ValidationDatasetError(
    ValueError
):
    """Raised when the registered validation dataset is incompatible."""


@dataclass(
    frozen=True,
    slots=True,
)
class RegisteredValidationDataset:
    """Validated metadata needed for downstream BraTS validation loading."""

    specification_path: Path
    raw_data_root: Path

    subject_count: int
    first_numeric_id: int
    last_numeric_id: int
    id_pattern: str

    volume_shape: tuple[
        int,
        int,
        int,
    ]

    slice_axis: int
    slices_per_subject: int

    modality_files: dict[
        str,
        str,
    ]


@dataclass(
    frozen=True,
    slots=True,
)
class ValidationSlice:
    """One preprocessed external BraTS validation slice."""

    image: torch.Tensor

    subject_numeric_id: int
    subject_name: str
    slice_index: int
    modality: str
    source_path: Path


def _require_mapping(
    parent: dict[str, Any],
    key: str,
) -> dict[str, Any]:
    """Return one required mapping from a dataset specification."""

    value = parent.get(
        key
    )

    if not isinstance(
        value,
        dict,
    ):
        raise ValidationDatasetError(
            "validation_dataset.yaml is missing the required "
            f"mapping: {key}"
        )

    return value


def load_validation_dataset_specification(
    path: str | Path,
) -> RegisteredValidationDataset:
    """
    Load and validate a registered BraTS 2020 validation specification.

    This validates the specification and required registered paths only.
    It deliberately does not rescan all 125 subjects or 500 NIfTI files.
    """

    specification_path = (
        Path(
            path
        )
        .expanduser()
        .resolve()
    )

    if not specification_path.is_file():
        raise FileNotFoundError(
            "Validation dataset specification not found:\n"
            f"{specification_path}"
        )

    try:
        with specification_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            specification = yaml.safe_load(
                file
            )

    except (
        OSError,
        yaml.YAMLError,
    ) as exc:
        raise ValidationDatasetError(
            "Validation dataset specification could not be read.\n\n"
            f"File:\n{specification_path}\n\n"
            f"Error:\n{exc}"
        ) from exc

    if not isinstance(
        specification,
        dict,
    ):
        raise ValidationDatasetError(
            "validation_dataset.yaml must contain a top-level mapping."
        )

    schema_version = specification.get(
        "schema_version"
    )

    if schema_version != SUPPORTED_SCHEMA_VERSION:
        raise ValidationDatasetError(
            "Unsupported validation dataset schema version.\n\n"
            f"Expected: {SUPPORTED_SCHEMA_VERSION}\n"
            f"Observed: {schema_version}"
        )

    dataset = _require_mapping(
        specification,
        "dataset",
    )

    subjects = _require_mapping(
        specification,
        "subjects",
    )

    volumes = _require_mapping(
        specification,
        "volumes",
    )

    modalities = _require_mapping(
        specification,
        "modalities",
    )

    segmentation = _require_mapping(
        specification,
        "segmentation",
    )

    validation = _require_mapping(
        specification,
        "validation",
    )

    if dataset.get(
        "id"
    ) != SUPPORTED_DATASET_ID:
        raise ValidationDatasetError(
            "This loader supports only the registered BraTS 2020 "
            "validation dataset.\n\n"
            f"Expected dataset id: {SUPPORTED_DATASET_ID}\n"
            f"Observed dataset id: {dataset.get('id')}"
        )

    if validation.get(
        "status"
    ) != "passed":
        raise ValidationDatasetError(
            "validation_dataset.yaml does not record successful "
            "dataset registration."
        )

    if segmentation.get(
        "available"
    ) is not False:
        raise ValidationDatasetError(
            "The registered BraTS validation contract must record "
            "segmentation.available=false."
        )

    raw_data_root_value = dataset.get(
        "raw_data_root"
    )

    if not isinstance(
        raw_data_root_value,
        str,
    ) or not raw_data_root_value:
        raise ValidationDatasetError(
            "validation_dataset.yaml does not contain a valid "
            "dataset.raw_data_root."
        )

    raw_data_root = (
        Path(
            raw_data_root_value
        )
        .expanduser()
        .resolve()
    )

    if not raw_data_root.is_dir():
        raise FileNotFoundError(
            "Registered validation dataset root is not accessible:\n"
            f"{raw_data_root}"
        )

    subject_count = int(
        subjects.get(
            "count"
        )
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
        raise ValidationDatasetError(
            "subjects.id_pattern must be a non-empty string."
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
        raise ValidationDatasetError(
            "Registered validation volume shape is incompatible.\n\n"
            f"Expected: {EXPECTED_VOLUME_SHAPE}\n"
            f"Observed: {volume_shape}"
        )

    slice_axis = int(
        volumes.get(
            "slice_axis"
        )
    )

    slices_per_subject = int(
        volumes.get(
            "slices_per_subject"
        )
    )

    if slice_axis != 2:
        raise ValidationDatasetError(
            "The external validation loader expects slice_axis=2."
        )

    if slices_per_subject != EXPECTED_VOLUME_SHAPE[
        2
    ]:
        raise ValidationDatasetError(
            "Unexpected registered slices-per-subject value."
        )

    modality_files = modalities.get(
        "files"
    )

    if not isinstance(
        modality_files,
        dict,
    ):
        raise ValidationDatasetError(
            "modalities.files must be a mapping."
        )

    normalized_modality_files = {}

    for modality in VALIDATION_MODALITIES:
        pattern = modality_files.get(
            modality
        )

        if not isinstance(
            pattern,
            str,
        ) or not pattern:
            raise ValidationDatasetError(
                "Missing registered file pattern for modality "
                f"'{modality}'."
            )

        normalized_modality_files[
            modality
        ] = pattern

    return RegisteredValidationDataset(
        specification_path=specification_path,
        raw_data_root=raw_data_root,
        subject_count=subject_count,
        first_numeric_id=first_numeric_id,
        last_numeric_id=last_numeric_id,
        id_pattern=id_pattern,
        volume_shape=EXPECTED_VOLUME_SHAPE,
        slice_axis=slice_axis,
        slices_per_subject=slices_per_subject,
        modality_files=normalized_modality_files,
    )


def validation_subject_name(
    dataset: RegisteredValidationDataset,
    subject_numeric_id: int,
) -> str:
    """Return the canonical registered validation subject name."""

    if (
        isinstance(
            subject_numeric_id,
            bool,
        )
        or not isinstance(
            subject_numeric_id,
            int,
        )
    ):
        raise TypeError(
            "subject_numeric_id must be an integer."
        )

    if not (
        dataset.first_numeric_id
        <= subject_numeric_id
        <= dataset.last_numeric_id
    ):
        raise ValidationDatasetError(
            "Validation subject numeric id is outside the registered "
            "range.\n\n"
            f"Observed: {subject_numeric_id}\n"
            f"Allowed: {dataset.first_numeric_id}-"
            f"{dataset.last_numeric_id}"
        )

    try:
        return dataset.id_pattern.format(
            id=subject_numeric_id
        )

    except (
        KeyError,
        ValueError,
    ) as exc:
        raise ValidationDatasetError(
            "Registered validation subject id pattern could not be "
            "formatted."
        ) from exc


def resolve_validation_modality_path(
    dataset: RegisteredValidationDataset,
    *,
    subject_numeric_id: int,
    modality: str,
) -> Path:
    """Resolve one registered validation NIfTI modality path."""

    if modality not in VALIDATION_MODALITIES:
        raise ValidationDatasetError(
            "Unsupported validation modality.\n\n"
            f"Observed: {modality}\n"
            f"Supported: {VALIDATION_MODALITIES}"
        )

    subject_name = validation_subject_name(
        dataset,
        subject_numeric_id,
    )

    subject_dir = (
        dataset.raw_data_root
        / subject_name
    )

    if not subject_dir.is_dir():
        raise FileNotFoundError(
            "Registered validation subject directory is not accessible:\n"
            f"{subject_dir}"
        )

    pattern = dataset.modality_files[
        modality
    ]

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
            "Registered validation modality is not accessible.\n\n"
            f"Subject: {subject_name}\n"
            f"Modality: {modality}\n"
            f"Expected file:\n{path}"
        )

    return path


def load_validation_nifti_volume(
    path: str | Path,
) -> np.ndarray:
    """Load one registered validation NIfTI volume as float64."""

    resolved = (
        Path(
            path
        )
        .expanduser()
        .resolve()
    )

    if not resolved.is_file():
        raise FileNotFoundError(
            f"Validation NIfTI file not found:\n{resolved}"
        )

    try:
        image = nib.load(
            str(
                resolved
            )
        )

        volume = image.get_fdata(
            dtype=np.float64
        )

    except Exception as exc:
        raise ValidationDatasetError(
            "Validation NIfTI volume could not be loaded.\n\n"
            f"File:\n{resolved}\n\n"
            f"Error:\n{exc}"
        ) from exc

    if volume.shape != EXPECTED_VOLUME_SHAPE:
        raise ValidationDatasetError(
            "Validation NIfTI volume has an unexpected shape.\n\n"
            f"File:\n{resolved}\n"
            f"Expected: {EXPECTED_VOLUME_SHAPE}\n"
            f"Observed: {volume.shape}"
        )

    if not np.isfinite(
        volume
    ).all():
        raise ValidationDatasetError(
            "Validation NIfTI volume contains non-finite values."
        )

    return volume


def load_validation_slice(
    dataset: RegisteredValidationDataset,
    *,
    subject_numeric_id: int,
    slice_index: int,
    modality: str = "flair",
) -> ValidationSlice:
    """
    Load one validation slice using the training-compatible image pipeline.

    No segmentation, tumor mask, or tumor-free classification is produced.
    """

    if (
        isinstance(
            slice_index,
            bool,
        )
        or not isinstance(
            slice_index,
            int,
        )
        or not (
            0
            <= slice_index
            < dataset.slices_per_subject
        )
    ):
        raise ValidationDatasetError(
            "slice_index must be an integer in the registered slice range "
            f"0-{dataset.slices_per_subject - 1}."
        )

    source_path = resolve_validation_modality_path(
        dataset,
        subject_numeric_id=subject_numeric_id,
        modality=modality,
    )

    volume = load_validation_nifti_volume(
        source_path
    )

    raw_slice = volume[
        :,
        :,
        slice_index,
    ]

    standardized = standardize_nifti_slice(
        raw_slice
    )

    normalized = normalize_image_channel(
        standardized
    )

    image = torch.from_numpy(
        normalized[
            None,
            :,
            :,
        ]
    ).float()

    if tuple(
        image.shape
    ) != (
        1,
        240,
        240,
    ):
        raise RuntimeError(
            "External validation slice has an unexpected tensor shape."
        )

    if not torch.isfinite(
        image
    ).all():
        raise RuntimeError(
            "External validation slice contains non-finite tensor values."
        )

    subject_name = validation_subject_name(
        dataset,
        subject_numeric_id,
    )

    return ValidationSlice(
        image=image,
        subject_numeric_id=subject_numeric_id,
        subject_name=subject_name,
        slice_index=slice_index,
        modality=modality,
        source_path=source_path,
    )


__all__ = [
    "RegisteredValidationDataset",
    "ValidationDatasetError",
    "ValidationSlice",
    "VALIDATION_MODALITIES",
    "load_validation_dataset_specification",
    "load_validation_nifti_volume",
    "load_validation_slice",
    "resolve_validation_modality_path",
    "validation_subject_name",
]
