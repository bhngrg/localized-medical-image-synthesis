"""
External evaluation-manifest utilities.

The external BR-LoRA evaluation workflow consumes a fixed case manifest rather
than discovering or selecting evaluation cases internally. This ensures that
all trained models are evaluated on exactly the same external base images and
donor lesion cases.

The manifest intentionally records only the information required to reconstruct
one evaluation case:

- unique case identifier,
- external BraTS validation subject,
- external axial slice index,
- external MRI modality, and
- donor H5 path from the labeled BraTS training dataset.

Tumor-free screening metadata may be added later without changing the required
evaluation contract.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


REQUIRED_COLUMNS = (
    "case_id",
    "external_subject_numeric_id",
    "external_slice_index",
    "external_modality",
    "donor_h5_path",
)

SUPPORTED_EXTERNAL_MODALITIES = (
    "flair",
    "t1",
    "t1ce",
    "t2",
)

MIN_EXTERNAL_SUBJECT_ID = 1
MAX_EXTERNAL_SUBJECT_ID = 125

MIN_EXTERNAL_SLICE_INDEX = 0
MAX_EXTERNAL_SLICE_INDEX = 154


class ExternalManifestError(
    ValueError
):
    """Raised when an external evaluation manifest is invalid."""


@dataclass(
    frozen=True,
    slots=True,
)
class ExternalEvaluationCase:
    """One validated external evaluation case."""

    case_id: str
    external_subject_numeric_id: int
    external_slice_index: int
    external_modality: str
    donor_h5_path: Path


def _require_nonempty_string(
    value: object,
    *,
    name: str,
) -> str:
    """Return one required non-empty string."""

    if not isinstance(
        value,
        str,
    ):
        raise ExternalManifestError(
            f"{name} must be a string."
        )

    normalized = value.strip()

    if not normalized:
        raise ExternalManifestError(
            f"{name} must not be empty."
        )

    return normalized


def _parse_integer(
    value: object,
    *,
    name: str,
) -> int:
    """Parse one manifest integer field strictly."""

    text = _require_nonempty_string(
        value,
        name=name,
    )

    try:
        parsed = int(
            text
        )

    except ValueError as exc:
        raise ExternalManifestError(
            f"{name} must contain an integer; received {text!r}."
        ) from exc

    if str(
        parsed
    ) != text:
        raise ExternalManifestError(
            f"{name} must use canonical integer notation; "
            f"received {text!r}."
        )

    return parsed


def _validate_header(
    fieldnames: list[str] | None,
) -> None:
    """Validate required external-manifest columns."""

    if fieldnames is None:
        raise ExternalManifestError(
            "External evaluation manifest is missing a CSV header."
        )

    missing = [
        column
        for column in REQUIRED_COLUMNS
        if column not in fieldnames
    ]

    if missing:
        raise ExternalManifestError(
            "External evaluation manifest is missing required columns:\n"
            + "\n".join(
                missing
            )
        )


def _parse_case(
    row: dict[str, str],
    *,
    row_number: int,
) -> ExternalEvaluationCase:
    """Parse and validate one external evaluation-manifest row."""

    case_id = _require_nonempty_string(
        row.get(
            "case_id"
        ),
        name=f"row {row_number} case_id",
    )

    subject_numeric_id = _parse_integer(
        row.get(
            "external_subject_numeric_id"
        ),
        name=(
            f"row {row_number} "
            "external_subject_numeric_id"
        ),
    )

    if not (
        MIN_EXTERNAL_SUBJECT_ID
        <= subject_numeric_id
        <= MAX_EXTERNAL_SUBJECT_ID
    ):
        raise ExternalManifestError(
            f"row {row_number} external_subject_numeric_id is outside "
            "the BraTS 2020 validation range.\n"
            f"Observed: {subject_numeric_id}\n"
            f"Allowed: {MIN_EXTERNAL_SUBJECT_ID}-"
            f"{MAX_EXTERNAL_SUBJECT_ID}"
        )

    slice_index = _parse_integer(
        row.get(
            "external_slice_index"
        ),
        name=(
            f"row {row_number} "
            "external_slice_index"
        ),
    )

    if not (
        MIN_EXTERNAL_SLICE_INDEX
        <= slice_index
        <= MAX_EXTERNAL_SLICE_INDEX
    ):
        raise ExternalManifestError(
            f"row {row_number} external_slice_index is outside "
            "the BraTS 2020 validation range.\n"
            f"Observed: {slice_index}\n"
            f"Allowed: {MIN_EXTERNAL_SLICE_INDEX}-"
            f"{MAX_EXTERNAL_SLICE_INDEX}"
        )

    modality = _require_nonempty_string(
        row.get(
            "external_modality"
        ),
        name=(
            f"row {row_number} "
            "external_modality"
        ),
    ).lower()

    if modality not in SUPPORTED_EXTERNAL_MODALITIES:
        raise ExternalManifestError(
            f"row {row_number} contains an unsupported external modality.\n"
            f"Observed: {modality}\n"
            f"Supported: {SUPPORTED_EXTERNAL_MODALITIES}"
        )

    donor_h5_value = _require_nonempty_string(
        row.get(
            "donor_h5_path"
        ),
        name=(
            f"row {row_number} "
            "donor_h5_path"
        ),
    )

    donor_h5_path = (
        Path(
            donor_h5_value
        )
        .expanduser()
        .resolve()
    )

    if not donor_h5_path.is_file():
        raise FileNotFoundError(
            "External evaluation donor H5 file is not accessible.\n\n"
            f"Row: {row_number}\n"
            f"Case: {case_id}\n"
            f"Path:\n{donor_h5_path}"
        )

    if donor_h5_path.suffix.lower() not in (
        ".h5",
        ".hdf5",
    ):
        raise ExternalManifestError(
            f"row {row_number} donor_h5_path must reference an H5 file.\n"
            f"Observed:\n{donor_h5_path}"
        )

    return ExternalEvaluationCase(
        case_id=case_id,
        external_subject_numeric_id=subject_numeric_id,
        external_slice_index=slice_index,
        external_modality=modality,
        donor_h5_path=donor_h5_path,
    )


def load_external_evaluation_manifest(
    path: str | Path,
) -> tuple[
    ExternalEvaluationCase,
    ...,
]:
    """
    Load and validate one fixed external evaluation manifest.

    Existing paths are resolved exactly as written. No automatic donor
    discovery, path correction, or case selection is performed.
    """

    manifest_path = (
        Path(
            path
        )
        .expanduser()
        .resolve()
    )

    if not manifest_path.is_file():
        raise FileNotFoundError(
            "External evaluation manifest not found:\n"
            f"{manifest_path}"
        )

    cases = []

    seen_case_ids = set()

    try:
        with manifest_path.open(
            "r",
            encoding="utf-8",
            newline="",
        ) as file:
            reader = csv.DictReader(
                file
            )

            _validate_header(
                reader.fieldnames
            )

            for row_number, row in enumerate(
                reader,
                start=2,
            ):
                case = _parse_case(
                    row,
                    row_number=row_number,
                )

                if case.case_id in seen_case_ids:
                    raise ExternalManifestError(
                        "External evaluation manifest contains duplicate "
                        f"case_id {case.case_id!r}."
                    )

                seen_case_ids.add(
                    case.case_id
                )

                cases.append(
                    case
                )

    except csv.Error as exc:
        raise ExternalManifestError(
            "External evaluation manifest could not be parsed as CSV.\n\n"
            f"File:\n{manifest_path}\n\n"
            f"Error:\n{exc}"
        ) from exc

    if not cases:
        raise ExternalManifestError(
            "External evaluation manifest contains no evaluation cases."
        )

    return tuple(
        cases
    )


__all__ = [
    "ExternalEvaluationCase",
    "ExternalManifestError",
    "REQUIRED_COLUMNS",
    "SUPPORTED_EXTERNAL_MODALITIES",
    "load_external_evaluation_manifest",
]
