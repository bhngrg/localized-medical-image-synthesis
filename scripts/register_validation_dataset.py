#!/usr/bin/env python3

"""
Register the raw BraTS 2020 validation dataset.

Responsibilities
----------------
1. Ask the user to select the directory that directly contains the
   BraTS20_Validation_XXX subject directories.
2. Strictly validate the raw NIfTI validation dataset.
3. Ask the user where to save the dataset specification.
4. Write validation_dataset.yaml.

This script performs the full raw-dataset validation once. Downstream
workflows should use validation_dataset.yaml and must not repeat this
registration scan.

Important
---------
The official BraTS 2020 validation release contains the four MRI modalities
but does not include segmentation masks. This script treats the absence of
segmentation as part of the dataset contract rather than as an error.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import sys
import tkinter as tk
from tkinter import filedialog

import nibabel as nib
import numpy as np
import yaml


DATASET_ID = "brats2020_validation"
DATASET_NAME = "BraTS 2020 Validation Data"

EXPECTED_SUBJECT_COUNT = 125
EXPECTED_SUBJECT_IDS = tuple(
    range(1, EXPECTED_SUBJECT_COUNT + 1)
)

EXPECTED_VOLUME_SHAPE = (240, 240, 155)

MODALITIES = {
    "flair": "_flair.nii",
    "t1": "_t1.nii",
    "t1ce": "_t1ce.nii",
    "t2": "_t2.nii",
}


def create_tk_root() -> tk.Tk:
    """Create a hidden Tk root window."""
    root = tk.Tk()
    root.withdraw()
    root.update()
    return root


def select_dataset_directory() -> Path:
    """
    Ask the user to select the directory containing validation subject folders.

    Returns
    -------
    pathlib.Path
        Selected raw validation-dataset directory.

    Raises
    ------
    RuntimeError
        If the user cancels the selection.
    """
    root = create_tk_root()

    try:
        selected = filedialog.askdirectory(
            parent=root,
            title=(
                "Select the BraTS 2020 validation directory containing "
                "BraTS20_Validation_XXX folders"
            ),
        )
    finally:
        root.destroy()

    if not selected:
        raise RuntimeError(
            "No validation dataset directory was selected. "
            "Dataset registration was cancelled."
        )

    return Path(selected).expanduser().resolve()


def select_yaml_output_path() -> Path:
    """
    Ask the user where validation_dataset.yaml should be saved.

    Returns
    -------
    pathlib.Path
        Selected YAML output path.

    Raises
    ------
    RuntimeError
        If the user cancels the save dialog.
    """
    root = create_tk_root()

    try:
        selected = filedialog.asksaveasfilename(
            parent=root,
            title="Save BraTS validation dataset specification",
            defaultextension=".yaml",
            initialfile="validation_dataset.yaml",
            filetypes=[
                ("YAML files", "*.yaml"),
            ],
        )
    finally:
        root.destroy()

    if not selected:
        raise RuntimeError(
            "No output file was selected. "
            "The validation dataset was validated, but no dataset "
            "specification was written."
        )

    output_path = Path(selected).expanduser().resolve()

    if output_path.suffix.lower() != ".yaml":
        raise ValueError(
            "The validation dataset specification must use the .yaml "
            "extension.\n\n"
            f"Selected output:\n{output_path}"
        )

    return output_path


def expected_subject_name(
    subject_id: int,
) -> str:
    """Return the canonical BraTS 2020 validation subject name."""
    return f"BraTS20_Validation_{subject_id:03d}"


def validate_root_directory(
    data_root: Path,
) -> list[Path]:
    """
    Validate the selected raw-data root and return subject directories.

    The directory must directly contain all 125 expected
    BraTS20_Validation_XXX directories.

    No recursive discovery or automatic path correction is performed.
    """
    if not data_root.exists():
        raise ValueError(
            "The selected validation dataset path does not exist:\n\n"
            f"{data_root}"
        )

    if not data_root.is_dir():
        raise ValueError(
            "The selected validation dataset path is not a directory:\n\n"
            f"{data_root}"
        )

    expected_names = {
        expected_subject_name(subject_id)
        for subject_id in EXPECTED_SUBJECT_IDS
    }

    observed_subject_dirs = {
        path.name: path
        for path in data_root.iterdir()
        if (
            path.is_dir()
            and path.name.startswith("BraTS20_Validation_")
        )
    }

    observed_names = set(
        observed_subject_dirs
    )

    missing = sorted(
        expected_names - observed_names
    )

    unexpected = sorted(
        observed_names - expected_names
    )

    if missing or unexpected:
        message = [
            "The selected directory does not contain the expected "
            "BraTS 2020 validation subject structure.",
            "",
            f"Selected directory:\n{data_root}",
            "",
            f"Expected subjects: {EXPECTED_SUBJECT_COUNT}",
            f"Observed BraTS validation subject directories: "
            f"{len(observed_subject_dirs)}",
        ]

        if missing:
            message.extend(
                [
                    "",
                    f"Missing subject directories: {len(missing)}",
                    *[
                        f"  {name}"
                        for name in missing[:10]
                    ],
                ]
            )

            if len(missing) > 10:
                message.append(
                    f"  ... and {len(missing) - 10} more"
                )

        if unexpected:
            message.extend(
                [
                    "",
                    f"Unexpected BraTS validation subject directories: "
                    f"{len(unexpected)}",
                    *[
                        f"  {name}"
                        for name in unexpected[:10]
                    ],
                ]
            )

        message.extend(
            [
                "",
                "Please select the directory that directly contains:",
                "",
                "BraTS20_Validation_001/",
                "BraTS20_Validation_002/",
                "...",
                "BraTS20_Validation_125/",
                "",
                "The script does not search parent or child directories "
                "automatically.",
            ]
        )

        raise ValueError(
            "\n".join(message)
        )

    return [
        observed_subject_dirs[
            expected_subject_name(subject_id)
        ]
        for subject_id in EXPECTED_SUBJECT_IDS
    ]


def expected_subject_files(
    subject_dir: Path,
) -> dict[str, Path]:
    """Construct the exact expected MRI NIfTI paths for one subject."""
    subject_name = subject_dir.name

    return {
        modality: (
            subject_dir
            / f"{subject_name}{suffix}"
        )
        for modality, suffix in MODALITIES.items()
    }


def validate_subject_files(
    subject_dir: Path,
) -> dict[str, Path]:
    """Verify that all four required MRI NIfTI files exist."""
    expected_files = expected_subject_files(
        subject_dir
    )

    missing = [
        path
        for path in expected_files.values()
        if not path.is_file()
    ]

    if missing:
        raise ValueError(
            "Required validation MRI file(s) are missing.\n\n"
            f"Subject:\n{subject_dir.name}\n\n"
            "Missing:\n"
            + "\n".join(
                f"  {path.name}"
                for path in missing
            )
        )

    return expected_files


def load_nifti_header(
    path: Path,
) -> nib.spatialimages.SpatialImage:
    """Open a NIfTI file and provide an informative error on failure."""
    try:
        return nib.load(
            str(path)
        )

    except Exception as exc:
        raise ValueError(
            "A validation NIfTI file could not be opened.\n\n"
            f"File:\n{path}\n\n"
            f"Error:\n{exc}"
        ) from exc


def validate_mri_values(
    subject_dir: Path,
    modality_name: str,
    image: nib.spatialimages.SpatialImage,
    path: Path,
) -> None:
    """Verify that one MRI volume contains only finite values."""
    volume = np.asanyarray(
        image.dataobj
    )

    if not np.all(
        np.isfinite(volume)
    ):
        raise ValueError(
            "Validation MRI volume contains non-finite values.\n\n"
            f"Subject: {subject_dir.name}\n"
            f"Modality: {modality_name}\n"
            f"File:\n{path}"
        )


def validate_subject_nifti_structure(
    subject_dir: Path,
    subject_files: dict[str, Path],
) -> dict[str, str]:
    """
    Validate shape, finite values, and within-subject spatial alignment.

    Returns
    -------
    dict[str, str]
        Observed dtype for each modality.
    """
    images = {
        name: load_nifti_header(path)
        for name, path in subject_files.items()
    }

    shapes = {
        name: tuple(image.shape)
        for name, image in images.items()
    }

    for name, shape in shapes.items():
        if shape != EXPECTED_VOLUME_SHAPE:
            raise ValueError(
                "Unexpected validation NIfTI volume shape.\n\n"
                f"Subject: {subject_dir.name}\n"
                f"Modality: {name}\n"
                f"Expected: {EXPECTED_VOLUME_SHAPE}\n"
                f"Observed: {shape}\n"
                f"File:\n{subject_files[name]}"
            )

    reference_name = "flair"

    reference_affine = images[
        reference_name
    ].affine

    for name, image in images.items():
        if not np.allclose(
            image.affine,
            reference_affine,
        ):
            raise ValueError(
                "NIfTI affine mismatch within validation subject.\n\n"
                f"Subject: {subject_dir.name}\n"
                f"Reference: {reference_name}\n"
                f"Mismatch: {name}\n\n"
                "The validation modalities are expected to be "
                "spatially aligned."
            )

    for modality_name in MODALITIES:
        validate_mri_values(
            subject_dir=subject_dir,
            modality_name=modality_name,
            image=images[modality_name],
            path=subject_files[modality_name],
        )

    return {
        name: str(
            image.get_data_dtype()
        )
        for name, image in images.items()
    }


def validate_dataset(
    data_root: Path,
) -> dict:
    """
    Perform the complete one-time raw validation-dataset scan.

    Returns
    -------
    dict
        Dataset-level information to be written to validation_dataset.yaml.
    """
    print(
        "\nValidating raw BraTS 2020 validation dataset...",
        flush=True,
    )

    print(
        "This is a one-time full validation.",
        flush=True,
    )

    subject_dirs = validate_root_directory(
        data_root
    )

    print(
        f"\nFound {len(subject_dirs):,} expected subject directories.",
        flush=True,
    )

    observed_modality_dtypes = {
        modality_name: set()
        for modality_name in MODALITIES
    }

    files_checked = 0

    for index, subject_dir in enumerate(
        subject_dirs,
        start=1,
    ):
        subject_files = validate_subject_files(
            subject_dir
        )

        dtypes = validate_subject_nifti_structure(
            subject_dir,
            subject_files,
        )

        for modality_name in MODALITIES:
            observed_modality_dtypes[
                modality_name
            ].add(
                dtypes[
                    modality_name
                ]
            )

        files_checked += len(
            subject_files
        )

        if (
            index % 25 == 0
            or index == len(subject_dirs)
        ):
            print(
                f"  Validated {index:,} / "
                f"{len(subject_dirs):,} subjects",
                flush=True,
            )

    expected_total_files = (
        EXPECTED_SUBJECT_COUNT
        * len(MODALITIES)
    )

    if files_checked != expected_total_files:
        raise ValueError(
            "Unexpected validated validation-file count.\n\n"
            f"Expected: {expected_total_files}\n"
            f"Observed: {files_checked}"
        )

    return {
        "subject_count": len(
            subject_dirs
        ),

        "files_checked": files_checked,

        "volume_shape": list(
            EXPECTED_VOLUME_SHAPE
        ),

        "modality_dtypes": {
            modality_name: sorted(
                dtype_values
            )
            for modality_name, dtype_values
            in observed_modality_dtypes.items()
        },
    }


def build_dataset_specification(
    data_root: Path,
    validation_result: dict,
) -> dict:
    """Construct validation_dataset.yaml."""
    return {
        "schema_version": 1,

        "dataset": {
            "id": DATASET_ID,
            "name": DATASET_NAME,
            "split": "validation",
            "format": "NIfTI",
            "raw_data_root": str(
                data_root
            ),
        },

        "subjects": {
            "count": validation_result[
                "subject_count"
            ],

            "id_pattern": (
                "BraTS20_Validation_{id:03d}"
            ),

            "first_numeric_id": 1,

            "last_numeric_id": (
                EXPECTED_SUBJECT_COUNT
            ),
        },

        "volumes": {
            "shape": validation_result[
                "volume_shape"
            ],

            "slice_axis": 2,

            "slices_per_subject": (
                EXPECTED_VOLUME_SHAPE[2]
            ),
        },

        "modalities": {
            "order": [
                "flair",
                "t1",
                "t1ce",
                "t2",
            ],

            "files": {
                "flair": "*_flair.nii",
                "t1": "*_t1.nii",
                "t1ce": "*_t1ce.nii",
                "t2": "*_t2.nii",
            },

            "observed_dtypes": (
                validation_result[
                    "modality_dtypes"
                ]
            ),
        },

        "segmentation": {
            "available": False,
        },

        "validation": {
            "status": "passed",

            "subjects_checked": (
                validation_result[
                    "subject_count"
                ]
            ),

            "files_checked": (
                validation_result[
                    "files_checked"
                ]
            ),

            "validated_at_utc": (
                datetime.now(
                    timezone.utc
                ).isoformat()
            ),
        },
    }


def write_dataset_yaml(
    output_path: Path,
    specification: dict,
) -> None:
    """Write the validated validation-dataset specification to YAML."""
    if output_path.exists():
        raise ValueError(
            "The selected output file already exists.\n\n"
            f"{output_path}\n\n"
            "No existing validation dataset specification will be "
            "overwritten implicitly. Select a new filename or remove "
            "the existing file explicitly."
        )

    try:
        with output_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            yaml.safe_dump(
                specification,
                file,
                sort_keys=False,
                default_flow_style=False,
                allow_unicode=True,
            )

    except OSError as exc:
        raise ValueError(
            "The validation dataset specification could not be written.\n\n"
            f"Output:\n{output_path}\n\n"
            f"Error:\n{exc}"
        ) from exc


def main() -> None:
    """Run interactive BraTS validation-dataset registration."""
    print("=" * 72)
    print(
        "BraTS 2020 Validation Dataset Registration"
    )
    print("=" * 72)

    try:
        data_root = (
            select_dataset_directory()
        )

        print(
            "\nSelected raw validation dataset directory:"
        )
        print(
            data_root
        )

        validation_result = (
            validate_dataset(
                data_root
            )
        )

        print(
            "\n" + "=" * 72
        )
        print(
            "VALIDATION DATASET CHECK PASSED"
        )
        print("=" * 72)

        print(
            f"Dataset directory : {data_root}"
        )

        print(
            f"Subjects          : "
            f"{validation_result['subject_count']:,}"
        )

        print(
            f"NIfTI files       : "
            f"{validation_result['files_checked']:,}"
        )

        print(
            f"Volume shape      : "
            f"{tuple(validation_result['volume_shape'])}"
        )

        print(
            f"MRI dtypes        : "
            f"{validation_result['modality_dtypes']}"
        )

        print(
            "Segmentation      : unavailable in this release"
        )

        print(
            "\nChoose where to save validation_dataset.yaml."
        )

        output_path = (
            select_yaml_output_path()
        )

        specification = (
            build_dataset_specification(
                data_root=data_root,
                validation_result=validation_result,
            )
        )

        write_dataset_yaml(
            output_path=output_path,
            specification=specification,
        )

    except (
        RuntimeError,
        ValueError,
        FileNotFoundError,
    ) as exc:
        print(
            "\n" + "=" * 72
        )
        print(
            "VALIDATION DATASET REGISTRATION FAILED"
        )
        print("=" * 72)
        print(
            exc
        )
        print(
            "\nNo validation dataset specification was written."
        )
        sys.exit(1)

    print(
        "\n" + "=" * 72
    )
    print(
        "VALIDATION DATASET REGISTRATION COMPLETE"
    )
    print("=" * 72)

    print(
        f"Dataset specification:\n{output_path}"
    )

    print(
        "\nThe raw BraTS validation dataset has been registered "
        "successfully."
    )

    print(
        "Downstream workflows should use this validation_dataset.yaml "
        "file instead of rescanning the raw validation dataset."
    )


if __name__ == "__main__":
    main()
