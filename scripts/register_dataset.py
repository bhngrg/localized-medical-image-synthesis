#!/usr/bin/env python3

"""
Register the raw BraTS 2020 training dataset.

Responsibilities
----------------
1. Ask the user to select the directory that directly contains the
   BraTS20_Training_XXX subject directories.
2. Strictly validate the raw NIfTI dataset.
3. Ask the user where to save the dataset specification.
4. Write dataset.yaml.

This script performs the full raw-dataset validation once. Downstream
workflows should use dataset.yaml and must not repeat this registration scan.
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
import argparse


DATASET_NAME = "BraTS 2020 Training Data"

EXPECTED_SUBJECT_COUNT = 369
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

SEGMENTATION_SUFFIX = "_seg.nii"

# Known filename anomaly in the BraTS 2020 training release.
SEGMENTATION_FILENAME_EXCEPTIONS = {
    "BraTS20_Training_355": "W39_1998.09.19_Segm.nii",
}

ALLOWED_SEGMENTATION_LABELS = {
    0,
    1,
    2,
    4,
}


def create_tk_root() -> tk.Tk:
    """Create a hidden Tk root window."""
    root = tk.Tk()
    root.withdraw()
    root.update()
    return root


def select_dataset_directory() -> Path:
    """
    Ask the user to select the directory containing the subject folders.

    Returns
    -------
    pathlib.Path
        Selected raw-dataset directory.

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
                "Select the BraTS 2020 training directory containing "
                "BraTS20_Training_XXX folders"
            ),
        )
    finally:
        root.destroy()

    if not selected:
        raise RuntimeError(
            "No dataset directory was selected. "
            "Dataset registration was cancelled."
        )

    return Path(selected).expanduser().resolve()


def select_yaml_output_path() -> Path:
    """
    Ask the user where dataset.yaml should be saved.

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
            title="Save BraTS dataset specification",
            defaultextension=".yaml",
            initialfile="dataset.yaml",
            filetypes=[
                ("YAML files", "*.yaml"),
            ],
        )
    finally:
        root.destroy()

    if not selected:
        raise RuntimeError(
            "No output file was selected. "
            "The dataset was validated, but no dataset specification "
            "was written."
        )

    output_path = Path(selected).expanduser().resolve()

    if output_path.suffix.lower() != ".yaml":
        raise ValueError(
            "The dataset specification must use the .yaml extension.\n\n"
            f"Selected output:\n{output_path}"
        )

    return output_path


def expected_subject_name(subject_id: int) -> str:
    """Return the canonical BraTS 2020 training subject name."""
    return f"BraTS20_Training_{subject_id:03d}"


def validate_root_directory(data_root: Path) -> list[Path]:
    """
    Validate the selected raw-data root and return subject directories.

    The directory must directly contain all 369 expected
    BraTS20_Training_XXX directories.

    No recursive discovery or automatic path correction is performed.
    """
    if not data_root.exists():
        raise ValueError(
            "The selected dataset path does not exist:\n\n"
            f"{data_root}"
        )

    if not data_root.is_dir():
        raise ValueError(
            "The selected dataset path is not a directory:\n\n"
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
            and path.name.startswith("BraTS20_Training_")
        )
    }

    observed_names = set(observed_subject_dirs)

    missing = sorted(expected_names - observed_names)
    unexpected = sorted(observed_names - expected_names)

    if missing or unexpected:
        message = [
            "The selected directory does not contain the expected "
            "BraTS 2020 training subject structure.",
            "",
            f"Selected directory:\n{data_root}",
            "",
            f"Expected subjects: {EXPECTED_SUBJECT_COUNT}",
            f"Observed BraTS subject directories: "
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
                    f"Unexpected BraTS subject directories: "
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
                "BraTS20_Training_001/",
                "BraTS20_Training_002/",
                "...",
                "BraTS20_Training_369/",
                "",
                "The script does not search parent or child directories "
                "automatically.",
            ]
        )

        raise ValueError("\n".join(message))

    return [
        observed_subject_dirs[
            expected_subject_name(subject_id)
        ]
        for subject_id in EXPECTED_SUBJECT_IDS
    ]


def expected_subject_files(
    subject_dir: Path,
) -> dict[str, Path]:
    """
    Construct the exact expected NIfTI paths for one subject.

    BraTS20_Training_355 is a documented exception in the BraTS 2020
    training release: its segmentation file is named
    'W39_1998.09.19_Segm.nii' instead of following the usual
    '<subject>_seg.nii' convention.
    """
    subject_name = subject_dir.name

    files = {
        modality: (
            subject_dir
            / f"{subject_name}{suffix}"
        )
        for modality, suffix in MODALITIES.items()
    }

    if subject_name in SEGMENTATION_FILENAME_EXCEPTIONS:
        segmentation_filename = (
            SEGMENTATION_FILENAME_EXCEPTIONS[
                subject_name
            ]
        )
    else:
        segmentation_filename = (
            f"{subject_name}{SEGMENTATION_SUFFIX}"
        )

    files["seg"] = (
        subject_dir
        / segmentation_filename
    )

    return files


def validate_subject_files(
    subject_dir: Path,
) -> dict[str, Path]:
    """Verify that all required NIfTI files exist for one subject."""
    expected_files = expected_subject_files(subject_dir)

    missing = [
        path
        for path in expected_files.values()
        if not path.is_file()
    ]

    if missing:
        raise ValueError(
            "Required NIfTI file(s) are missing.\n\n"
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
        return nib.load(str(path))

    except Exception as exc:
        raise ValueError(
            "A NIfTI file could not be opened.\n\n"
            f"File:\n{path}\n\n"
            f"Error:\n{exc}"
        ) from exc


def validate_mri_values(
    subject_dir: Path,
    modality_name: str,
    image: nib.spatialimages.SpatialImage,
    path: Path,
) -> None:
    """Verify that an MRI volume contains only finite values."""
    volume = np.asanyarray(image.dataobj)

    if not np.all(np.isfinite(volume)):
        raise ValueError(
            "MRI volume contains non-finite values.\n\n"
            f"Subject: {subject_dir.name}\n"
            f"Modality: {modality_name}\n"
            f"File:\n{path}"
        )


def validate_subject_nifti_structure(
    subject_dir: Path,
    subject_files: dict[str, Path],
) -> dict[str, str]:
    """
    Validate shape, dtype, finite MRI values, and within-subject alignment.

    Returns
    -------
    dict[str, str]
        Observed dtype for each modality and segmentation.
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
                "Unexpected NIfTI volume shape.\n\n"
                f"Subject: {subject_dir.name}\n"
                f"File type: {name}\n"
                f"Expected: {EXPECTED_VOLUME_SHAPE}\n"
                f"Observed: {shape}\n"
                f"File:\n{subject_files[name]}"
            )

    reference_name = "flair"
    reference_affine = images[reference_name].affine

    for name, image in images.items():
        if not np.allclose(
            image.affine,
            reference_affine,
        ):
            raise ValueError(
                "NIfTI affine mismatch within subject.\n\n"
                f"Subject: {subject_dir.name}\n"
                f"Reference: {reference_name}\n"
                f"Mismatch: {name}\n\n"
                "The modalities and segmentation are expected to be "
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
        name: str(image.get_data_dtype())
        for name, image in images.items()
    }


def validate_segmentation_labels(
    subject_dir: Path,
    seg_path: Path,
) -> set[int]:
    """Load one segmentation volume and verify its dtype and label values."""
    seg_nii = load_nifti_header(seg_path)
    seg = np.asanyarray(seg_nii.dataobj)

    if not np.all(np.isfinite(seg)):
        raise ValueError(
            "Segmentation contains non-finite values.\n\n"
            f"Subject: {subject_dir.name}\n"
            f"File:\n{seg_path}"
        )

    if not np.issubdtype(
        seg.dtype,
        np.integer,
    ):
        raise ValueError(
            "Segmentation must use an integer dtype.\n\n"
            f"Subject: {subject_dir.name}\n"
            f"Observed dtype: {seg.dtype}\n"
            f"File:\n{seg_path}"
        )

    observed_labels = {
        int(value)
        for value in np.unique(seg)
    }

    invalid_labels = (
        observed_labels
        - ALLOWED_SEGMENTATION_LABELS
    )

    if invalid_labels:
        raise ValueError(
            "Unexpected segmentation label(s) detected.\n\n"
            f"Subject: {subject_dir.name}\n"
            f"Observed labels: {sorted(observed_labels)}\n"
            f"Allowed labels: "
            f"{sorted(ALLOWED_SEGMENTATION_LABELS)}\n"
            f"Unexpected labels: {sorted(invalid_labels)}\n"
            f"File:\n{seg_path}"
        )

    return observed_labels


def validate_dataset(
    data_root: Path,
) -> dict:
    """
    Perform the complete one-time raw-dataset validation.

    Returns
    -------
    dict
        Dataset-level information to be written to dataset.yaml.
    """
    print(
        "\nValidating raw BraTS 2020 training dataset...",
        flush=True,
    )

    print(
        "This is a one-time full validation.",
        flush=True,
    )

    subject_dirs = validate_root_directory(data_root)

    print(
        f"\nFound {len(subject_dirs):,} expected subject directories.",
        flush=True,
    )

    observed_modality_dtypes = {
        modality_name: set()
        for modality_name in MODALITIES
    }

    observed_segmentation_dtypes = set()
    observed_segmentation_labels = set()

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

        labels = validate_segmentation_labels(
            subject_dir,
            subject_files["seg"],
        )

        for modality_name in MODALITIES:
            observed_modality_dtypes[
                modality_name
            ].add(
                dtypes[modality_name]
            )

        observed_segmentation_dtypes.add(
            dtypes["seg"]
        )

        observed_segmentation_labels.update(
            labels
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
        * (
            len(MODALITIES)
            + 1
        )
    )

    if files_checked != expected_total_files:
        raise ValueError(
            "Unexpected validated file count.\n\n"
            f"Expected: {expected_total_files}\n"
            f"Observed: {files_checked}"
        )

    if (
        observed_segmentation_labels
        != ALLOWED_SEGMENTATION_LABELS
    ):
        raise ValueError(
            "Dataset-wide segmentation labels do not match the "
            "expected BraTS label set.\n\n"
            f"Expected: "
            f"{sorted(ALLOWED_SEGMENTATION_LABELS)}\n"
            f"Observed: "
            f"{sorted(observed_segmentation_labels)}"
        )

    return {
        "subject_count": len(subject_dirs),

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

        "segmentation_dtypes": sorted(
            observed_segmentation_dtypes
        ),

        "segmentation_labels": sorted(
            observed_segmentation_labels
        ),
    }


def build_dataset_specification(
    data_root: Path,
    validation_result: dict,
) -> dict:
    """Construct the dataset.yaml contents."""
    return {
        "schema_version": 1,

        "dataset": {
            "id": "brats2020_training",
            "name": DATASET_NAME,
            "split": "training",
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
                "BraTS20_Training_{id:03d}"
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
            "file_pattern": "*_seg.nii",

            "filename_exceptions": {
                subject_name: filename
                for subject_name, filename
                in SEGMENTATION_FILENAME_EXCEPTIONS.items()
            },

            "observed_dtypes": (
                validation_result[
                    "segmentation_dtypes"
                ]
            ),

            "labels": validation_result[
                "segmentation_labels"
            ],

            "label_meaning": {
                0: "background",
                1: "NCR/NET",
                2: "edema",
                4: "enhancing tumor",
            },
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
    """Write the validated dataset specification to YAML."""
    if output_path.exists():
        raise ValueError(
            "The selected output file already exists.\n\n"
            f"{output_path}\n\n"
            "No existing dataset specification will be overwritten "
            "implicitly. Select a new filename or remove the existing "
            "file explicitly."
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
            "The dataset specification could not be written.\n\n"
            f"Output:\n{output_path}\n\n"
            f"Error:\n{exc}"
        ) from exc


# returns path, updated args, updated conf
#   
def get_path(path_name, args, conf, selector_function):
    out_val = None
    args_val = vars(args).get(path_name, None)
    conf_val = conf.get(path_name, None)
    if args_val is not None:
        out_val = Path(args_val)
        conf[path_name] = args_val
    elif conf_val is not None:
        out_val = Path(conf_val)
    else:
        out_val = selector_function()
        conf[path_name] = str(out_val)
    return out_val, conf

def get_folders_config(args):
    if args.folders_file is not None:
        if Path(args.folders_file).exists():
            with open(args.folders_file, "r") as file:
                conf = yaml.safe_load(file)
        else:
            conf = dict()
    else:
        conf = dict()
    print(f"conf={conf}")
    return conf    

def get_folders(args):
    conf = get_folders_config(args)
    data_root, conf = get_path("data_root", args, conf, select_dataset_directory)
    output_path, conf = get_path("yaml_dataset_path", args, conf, select_yaml_output_path)
    if args.folders_file is not None:
        with open(args.folders_file, "w") as file:
            yaml.safe_dump(conf, file)
    if output_path.exists() and not args.overwrite:
        print(f"File {output_path} exists already. nothing to do. Exiting")
        exit(1)
    return data_root, output_path


def main(args) -> None:
    """Run interactive BraTS dataset registration."""
    print("=" * 72)
    print(
        "BraTS 2020 Dataset Registration"
    )
    print("=" * 72)

    try:
        data_root, output_path = get_folders(args)
        
        validation_result = validate_dataset(
            data_root
        )

        print(
            "\n" + "=" * 72
        )
        print(
            "DATASET VALIDATION PASSED"
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
            f"Segmentation dtypes: "
            f"{validation_result['segmentation_dtypes']}"
        )

        print(
            f"Segmentation labels: "
            f"{validation_result['segmentation_labels']}"
        )

            
        specification = build_dataset_specification(
            data_root=data_root,
            validation_result=validation_result,
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
            "DATASET REGISTRATION FAILED"
        )
        print("=" * 72)
        print(exc)

        print(
            "\nNo dataset specification was written."
        )

        sys.exit(1)

    print(
        "\n" + "=" * 72
    )
    print(
        "DATASET REGISTRATION COMPLETE"
    )
    print("=" * 72)

    print(
        f"Dataset specification:\n{output_path}"
    )

    print(
        "\nThe raw BraTS dataset has been registered successfully."
    )

    print(
        "Downstream workflows should use this dataset.yaml file "
        "instead of rescanning the raw dataset."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type = str, default = None)
    parser.add_argument("--yaml_dataset_path", type = str, default = None)
    parser.add_argument("--folders_file", type=str, default="./data/folders.yaml")
    parser.add_argument("--overwrite", action='store_true')
    args = parser.parse_args()
    main(args)
