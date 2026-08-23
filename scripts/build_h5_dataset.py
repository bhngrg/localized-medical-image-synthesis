#!/usr/bin/env python3

"""
Build the historical BraTS 2020 H5 slice dataset from a registered raw dataset.

Responsibilities
----------------
1. Ask the user to select a dataset.yaml created by register_dataset.py.
2. Validate the dataset specification itself (not the raw dataset).
3. Ask the user to select an existing output directory for the H5 files.
4. Reproduce the historical NIfTI -> H5 conversion exactly.
5. Write one H5 file per axial slice.

Historical conversion reproduced by this script
------------------------------------------------
For each BraTS training subject and each axial slice:

Image channels
    channel 0 = FLAIR
    channel 1 = T1
    channel 2 = T1ce
    channel 3 = T2

Each 2-D modality slice is standardized independently using all pixels:

    z = (x - mean(x)) / std(x)

where NumPy's population standard deviation (ddof=0) is used.

If a slice has zero standard deviation, the output slice is all zeros.

Mask channels
    channel 0 = (segmentation == 1)
    channel 1 = (segmentation == 2)
    channel 2 = (segmentation == 4)

Output
------
Each H5 file contains:
    image : float64 array with shape (240, 240, 4)
    mask  : uint8 array with shape (240, 240, 3)

Files are named:
    volume_<subject_numeric_id>_slice_<slice_index>.h5

For BraTS 2020 training data this produces:
    369 subjects x 155 slices = 57,195 H5 files

This script does not repeat the full raw-dataset registration scan.
It trusts a successfully generated dataset.yaml and fails if required
registered files are no longer accessible.
"""

from __future__ import annotations

from pathlib import Path
import sys
import tkinter as tk
from tkinter import filedialog

import h5py
import nibabel as nib
import numpy as np
import yaml
import argparse
from register_dataset import get_path, get_folders_config

SUPPORTED_SCHEMA_VERSION = 1
SUPPORTED_DATASET_ID = "brats2020_training"

EXPECTED_VOLUME_SHAPE = (240, 240, 155)

H5_IMAGE_MODALITY_ORDER = (
    "flair",
    "t1",
    "t1ce",
    "t2",
)

H5_MASK_LABEL_ORDER = (
    1,
    2,
    4,
)


def create_tk_root() -> tk.Tk:
    """Create a hidden Tk root window."""
    root = tk.Tk()
    root.withdraw()
    root.update()
    return root


def select_dataset_yaml() -> Path:
    """
    Ask the user to select dataset.yaml.

    Returns
    -------
    pathlib.Path
        Path to the selected dataset specification.

    Raises
    ------
    RuntimeError
        If the user cancels the dialog.
    """
    root = create_tk_root()

    try:
        selected = filedialog.askopenfilename(
            parent=root,
            title="Select the registered BraTS dataset.yaml",
            filetypes=[
                ("YAML files", "*.yaml"),
                ("YAML files", "*.yml"),
            ],
        )
    finally:
        root.destroy()

    if not selected:
        raise RuntimeError(
            "No dataset specification was selected. "
            "H5 dataset construction was cancelled."
        )

    return Path(selected).expanduser().resolve()


def select_output_directory() -> Path:
    """
    Ask the user to select an existing output directory.

    The directory must not already contain H5 files.

    Returns
    -------
    pathlib.Path
        Selected output directory.

    Raises
    ------
    RuntimeError
        If the user cancels the dialog.
    """
    root = create_tk_root()

    try:
        selected = filedialog.askdirectory(
            parent=root,
            title="Select an empty directory for the generated H5 files",
        )
    finally:
        root.destroy()

    if not selected:
        raise RuntimeError(
            "No output directory was selected. "
            "H5 dataset construction was cancelled."
        )

    return Path(selected).expanduser().resolve()


def load_dataset_specification(
    yaml_path: Path,
) -> dict:
    """
    Load dataset.yaml.

    This validates the YAML specification itself. It does not repeat the
    full raw-data registration scan.
    """
    if not yaml_path.exists():
        raise ValueError(
            "The selected dataset specification does not exist:\n\n"
            f"{yaml_path}"
        )

    if not yaml_path.is_file():
        raise ValueError(
            "The selected dataset specification is not a file:\n\n"
            f"{yaml_path}"
        )

    try:
        with yaml_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            specification = yaml.safe_load(file)

    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(
            "The dataset specification could not be read.\n\n"
            f"File:\n{yaml_path}\n\n"
            f"Error:\n{exc}"
        ) from exc

    if not isinstance(
        specification,
        dict,
    ):
        raise ValueError(
            "The dataset specification must contain a YAML mapping "
            "at the top level."
        )

    return specification


def require_mapping(
    parent: dict,
    key: str,
) -> dict:
    """Return a required mapping from a YAML structure."""
    value = parent.get(key)

    if not isinstance(
        value,
        dict,
    ):
        raise ValueError(
            f"dataset.yaml is missing the required mapping: {key}"
        )

    return value


def validate_dataset_specification(
    specification: dict,
) -> dict:
    """
    Validate the registered dataset specification.

    This checks compatibility and required fields only. It deliberately
    does not rescan all raw NIfTI volumes.
    """
    schema_version = specification.get(
        "schema_version"
    )

    if schema_version != SUPPORTED_SCHEMA_VERSION:
        raise ValueError(
            "Unsupported dataset.yaml schema version.\n\n"
            f"Expected: {SUPPORTED_SCHEMA_VERSION}\n"
            f"Observed: {schema_version}"
        )

    dataset = require_mapping(
        specification,
        "dataset",
    )

    subjects = require_mapping(
        specification,
        "subjects",
    )

    volumes = require_mapping(
        specification,
        "volumes",
    )

    modalities = require_mapping(
        specification,
        "modalities",
    )

    segmentation = require_mapping(
        specification,
        "segmentation",
    )

    validation = require_mapping(
        specification,
        "validation",
    )

    dataset_id = dataset.get("id")

    if dataset_id != SUPPORTED_DATASET_ID:
        raise ValueError(
            "This H5 builder currently supports only the registered "
            "BraTS 2020 training dataset.\n\n"
            f"Expected dataset id: {SUPPORTED_DATASET_ID}\n"
            f"Observed dataset id: {dataset_id}"
        )

    if validation.get("status") != "passed":
        raise ValueError(
            "dataset.yaml does not record a successful registration.\n\n"
            "Run scripts/register_dataset.py first."
        )

    raw_data_root_value = dataset.get(
        "raw_data_root"
    )

    if not isinstance(
        raw_data_root_value,
        str,
    ) or not raw_data_root_value:
        raise ValueError(
            "dataset.yaml does not contain a valid dataset.raw_data_root."
        )

    raw_data_root = Path(
        raw_data_root_value
    ).expanduser().resolve()

    if not raw_data_root.exists():
        raise ValueError(
            "The registered raw dataset directory is no longer accessible.\n\n"
            f"Registered path:\n{raw_data_root}\n\n"
            "If the dataset was moved, run register_dataset.py again "
            "to create a new dataset.yaml."
        )

    if not raw_data_root.is_dir():
        raise ValueError(
            "The registered raw dataset path is no longer a directory.\n\n"
            f"Registered path:\n{raw_data_root}"
        )

    subject_count = subjects.get(
        "count"
    )

    first_numeric_id = subjects.get(
        "first_numeric_id"
    )

    last_numeric_id = subjects.get(
        "last_numeric_id"
    )

    id_pattern = subjects.get(
        "id_pattern"
    )

    if not isinstance(
        subject_count,
        int,
    ) or subject_count <= 0:
        raise ValueError(
            "dataset.yaml contains an invalid subjects.count."
        )

    if not isinstance(
        first_numeric_id,
        int,
    ):
        raise ValueError(
            "dataset.yaml contains an invalid subjects.first_numeric_id."
        )

    if not isinstance(
        last_numeric_id,
        int,
    ):
        raise ValueError(
            "dataset.yaml contains an invalid subjects.last_numeric_id."
        )

    if not isinstance(
        id_pattern,
        str,
    ) or "{id" not in id_pattern:
        raise ValueError(
            "dataset.yaml contains an invalid subjects.id_pattern."
        )

    numeric_ids = list(
        range(
            first_numeric_id,
            last_numeric_id + 1,
        )
    )

    if len(numeric_ids) != subject_count:
        raise ValueError(
            "Subject count and numeric subject range in dataset.yaml "
            "are inconsistent.\n\n"
            f"subjects.count: {subject_count}\n"
            f"numeric range size: {len(numeric_ids)}"
        )

    volume_shape = volumes.get(
        "shape"
    )

    slice_axis = volumes.get(
        "slice_axis"
    )

    slices_per_subject = volumes.get(
        "slices_per_subject"
    )

    if tuple(volume_shape or ()) != EXPECTED_VOLUME_SHAPE:
        raise ValueError(
            "Unexpected registered volume shape.\n\n"
            f"Expected: {EXPECTED_VOLUME_SHAPE}\n"
            f"Observed: {volume_shape}"
        )

    if slice_axis != 2:
        raise ValueError(
            "This historical H5 conversion expects slice_axis = 2.\n\n"
            f"Observed: {slice_axis}"
        )

    if slices_per_subject != EXPECTED_VOLUME_SHAPE[2]:
        raise ValueError(
            "Unexpected number of slices per subject.\n\n"
            f"Expected: {EXPECTED_VOLUME_SHAPE[2]}\n"
            f"Observed: {slices_per_subject}"
        )

    modality_files = modalities.get(
        "files"
    )

    if not isinstance(
        modality_files,
        dict,
    ):
        raise ValueError(
            "dataset.yaml is missing modalities.files."
        )

    for modality_name in H5_IMAGE_MODALITY_ORDER:
        pattern = modality_files.get(
            modality_name
        )

        if not isinstance(
            pattern,
            str,
        ) or not pattern:
            raise ValueError(
                "dataset.yaml is missing a file pattern for "
                f"modality '{modality_name}'."
            )

    segmentation_pattern = segmentation.get(
        "file_pattern"
    )

    if not isinstance(
        segmentation_pattern,
        str,
    ) or not segmentation_pattern:
        raise ValueError(
            "dataset.yaml is missing segmentation.file_pattern."
        )

    filename_exceptions = segmentation.get(
        "filename_exceptions",
        {},
    )

    if not isinstance(
        filename_exceptions,
        dict,
    ):
        raise ValueError(
            "dataset.yaml segmentation.filename_exceptions must "
            "be a mapping."
        )

    return {
        "raw_data_root": raw_data_root,
        "subject_count": subject_count,
        "numeric_ids": numeric_ids,
        "id_pattern": id_pattern,
        "volume_shape": EXPECTED_VOLUME_SHAPE,
        "slices_per_subject": slices_per_subject,
        "modality_files": modality_files,
        "segmentation_pattern": segmentation_pattern,
        "segmentation_filename_exceptions": filename_exceptions,
    }


def validate_output_directory(
    output_dir: Path,
) -> None:
    """
    Validate the selected H5 output directory.

    The directory must already exist and must not contain H5 files.
    No directories are created implicitly and existing H5 files are not
    overwritten or merged.
    """
    if not output_dir.exists():
        raise ValueError(
            "The selected output directory does not exist:\n\n"
            f"{output_dir}"
        )

    if not output_dir.is_dir():
        raise ValueError(
            "The selected output path is not a directory:\n\n"
            f"{output_dir}"
        )

    existing_h5_files = list(
        output_dir.glob("*.h5")
    )

    # if existing_h5_files:
    #     raise ValueError(
    #         "The selected output directory already contains H5 files.\n\n"
    #         f"Directory:\n{output_dir}\n\n"
    #         f"Existing H5 files found: {len(existing_h5_files):,}\n\n"
    #         "No existing H5 files will be overwritten or merged "
    #         "implicitly. Select an empty output directory."
    #     )


def subject_name_from_spec(
    id_pattern: str,
    subject_numeric_id: int,
) -> str:
    """Create one registered subject name from the YAML pattern."""
    try:
        subject_name = id_pattern.format(
            id=subject_numeric_id
        )

    except (KeyError, ValueError) as exc:
        raise ValueError(
            "The registered subjects.id_pattern could not be formatted.\n\n"
            f"Pattern: {id_pattern}\n"
            f"Subject numeric id: {subject_numeric_id}"
        ) from exc

    return subject_name


def resolve_modality_path(
    subject_dir: Path,
    subject_name: str,
    modality_name: str,
    registered_pattern: str,
) -> Path:
    """
    Resolve one modality file using the registered filename pattern.

    No recursive search is performed.
    """
    expected_filename = registered_pattern.replace(
        "*",
        subject_name,
        1,
    )

    path = subject_dir / expected_filename

    if not path.is_file():
        raise ValueError(
            "A required registered MRI file is no longer accessible.\n\n"
            f"Subject: {subject_name}\n"
            f"Modality: {modality_name}\n"
            f"Expected file:\n{path}\n\n"
            "The raw dataset may have been changed after registration."
        )

    return path


def resolve_segmentation_path(
    subject_dir: Path,
    subject_name: str,
    registered_pattern: str,
    filename_exceptions: dict,
) -> Path:
    """
    Resolve the segmentation path using the registered explicit exception map.
    """
    if subject_name in filename_exceptions:
        filename = filename_exceptions[
            subject_name
        ]
    else:
        filename = registered_pattern.replace(
            "*",
            subject_name,
            1,
        )

    path = subject_dir / filename

    if not path.is_file():
        raise ValueError(
            "The registered segmentation file is no longer accessible.\n\n"
            f"Subject: {subject_name}\n"
            f"Expected file:\n{path}\n\n"
            "The raw dataset may have been changed after registration."
        )

    return path


def load_nifti_volume(
    path: Path,
) -> np.ndarray:
    """
    Load one NIfTI volume as float64 using nibabel.

    nibabel.get_fdata() returns floating-point values after applying the
    NIfTI header scaling, matching the historical conversion verified
    against the original H5 dataset.
    """
    try:
        image = nib.load(
            str(path)
        )
        volume = image.get_fdata(
            dtype=np.float64
        )

    except Exception as exc:
        raise ValueError(
            "A registered NIfTI file could not be loaded.\n\n"
            f"File:\n{path}\n\n"
            f"Error:\n{exc}"
        ) from exc

    if volume.shape != EXPECTED_VOLUME_SHAPE:
        raise ValueError(
            "A registered NIfTI file no longer has the expected shape.\n\n"
            f"File:\n{path}\n"
            f"Expected: {EXPECTED_VOLUME_SHAPE}\n"
            f"Observed: {volume.shape}\n\n"
            "The raw dataset may have been changed after registration."
        )

    return volume


def load_segmentation_volume(
    path: Path,
) -> np.ndarray:
    """
    Load the segmentation while preserving its integer labels.
    """
    try:
        image = nib.load(
            str(path)
        )
        segmentation = np.asanyarray(
            image.dataobj
        )

    except Exception as exc:
        raise ValueError(
            "A registered segmentation file could not be loaded.\n\n"
            f"File:\n{path}\n\n"
            f"Error:\n{exc}"
        ) from exc

    if segmentation.shape != EXPECTED_VOLUME_SHAPE:
        raise ValueError(
            "A registered segmentation file no longer has the expected "
            "shape.\n\n"
            f"File:\n{path}\n"
            f"Expected: {EXPECTED_VOLUME_SHAPE}\n"
            f"Observed: {segmentation.shape}\n\n"
            "The raw dataset may have been changed after registration."
        )

    return segmentation


def standardize_slice(
    image_slice: np.ndarray,
) -> np.ndarray:
    """
    Reproduce the historical per-slice all-pixel z-score standardization.

    All pixels, including background, contribute to the mean and standard
    deviation. Population standard deviation (ddof=0) is used.

    Constant slices are returned as all-zero float64 arrays.
    """
    image_slice = np.asarray(
        image_slice,
        dtype=np.float64,
    )

    mean = image_slice.mean()
    std = image_slice.std(
        ddof=0
    )

    if std == 0:
        return np.zeros_like(
            image_slice,
            dtype=np.float64,
        )

    return (
        image_slice - mean
    ) / std


def build_h5_image(
    modality_volumes: dict[str, np.ndarray],
    slice_index: int,
) -> np.ndarray:
    """Build the historical four-channel H5 image for one axial slice."""
    channels = []

    for modality_name in H5_IMAGE_MODALITY_ORDER:
        raw_slice = modality_volumes[
            modality_name
        ][
            :,
            :,
            slice_index,
        ]

        standardized = standardize_slice(
            raw_slice
        )

        channels.append(
            standardized
        )

    return np.stack(
        channels,
        axis=-1,
    ).astype(
        np.float64,
        copy=False,
    )


def build_h5_mask(
    segmentation_volume: np.ndarray,
    slice_index: int,
) -> np.ndarray:
    """Build the historical three-channel uint8 mask for one axial slice."""
    seg_slice = segmentation_volume[
        :,
        :,
        slice_index,
    ]

    channels = [
        (
            seg_slice == label
        ).astype(
            np.uint8
        )
        for label in H5_MASK_LABEL_ORDER
    ]

    return np.stack(
        channels,
        axis=-1,
    ).astype(
        np.uint8,
        copy=False,
    )


def write_h5_slice(
    output_path: Path,
    image: np.ndarray,
    mask: np.ndarray,
) -> None:
    """
    Write one historical-style H5 slice.

    Existing files are never overwritten.
    """
    # if output_path.exists():
    #     raise ValueError(
    #         "Refusing to overwrite an existing H5 file.\n\n"
    #         f"{output_path}"
        # )

    try:
        with h5py.File(
            output_path,
            "w",
        ) as h5_file:
            h5_file.create_dataset(
                "image",
                data=image,
                dtype=np.float64,
            )

            h5_file.create_dataset(
                "mask",
                data=mask,
                dtype=np.uint8,
            )

    except Exception as exc:
        raise ValueError(
            "An H5 slice could not be written.\n\n"
            f"File:\n{output_path}\n\n"
            f"Error:\n{exc}"
        ) from exc


def build_subject_h5_files(
    raw_data_root: Path,
    output_dir: Path,
    subject_numeric_id: int,
    id_pattern: str,
    modality_files: dict[str, str],
    segmentation_pattern: str,
    segmentation_filename_exceptions: dict,
    slices_per_subject: int,
) -> int:
    """
    Convert one registered BraTS subject to H5 slices.

    Returns
    -------
    int
        Number of H5 files written.
    """
    subject_name = subject_name_from_spec(
        id_pattern=id_pattern,
        subject_numeric_id=subject_numeric_id,
    )

    subject_dir = (
        raw_data_root
        / subject_name
    )

    if not subject_dir.is_dir():
        raise ValueError(
            "A registered subject directory is no longer accessible.\n\n"
            f"Subject: {subject_name}\n"
            f"Expected directory:\n{subject_dir}\n\n"
            "The raw dataset may have been moved or changed after registration."
        )

    modality_volumes = {}

    for modality_name in H5_IMAGE_MODALITY_ORDER:
        modality_path = resolve_modality_path(
            subject_dir=subject_dir,
            subject_name=subject_name,
            modality_name=modality_name,
            registered_pattern=modality_files[
                modality_name
            ],
        )

        modality_volumes[
            modality_name
        ] = load_nifti_volume(
            modality_path
        )

    segmentation_path = resolve_segmentation_path(
        subject_dir=subject_dir,
        subject_name=subject_name,
        registered_pattern=segmentation_pattern,
        filename_exceptions=segmentation_filename_exceptions,
    )

    segmentation_volume = load_segmentation_volume(
        segmentation_path
    )

    files_written = 0

    for slice_index in range(
        slices_per_subject
    ):
        image = build_h5_image(
            modality_volumes=modality_volumes,
            slice_index=slice_index,
        )

        mask = build_h5_mask(
            segmentation_volume=segmentation_volume,
            slice_index=slice_index,
        )

        output_path = (
            output_dir
            / (
                f"volume_{subject_numeric_id}_"
                f"slice_{slice_index}.h5"
            )
        )

        write_h5_slice(
            output_path=output_path,
            image=image,
            mask=mask,
        )

        files_written += 1

    return files_written


def build_h5_dataset(
    registered: dict,
    output_dir: Path,
) -> int:
    """
    Build all historical-style H5 slices from the registered dataset.

    Returns
    -------
    int
        Total number of H5 files written.
    """
    numeric_ids = registered[
        "numeric_ids"
    ]

    subject_count = registered[
        "subject_count"
    ]

    slices_per_subject = registered[
        "slices_per_subject"
    ]

    expected_h5_count = (
        subject_count
        * slices_per_subject
    )

    print(
        "\nBuilding historical BraTS H5 slice dataset...",
        flush=True,
    )

    print(
        f"Subjects        : {subject_count:,}",
        flush=True,
    )

    print(
        f"Slices/subject  : {slices_per_subject:,}",
        flush=True,
    )

    print(
        f"Expected H5 files: {expected_h5_count:,}",
        flush=True,
    )

    print(
        f"Output directory: {output_dir}",
        flush=True,
    )

    total_written = 0

    for subject_index, subject_numeric_id in enumerate(
        numeric_ids,
        start=1,
    ):
        files_written = build_subject_h5_files(
            raw_data_root=registered[
                "raw_data_root"
            ],
            output_dir=output_dir,
            subject_numeric_id=subject_numeric_id,
            id_pattern=registered[
                "id_pattern"
            ],
            modality_files=registered[
                "modality_files"
            ],
            segmentation_pattern=registered[
                "segmentation_pattern"
            ],
            segmentation_filename_exceptions=registered[
                "segmentation_filename_exceptions"
            ],
            slices_per_subject=slices_per_subject,
        )

        total_written += files_written

        if (
            subject_index % 10 == 0
            or subject_index == subject_count
        ):
            print(
                f"  Converted {subject_index:,} / "
                f"{subject_count:,} subjects "
                f"({total_written:,} H5 files written)",
                flush=True,
            )

    if total_written != expected_h5_count:
        raise ValueError(
            "Unexpected H5 output count.\n\n"
            f"Expected: {expected_h5_count}\n"
            f"Written: {total_written}"
        )

    return total_written




def get_folders(args):
    conf = get_folders_config(args)
    yaml_path, conf = get_path("yaml_dataset_path", args, conf, select_dataset_yaml)
    output_path, conf = get_path("h5_root", args, conf, select_output_directory)
    if args.folders_file is not None:
        with open(args.folders_file, "w") as file:
            yaml.safe_dump(conf, file)
    if output_path.exists() and not args.overwrite:
        print(f"File {output_path} exists already. nothing to do. Exiting")
        exit(1)
    return yaml_path, output_path


def main(args) -> None:
    """Run interactive historical H5 dataset construction."""
    print("=" * 72)
    print(
        "BraTS 2020 H5 Dataset Builder"
    )
    print("=" * 72)

    try:
        # yaml_path = select_dataset_yaml()
        yaml_path, output_dir = get_folders(args)

        print(
            "\nSelected dataset specification:"
        )
        print(
            yaml_path
        )

        specification = load_dataset_specification(
            yaml_path
        )

        registered = validate_dataset_specification(
            specification
        )

        print(
            "\nRegistered dataset specification accepted."
        )
        print(
            f"Raw dataset root : "
            f"{registered['raw_data_root']}"
        )
        print(
            f"Subjects         : "
            f"{registered['subject_count']:,}"
        )
        print(
            f"Volume shape     : "
            f"{registered['volume_shape']}"
        )


        validate_output_directory(
            output_dir
        )

        total_written = build_h5_dataset(
            registered=registered,
            output_dir=output_dir,
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
            "H5 DATASET BUILD FAILED"
        )
        print("=" * 72)
        print(
            exc
        )
        print(
            "\nNo implicit recovery or overwrite was attempted."
        )
        sys.exit(1)

    print(
        "\n" + "=" * 72
    )
    print(
        "H5 DATASET BUILD COMPLETE"
    )
    print("=" * 72)

    print(
        f"Output directory : {output_dir}"
    )
    print(
        f"H5 files written : {total_written:,}"
    )

    print(
        "\nHistorical NIfTI -> H5 conversion completed successfully."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml_dataset_path", type=str, default=None)
    parser.add_argument("--h5_root", type = str, default = None)
    parser.add_argument("--folders_file", type=str, default="./data/folders.yaml")
    parser.add_argument("--overwrite", action='store_true')
    args = parser.parse_args()
    main(args)
