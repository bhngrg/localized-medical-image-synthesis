#!/usr/bin/env python3

"""
Create a sample-level manifest for the BraTS 2020 H5 dataset.

Responsibilities
----------------
1. Ask the user to select dataset.yaml created by register_dataset.py.
2. Ask the user to select the directory containing the generated H5 files.
3. Validate the expected H5 filename grid against the registered dataset.
4. Read each H5 mask once and compute sample-level metadata.
5. Ask the user where to save manifest.csv.
6. Write the manifest without modifying the H5 dataset.

Manifest columns
----------------
slice_path
    Relative H5 filename, e.g. volume_1_slice_70.h5.

target
    1 if any tumor-label pixel is present in the slice, otherwise 0.

volume
    Numeric BraTS training subject identifier used by the historical H5
    naming convention.

slice
    Zero-based axial slice index.

label0_pxl_cnt
    Number of positive pixels in H5 mask channel 0 (BraTS label 1).

label1_pxl_cnt
    Number of positive pixels in H5 mask channel 1 (BraTS label 2).

label2_pxl_cnt
    Number of positive pixels in H5 mask channel 2 (BraTS label 4).

background_ratio
    Fraction of spatial pixels not assigned to any of the three tumor
    mask channels.

This script is intended to run once after build_h5_dataset.py. Downstream
training and evaluation code should consume manifest.csv rather than
recomputing these mask summaries.
"""

from __future__ import annotations

import csv
from pathlib import Path
import re
import sys
import tkinter as tk
from tkinter import filedialog
from register_dataset import get_path, get_folders_config
import h5py
import numpy as np
import yaml
import argparse

SUPPORTED_SCHEMA_VERSION = 1
SUPPORTED_DATASET_ID = "brats2020_training"

EXPECTED_IMAGE_SHAPE = (240, 240, 4)
EXPECTED_MASK_SHAPE = (240, 240, 3)
EXPECTED_IMAGE_DTYPE = np.dtype("float64")
EXPECTED_MASK_DTYPE = np.dtype("uint8")

H5_FILENAME_RE = re.compile(
    r"^volume_(?P<volume>\d+)_slice_(?P<slice>\d+)\.h5$"
)

MANIFEST_FIELDNAMES = [
    "slice_path",
    "target",
    "volume",
    "slice",
    "label0_pxl_cnt",
    "label1_pxl_cnt",
    "label2_pxl_cnt",
    "background_ratio",
]


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
        Selected dataset specification.

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
            "Manifest creation was cancelled."
        )

    return Path(selected).expanduser().resolve()


def select_h5_directory() -> Path:
    """
    Ask the user to select the directory containing H5 files.

    Returns
    -------
    pathlib.Path
        H5 dataset directory.

    Raises
    ------
    RuntimeError
        If the user cancels the dialog.
    """
    root = create_tk_root()

    try:
        selected = filedialog.askdirectory(
            parent=root,
            title="Select the directory containing the BraTS H5 files",
        )
    finally:
        root.destroy()

    if not selected:
        raise RuntimeError(
            "No H5 dataset directory was selected. "
            "Manifest creation was cancelled."
        )

    return Path(selected).expanduser().resolve()


def select_manifest_output_path() -> Path:
    """
    Ask the user where manifest.csv should be saved.

    Returns
    -------
    pathlib.Path
        Selected CSV output path.

    Raises
    ------
    RuntimeError
        If the user cancels the save dialog.
    """
    root = create_tk_root()

    try:
        selected = filedialog.asksaveasfilename(
            parent=root,
            title="Save BraTS H5 dataset manifest",
            defaultextension=".csv",
            initialfile="manifest.csv",
            filetypes=[
                ("CSV files", "*.csv"),
            ],
        )
    finally:
        root.destroy()

    if not selected:
        raise RuntimeError(
            "No manifest output file was selected. "
            "Manifest creation was cancelled."
        )

    output_path = Path(selected).expanduser().resolve()

    if output_path.suffix.lower() != ".csv":
        raise ValueError(
            "The dataset manifest must use the .csv extension.\n\n"
            f"Selected output:\n{output_path}"
        )

    return output_path


def load_dataset_specification(
    yaml_path: Path,
) -> dict:
    """Load and minimally validate dataset.yaml."""
    if not yaml_path.is_file():
        raise ValueError(
            "The selected dataset specification is not accessible:\n\n"
            f"{yaml_path}"
        )

    try:
        with yaml_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            specification = yaml.safe_load(
                file
            )

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
            "dataset.yaml must contain a YAML mapping at the top level."
        )

    return specification


def require_mapping(
    parent: dict,
    key: str,
) -> dict:
    """Return one required YAML mapping."""
    value = parent.get(
        key
    )

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
    Validate only the fields needed to construct the H5 manifest.
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

    validation = require_mapping(
        specification,
        "validation",
    )

    if dataset.get(
        "id"
    ) != SUPPORTED_DATASET_ID:
        raise ValueError(
            "This manifest builder currently supports only the registered "
            "BraTS 2020 training dataset.\n\n"
            f"Expected dataset id: {SUPPORTED_DATASET_ID}\n"
            f"Observed dataset id: {dataset.get('id')}"
        )

    if validation.get(
        "status"
    ) != "passed":
        raise ValueError(
            "dataset.yaml does not record a successful dataset registration."
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

    slices_per_subject = volumes.get(
        "slices_per_subject"
    )

    volume_shape = tuple(
        volumes.get(
            "shape",
            [],
        )
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
    ) or not isinstance(
        last_numeric_id,
        int,
    ):
        raise ValueError(
            "dataset.yaml contains an invalid numeric subject range."
        )

    if not isinstance(
        slices_per_subject,
        int,
    ) or slices_per_subject <= 0:
        raise ValueError(
            "dataset.yaml contains an invalid volumes.slices_per_subject."
        )

    if len(
        range(
            first_numeric_id,
            last_numeric_id + 1,
        )
    ) != subject_count:
        raise ValueError(
            "dataset.yaml subject count and numeric range are inconsistent."
        )

    if volume_shape[:2] != EXPECTED_MASK_SHAPE[:2]:
        raise ValueError(
            "Unexpected registered spatial volume shape.\n\n"
            f"Expected first two dimensions: {EXPECTED_MASK_SHAPE[:2]}\n"
            f"Observed: {volume_shape}"
        )

    return {
        "subject_count": subject_count,
        "first_numeric_id": first_numeric_id,
        "last_numeric_id": last_numeric_id,
        "slices_per_subject": slices_per_subject,
        "spatial_shape": volume_shape[:2],
        "expected_h5_count": (
            subject_count
            * slices_per_subject
        ),
    }


def validate_h5_directory(
    h5_root: Path,
    registered: dict,
) -> list[Path]:
    """
    Validate the expected H5 filename grid without reopening every file.

    Returns
    -------
    list[pathlib.Path]
        H5 files in deterministic volume/slice order.
    """
    if not h5_root.exists():
        raise ValueError(
            "The selected H5 directory does not exist:\n\n"
            f"{h5_root}"
        )

    if not h5_root.is_dir():
        raise ValueError(
            "The selected H5 path is not a directory:\n\n"
            f"{h5_root}"
        )

    observed_files = list(
        h5_root.glob(
            "*.h5"
        )
    )

    expected_count = registered[
        "expected_h5_count"
    ]

    if len(
        observed_files
    ) != expected_count:
        raise ValueError(
            "The selected H5 directory does not contain the expected "
            "number of H5 files.\n\n"
            f"Expected: {expected_count:,}\n"
            f"Observed: {len(observed_files):,}\n"
            f"Directory:\n{h5_root}"
        )

    observed_names = {
        path.name
        for path in observed_files
    }

    expected_paths = []
    missing = []

    for volume_id in range(
        registered[
            "first_numeric_id"
        ],
        registered[
            "last_numeric_id"
        ] + 1,
    ):
        for slice_index in range(
            registered[
                "slices_per_subject"
            ]
        ):
            filename = (
                f"volume_{volume_id}_"
                f"slice_{slice_index}.h5"
            )

            if filename not in observed_names:
                missing.append(
                    filename
                )

            expected_paths.append(
                h5_root / filename
            )

    if missing:
        raise ValueError(
            "The H5 filename grid is incomplete.\n\n"
            f"Missing files: {len(missing):,}\n"
            + "\n".join(
                f"  {name}"
                for name in missing[:10]
            )
            + (
                f"\n  ... and {len(missing) - 10} more"
                if len(missing) > 10
                else ""
            )
        )

    unexpected = []

    for path in observed_files:
        match = H5_FILENAME_RE.match(
            path.name
        )

        if match is None:
            unexpected.append(
                path.name
            )
            continue

        volume_id = int(
            match.group(
                "volume"
            )
        )

        slice_index = int(
            match.group(
                "slice"
            )
        )

        if not (
            registered["first_numeric_id"]
            <= volume_id
            <= registered["last_numeric_id"]
        ):
            unexpected.append(
                path.name
            )

        elif not (
            0
            <= slice_index
            < registered["slices_per_subject"]
        ):
            unexpected.append(
                path.name
            )

    if unexpected:
        raise ValueError(
            "Unexpected H5 filename(s) were found in the selected "
            "directory.\n\n"
            + "\n".join(
                f"  {name}"
                for name in unexpected[:10]
            )
            + (
                f"\n  ... and {len(unexpected) - 10} more"
                if len(unexpected) > 10
                else ""
            )
        )

    return expected_paths


def inspect_h5_file(
    path: Path,
    spatial_pixel_count: int,
) -> dict:
    """
    Validate one H5 file and compute its manifest row.
    """
    match = H5_FILENAME_RE.match(
        path.name
    )

    if match is None:
        raise ValueError(
            "Unexpected H5 filename encountered:\n\n"
            f"{path}"
        )

    volume_id = int(
        match.group(
            "volume"
        )
    )

    slice_index = int(
        match.group(
            "slice"
        )
    )

    try:
        with h5py.File(
            path,
            "r",
        ) as h5_file:
            if "image" not in h5_file:
                raise ValueError(
                    "H5 file is missing the 'image' dataset.\n\n"
                    f"File:\n{path}"
                )

            if "mask" not in h5_file:
                raise ValueError(
                    "H5 file is missing the 'mask' dataset.\n\n"
                    f"File:\n{path}"
                )

            image = h5_file[
                "image"
            ]

            mask = h5_file[
                "mask"
            ]

            if image.shape != EXPECTED_IMAGE_SHAPE:
                raise ValueError(
                    "Unexpected H5 image shape.\n\n"
                    f"Expected: {EXPECTED_IMAGE_SHAPE}\n"
                    f"Observed: {image.shape}\n"
                    f"File:\n{path}"
                )

            if mask.shape != EXPECTED_MASK_SHAPE:
                raise ValueError(
                    "Unexpected H5 mask shape.\n\n"
                    f"Expected: {EXPECTED_MASK_SHAPE}\n"
                    f"Observed: {mask.shape}\n"
                    f"File:\n{path}"
                )

            if np.dtype(
                image.dtype
            ) != EXPECTED_IMAGE_DTYPE:
                raise ValueError(
                    "Unexpected H5 image dtype.\n\n"
                    f"Expected: {EXPECTED_IMAGE_DTYPE}\n"
                    f"Observed: {image.dtype}\n"
                    f"File:\n{path}"
                )

            if np.dtype(
                mask.dtype
            ) != EXPECTED_MASK_DTYPE:
                raise ValueError(
                    "Unexpected H5 mask dtype.\n\n"
                    f"Expected: {EXPECTED_MASK_DTYPE}\n"
                    f"Observed: {mask.dtype}\n"
                    f"File:\n{path}"
                )

            # Only the mask data need to be loaded to compute manifest
            # summaries. The image array is not read into memory.
            mask_array = np.asarray(
                mask
            )

    except OSError as exc:
        raise ValueError(
            "An H5 file could not be opened.\n\n"
            f"File:\n{path}\n\n"
            f"HDF5 error:\n{exc}"
        ) from exc

    if not np.all(
        (mask_array == 0)
        | (mask_array == 1)
    ):
        raise ValueError(
            "H5 mask contains values other than 0 and 1.\n\n"
            f"File:\n{path}"
        )

    label_counts = [
        int(
            np.count_nonzero(
                mask_array[
                    :,
                    :,
                    channel_index,
                ]
            )
        )
        for channel_index in range(
            EXPECTED_MASK_SHAPE[-1]
        )
    ]

    tumor_pixels = sum(
        label_counts
    )

    if tumor_pixels > spatial_pixel_count:
        raise ValueError(
            "The H5 mask channels overlap unexpectedly.\n\n"
            f"File:\n{path}\n"
            f"Total positive mask pixels across channels: {tumor_pixels:,}\n"
            f"Spatial pixels: {spatial_pixel_count:,}"
        )

    target = int(
        tumor_pixels > 0
    )

    background_ratio = (
        1.0
        - (
            tumor_pixels
            / spatial_pixel_count
        )
    )

    return {
        "slice_path": path.name,
        "target": target,
        "volume": volume_id,
        "slice": slice_index,
        "label0_pxl_cnt": label_counts[0],
        "label1_pxl_cnt": label_counts[1],
        "label2_pxl_cnt": label_counts[2],
        "background_ratio": background_ratio,
    }


def write_manifest(
    output_path: Path,
    h5_paths: list[Path],
    registered: dict,
) -> dict:
    """
    Inspect all H5 files and write manifest.csv.

    Returns
    -------
    dict
        Summary counts for reporting.
    """
    # if output_path.exists():
    #     raise ValueError(
    #         "The selected manifest file already exists.\n\n"
    #         f"{output_path}\n\n"
    #         "No existing manifest will be overwritten implicitly."
    #     )

    spatial_pixel_count = int(
        np.prod(
            registered[
                "spatial_shape"
            ]
        )
    )

    total_rows = len(
        h5_paths
    )

    tumor_slices = 0
    non_tumor_slices = 0

    total_label_counts = [
        0,
        0,
        0,
    ]

    print(
        "\nCreating H5 dataset manifest...",
        flush=True,
    )

    print(
        f"H5 files to inspect: {total_rows:,}",
        flush=True,
    )

    try:
        with output_path.open(
            "w",
            encoding="utf-8",
            newline="",
        ) as csv_file:
            writer = csv.DictWriter(
                csv_file,
                fieldnames=MANIFEST_FIELDNAMES,
            )

            writer.writeheader()

            for index, path in enumerate(
                h5_paths,
                start=1,
            ):
                row = inspect_h5_file(
                    path=path,
                    spatial_pixel_count=spatial_pixel_count,
                )

                writer.writerow(
                    row
                )

                if row["target"] == 1:
                    tumor_slices += 1
                else:
                    non_tumor_slices += 1

                total_label_counts[0] += row[
                    "label0_pxl_cnt"
                ]

                total_label_counts[1] += row[
                    "label1_pxl_cnt"
                ]

                total_label_counts[2] += row[
                    "label2_pxl_cnt"
                ]

                if (
                    index % 5000 == 0
                    or index == total_rows
                ):
                    print(
                        f"  Processed {index:,} / "
                        f"{total_rows:,} H5 files",
                        flush=True,
                    )

    except OSError as exc:
        raise ValueError(
            "The manifest could not be written.\n\n"
            f"Output:\n{output_path}\n\n"
            f"Error:\n{exc}"
        ) from exc

    return {
        "rows": total_rows,
        "tumor_slices": tumor_slices,
        "non_tumor_slices": non_tumor_slices,
        "total_label0_pixels": total_label_counts[0],
        "total_label1_pixels": total_label_counts[1],
        "total_label2_pixels": total_label_counts[2],
    }

def get_folders(args):
    conf = get_folders_config(args)
    yaml_path, conf = get_path("yaml_dataset_path", args, conf, select_dataset_yaml)
    # yaml_path = select_dataset_yaml() if args.yaml_path is None else Path(args.yaml_path)
    # h5_root = select_h5_directory() if args.h5_root is None else Path(args.h5_root)
    h5_root, conf = get_path("h5_files_path", args, conf, select_h5_directory)
    # output_path, conf  = select_manifest_output_path if args.output_path is None else Path(args.output_path)
    output_path, conf = get_path("manifest_path", args, conf, select_manifest_output_path)
    if args.folders_file is not None:
        with open(args.folders_file, "w") as file:
            yaml.safe_dump(conf, file)
    if output_path.exists() and not args.overwrite:
        print(f"File {output_path} exists already. nothing to do. Exiting")
        exit(1)
    return yaml_path, h5_root, output_path

def main(args) -> None:
    """Run interactive H5 manifest creation."""
    print("=" * 72)
    print(
        "BraTS 2020 H5 Dataset Manifest Builder"
    )
    print("=" * 72)



    try:
        yaml_path, h5_root, output_path = get_folders(args)

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
            f"Expected subjects : "
            f"{registered['subject_count']:,}"
        )

        print(
            f"Expected H5 files : "
            f"{registered['expected_h5_count']:,}"
        )


        print(
            "\nSelected H5 dataset directory:"
        )
        print(
            h5_root
        )

        h5_paths = validate_h5_directory(
            h5_root=h5_root,
            registered=registered,
        )

        print(
            "\nH5 filename grid validated."
        )

        print(
            "\nChoose where to save manifest.csv."
        )

        summary = write_manifest(
            output_path=output_path,
            h5_paths=h5_paths,
            registered=registered,
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
            "DATASET MANIFEST CREATION FAILED"
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
        "DATASET MANIFEST CREATION COMPLETE"
    )

    print("=" * 72)

    print(
        f"Manifest           : {output_path}"
    )

    print(
        f"Rows               : {summary['rows']:,}"
    )

    print(
        f"Tumor slices       : {summary['tumor_slices']:,}"
    )

    print(
        f"Non-tumor slices   : {summary['non_tumor_slices']:,}"
    )

    print(
        f"Label 0 pixels     : "
        f"{summary['total_label0_pixels']:,}"
    )

    print(
        f"Label 1 pixels     : "
        f"{summary['total_label1_pixels']:,}"
    )

    print(
        f"Label 2 pixels     : "
        f"{summary['total_label2_pixels']:,}"
    )

    print(
        "\nDownstream workflows can now use manifest.csv instead of "
        "recomputing slice-level tumor metadata."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml_dataset_path", type=str, default=None)
    parser.add_argument("--h5_files_path", type=str, default=None)
    parser.add_argument("--manifest_path", default=None)
    parser.add_argument("--folders_file", type=str, default="./data/folders.yaml")
    parser.add_argument("--overwrite", action='store_true')
    args = parser.parse_args()
    main(args)
