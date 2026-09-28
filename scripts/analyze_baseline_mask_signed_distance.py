#!/usr/bin/env python3

"""
Characterize signed-distance perturbations of baseline-training tumor masks.

This script implements the perturbation-geometry stage of the baseline
mask-sensitivity analysis. It reproduces the exact full-training baseline
slice-selection contract and loads the exact whole-tumor mask returned by
``load_h5_full()``.

For each mask, deterministic Euclidean-distance perturbations are constructed
at integer signed radii.

Radius convention
-----------------
radius = 0
    Original whole-tumor mask.

radius < 0
    Erosion. For radius -r, retain foreground pixels whose interior Euclidean
    distance is strictly greater than r. Consequently, radius -1 removes the
    foreground layer with EDT = 1, radius -2 removes foreground pixels with
    EDT <= 2, and so forth.

radius > 0
    Dilation. For radius r, retain the original mask and add background pixels
    whose Euclidean distance to the original foreground is <= r.

The convention is deliberately based on SciPy's Euclidean distance transform,
which is also used by the production inner-only feathering implementation.
This script characterizes perturbation geometry only. It performs no model
inference and does not modify scientific artifacts.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import sys
from typing import Optional

import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )

from src.config import (
    load_folders_config,
    resolve_path,
)
from src.data.preprocessing import load_h5_full


DEFAULT_FOLDERS_FILE = PROJECT_ROOT / "data" / "folders.yaml"
DEFAULT_OUTPUT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "baseline_mask_sensitivity"
    / "signed_distance"
    / "signed_distance_perturbation_catalog.csv"
)

MIN_TUMOR_PIXELS = 300
EXPECTED_ELIGIBLE_SLICES = 19_941

DEFAULT_MIN_RADIUS = -8
DEFAULT_MAX_RADIUS = 8

REQUIRED_MANIFEST_COLUMNS = {
    "slice_path",
    "volume",
    "slice",
    "label0_pxl_cnt",
    "label1_pxl_cnt",
    "label2_pxl_cnt",
}

OUTPUT_FIELDNAMES = [
    "volume",
    "slice",
    "slice_path",
    "selection_pixels",
    "original_mask_pixels",
    "original_max_edt",
    "radius_pixels",
    "perturbation",
    "perturbed_mask_pixels",
    "pixel_delta",
    "relative_area",
    "fractional_area_change",
    "retained_original_pixels",
    "retained_original_fraction",
    "removed_original_pixels",
    "removed_original_fraction",
    "added_pixels",
    "added_fraction_of_original",
    "intersection_pixels",
    "union_pixels",
    "iou_with_original",
    "dice_with_original",
    "is_empty",
]


def parse_case(value: str) -> tuple[int, int]:
    """
    Parse one diagnostic case encoded as VOLUME,SLICE.
    """
    parts = value.split(",")

    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            "Diagnostic cases must use VOLUME,SLICE format."
        )

    try:
        volume = int(parts[0])
        slice_index = int(parts[1])
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Diagnostic volume and slice values must be integers."
        ) from exc

    if volume < 0 or slice_index < 0:
        raise argparse.ArgumentTypeError(
            "Diagnostic volume and slice values must be non-negative."
        )

    return volume, slice_index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Characterize deterministic signed Euclidean-distance "
            "perturbations of the exact whole-tumor masks used by "
            "baseline training."
        )
    )

    parser.add_argument(
        "--folders-file",
        type=Path,
        default=DEFAULT_FOLDERS_FILE,
        help=(
            "Machine-specific folders YAML. "
            f"Default: {DEFAULT_FOLDERS_FILE}"
        ),
    )

    parser.add_argument(
        "--h5-root",
        type=Path,
        default=None,
        help=(
            "Override the H5 dataset root. Otherwise use h5_root from "
            "the folders configuration."
        ),
    )

    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=None,
        help=(
            "Override the BraTS H5 manifest. Otherwise use manifest_path "
            "from the folders configuration."
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=(
            "Output CSV path. "
            f"Default: {DEFAULT_OUTPUT_PATH}"
        ),
    )

    parser.add_argument(
        "--min-radius",
        type=int,
        default=DEFAULT_MIN_RADIUS,
        help=(
            "Minimum signed perturbation radius in pixels. "
            f"Default: {DEFAULT_MIN_RADIUS}"
        ),
    )

    parser.add_argument(
        "--max-radius",
        type=int,
        default=DEFAULT_MAX_RADIUS,
        help=(
            "Maximum signed perturbation radius in pixels. "
            f"Default: {DEFAULT_MAX_RADIUS}"
        ),
    )

    parser.add_argument(
        "--case",
        action="append",
        type=parse_case,
        default=None,
        metavar="VOLUME,SLICE",
        help=(
            "Analyze only one explicit diagnostic volume/slice pair. "
            "May be supplied multiple times."
        ),
    )

    args = parser.parse_args()

    if args.min_radius > 0:
        parser.error(
            "--min-radius must be <= 0."
        )

    if args.max_radius < 0:
        parser.error(
            "--max-radius must be >= 0."
        )

    if args.min_radius > args.max_radius:
        parser.error(
            "--min-radius must not exceed --max-radius."
        )

    return args


def select_baseline_training_slices(
    manifest_path: Path,
) -> pd.DataFrame:
    """
    Reproduce the exact full-training baseline slice-selection rule.
    """
    manifest = pd.read_csv(
        manifest_path
    )

    missing_columns = sorted(
        REQUIRED_MANIFEST_COLUMNS
        - set(manifest.columns)
    )

    if missing_columns:
        raise ValueError(
            "Manifest is missing required column(s): "
            + ", ".join(missing_columns)
        )

    selection_pixels = (
        manifest["label0_pxl_cnt"]
        + manifest["label1_pxl_cnt"]
        + manifest["label2_pxl_cnt"]
    )

    selected = manifest.loc[
        selection_pixels >= MIN_TUMOR_PIXELS
    ].copy()

    selected["selection_pixels"] = selection_pixels.loc[
        selected.index
    ].astype(int)

    selected = selected.reset_index(
        drop=True
    )

    if len(selected) != EXPECTED_ELIGIBLE_SLICES:
        raise RuntimeError(
            "Baseline-training cohort size does not match the verified "
            "full-training contract.\n\n"
            f"Expected: {EXPECTED_ELIGIBLE_SLICES:,}\n"
            f"Observed: {len(selected):,}"
        )

    duplicate_pairs = selected.duplicated(
        subset=[
            "volume",
            "slice",
        ],
        keep=False,
    )

    if duplicate_pairs.any():
        raise RuntimeError(
            "Eligible baseline-training cohort contains duplicate "
            "(volume, slice) pairs."
        )

    return selected


def restrict_to_cases(
    selected: pd.DataFrame,
    cases: Optional[list[tuple[int, int]]],
) -> pd.DataFrame:
    """
    Restrict the verified full cohort to explicitly requested diagnostic cases.
    """
    if not cases:
        return selected

    if len(cases) != len(set(cases)):
        raise ValueError(
            "Duplicate --case arguments were supplied."
        )

    indexed = selected.set_index(
        [
            "volume",
            "slice",
        ],
        drop=False,
    )

    missing = [
        case
        for case in cases
        if case not in indexed.index
    ]

    if missing:
        formatted = "\n".join(
            f"  volume={volume}, slice={slice_index}"
            for volume, slice_index in missing
        )

        raise ValueError(
            "Requested diagnostic case(s) are not members of the verified "
            "baseline-training cohort:\n"
            f"{formatted}"
        )

    rows = [
        indexed.loc[
            case
        ]
        for case in cases
    ]

    return pd.DataFrame(
        rows
    ).reset_index(
        drop=True
    )


def resolve_h5_path(
    h5_root: Path,
    slice_path: str,
) -> Path:
    """
    Resolve one manifest slice_path against the registered H5 root.

    The reconstructed H5 dataset uses the filename from the historical
    manifest path rather than the manifest's original parent directories.
    """
    path = h5_root / Path(
        str(slice_path)
    ).name

    if not path.is_file():
        raise FileNotFoundError(
            "Manifest-selected H5 file does not exist:\n"
            f"{path}"
        )

    return path


def load_verified_mask(
    *,
    h5_path: Path,
    selection_pixels: int,
) -> np.ndarray:
    """
    Load and validate one exact baseline whole-tumor mask.
    """
    _, mask_tensor, _ = load_h5_full(
        h5_path,
        image_channel=0,
    )

    mask_array = mask_tensor.detach().cpu().numpy()

    if mask_array.ndim != 3 or mask_array.shape[0] != 1:
        raise RuntimeError(
            "load_h5_full() returned an unexpected mask shape for "
            f"{h5_path}: {mask_array.shape}"
        )

    mask = mask_array[0]

    unique_values = np.unique(
        mask
    )

    if not np.all(
        np.isin(
            unique_values,
            [0.0, 1.0],
        )
    ):
        raise RuntimeError(
            "load_h5_full() returned a non-binary mask for "
            f"{h5_path}: {unique_values.tolist()}"
        )

    binary_mask = mask.astype(
        bool
    )

    mask_pixels = int(
        binary_mask.sum()
    )

    if mask_pixels <= 0:
        raise RuntimeError(
            f"Eligible baseline mask is empty: {h5_path}"
        )

    if mask_pixels != int(selection_pixels):
        raise RuntimeError(
            "Manifest selection-pixel count disagrees with the exact "
            "whole-tumor mask returned by load_h5_full().\n\n"
            f"Path: {h5_path}\n"
            f"Manifest selection pixels: {selection_pixels}\n"
            f"Actual mask pixels: {mask_pixels}"
        )

    return binary_mask


def perturb_mask(
    *,
    binary_mask: np.ndarray,
    radius: int,
    interior_distance: np.ndarray,
    exterior_distance: np.ndarray,
) -> np.ndarray:
    """
    Construct one deterministic signed Euclidean-distance perturbation.
    """
    if binary_mask.ndim != 2:
        raise ValueError(
            "binary_mask must be two-dimensional."
        )

    if binary_mask.dtype != np.bool_:
        raise TypeError(
            "binary_mask must be a boolean NumPy array."
        )

    if interior_distance.shape != binary_mask.shape:
        raise ValueError(
            "interior_distance shape does not match binary_mask."
        )

    if exterior_distance.shape != binary_mask.shape:
        raise ValueError(
            "exterior_distance shape does not match binary_mask."
        )

    if radius == 0:
        return binary_mask.copy()

    if radius < 0:
        erosion_distance = abs(
            radius
        )

        return (
            binary_mask
            & (
                interior_distance
                > float(
                    erosion_distance
                )
            )
        )

    return (
        binary_mask
        | (
            (~binary_mask)
            & (
                exterior_distance
                <= float(
                    radius
                )
            )
        )
    )


def summarize_perturbation(
    *,
    original_mask: np.ndarray,
    perturbed_mask: np.ndarray,
    radius: int,
    original_max_edt: float,
) -> dict[str, int | float | bool | str]:
    """
    Summarize one perturbed mask relative to its original mask.
    """
    original_pixels = int(
        original_mask.sum()
    )

    perturbed_pixels = int(
        perturbed_mask.sum()
    )

    intersection_pixels = int(
        np.logical_and(
            original_mask,
            perturbed_mask,
        ).sum()
    )

    union_pixels = int(
        np.logical_or(
            original_mask,
            perturbed_mask,
        ).sum()
    )

    retained_original_pixels = intersection_pixels

    removed_original_pixels = (
        original_pixels
        - retained_original_pixels
    )

    added_pixels = int(
        np.logical_and(
            perturbed_mask,
            ~original_mask,
        ).sum()
    )

    pixel_delta = (
        perturbed_pixels
        - original_pixels
    )

    relative_area = (
        perturbed_pixels
        / original_pixels
    )

    fractional_area_change = (
        pixel_delta
        / original_pixels
    )

    retained_original_fraction = (
        retained_original_pixels
        / original_pixels
    )

    removed_original_fraction = (
        removed_original_pixels
        / original_pixels
    )

    added_fraction_of_original = (
        added_pixels
        / original_pixels
    )

    if union_pixels <= 0:
        raise RuntimeError(
            "Original/perturbed mask union is unexpectedly empty."
        )

    iou_with_original = (
        intersection_pixels
        / union_pixels
    )

    dice_denominator = (
        original_pixels
        + perturbed_pixels
    )

    if dice_denominator <= 0:
        raise RuntimeError(
            "Original/perturbed Dice denominator is unexpectedly zero."
        )

    dice_with_original = (
        2.0
        * intersection_pixels
        / dice_denominator
    )

    if radius < 0:
        perturbation = "erosion"
    elif radius > 0:
        perturbation = "dilation"
    else:
        perturbation = "original"

    values_to_check = (
        original_max_edt,
        relative_area,
        fractional_area_change,
        retained_original_fraction,
        removed_original_fraction,
        added_fraction_of_original,
        iou_with_original,
        dice_with_original,
    )

    if not all(
        math.isfinite(
            float(value)
        )
        for value in values_to_check
    ):
        raise RuntimeError(
            "Perturbation summary contains a non-finite value."
        )

    if not (
        0.0
        <= retained_original_fraction
        <= 1.0
    ):
        raise RuntimeError(
            "Invalid retained-original fraction."
        )

    if not (
        0.0
        <= removed_original_fraction
        <= 1.0
    ):
        raise RuntimeError(
            "Invalid removed-original fraction."
        )

    if not (
        0.0
        <= iou_with_original
        <= 1.0
    ):
        raise RuntimeError(
            "Invalid IoU."
        )

    if not (
        0.0
        <= dice_with_original
        <= 1.0
    ):
        raise RuntimeError(
            "Invalid Dice coefficient."
        )

    return {
        "original_mask_pixels": original_pixels,
        "original_max_edt": original_max_edt,
        "radius_pixels": radius,
        "perturbation": perturbation,
        "perturbed_mask_pixels": perturbed_pixels,
        "pixel_delta": pixel_delta,
        "relative_area": relative_area,
        "fractional_area_change": fractional_area_change,
        "retained_original_pixels": retained_original_pixels,
        "retained_original_fraction": retained_original_fraction,
        "removed_original_pixels": removed_original_pixels,
        "removed_original_fraction": removed_original_fraction,
        "added_pixels": added_pixels,
        "added_fraction_of_original": added_fraction_of_original,
        "intersection_pixels": intersection_pixels,
        "union_pixels": union_pixels,
        "iou_with_original": iou_with_original,
        "dice_with_original": dice_with_original,
        "is_empty": bool(
            perturbed_pixels == 0
        ),
    }


def analyze_mask(
    *,
    h5_path: Path,
    selection_pixels: int,
    radii: tuple[int, ...],
) -> list[dict[str, int | float | bool | str]]:
    """
    Load one mask and characterize all requested signed-distance radii.
    """
    binary_mask = load_verified_mask(
        h5_path=h5_path,
        selection_pixels=selection_pixels,
    )

    interior_distance = distance_transform_edt(
        binary_mask
    )

    exterior_distance = distance_transform_edt(
        ~binary_mask
    )

    original_max_edt = float(
        interior_distance.max()
    )

    if (
        not math.isfinite(
            original_max_edt
        )
        or original_max_edt <= 0
    ):
        raise RuntimeError(
            f"Invalid original maximum EDT for {h5_path}: "
            f"{original_max_edt}"
        )

    summaries = []

    previous_erosion_pixels: int | None = None
    previous_dilation_pixels: int | None = None

    for radius in radii:
        perturbed_mask = perturb_mask(
            binary_mask=binary_mask,
            radius=radius,
            interior_distance=interior_distance,
            exterior_distance=exterior_distance,
        )

        summary = summarize_perturbation(
            original_mask=binary_mask,
            perturbed_mask=perturbed_mask,
            radius=radius,
            original_max_edt=original_max_edt,
        )

        perturbed_pixels = int(
            summary[
                "perturbed_mask_pixels"
            ]
        )

        if radius == 0:
            if not np.array_equal(
                perturbed_mask,
                binary_mask,
            ):
                raise RuntimeError(
                    "Radius-zero perturbation does not exactly reproduce "
                    "the original mask."
                )

            if (
                perturbed_pixels
                != int(
                    binary_mask.sum()
                )
            ):
                raise RuntimeError(
                    "Radius-zero perturbation changed mask area."
                )

        if radius < 0:
            if not np.all(
                ~perturbed_mask
                | binary_mask
            ):
                raise RuntimeError(
                    "Erosion introduced pixels outside the original mask."
                )

            if (
                previous_erosion_pixels is not None
                and perturbed_pixels
                < previous_erosion_pixels
            ):
                raise RuntimeError(
                    "Erosion area is not monotone across radii."
                )

            previous_erosion_pixels = perturbed_pixels

        elif radius > 0:
            if not np.all(
                ~binary_mask
                | perturbed_mask
            ):
                raise RuntimeError(
                    "Dilation removed pixels from the original mask."
                )

            if (
                previous_dilation_pixels is not None
                and perturbed_pixels
                < previous_dilation_pixels
            ):
                raise RuntimeError(
                    "Dilation area is not monotone across radii."
                )

            previous_dilation_pixels = perturbed_pixels

        summaries.append(
            summary
        )

    return summaries


def write_catalog(
    rows: list[dict[str, object]],
    output_path: Path,
) -> None:
    """
    Atomically write the completed signed-distance perturbation catalog.
    """
    output_path = output_path.expanduser().resolve()

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = output_path.with_suffix(
        output_path.suffix + ".tmp"
    )

    try:
        with temporary_path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=OUTPUT_FIELDNAMES,
                lineterminator="\n",
            )

            writer.writeheader()
            writer.writerows(
                rows
            )

        temporary_path.replace(
            output_path
        )
    except Exception:
        if temporary_path.exists():
            temporary_path.unlink()
        raise


def main() -> None:
    args = parse_args()

    folders_file = args.folders_file.expanduser().resolve()

    folders_config = load_folders_config(
        folders_file
    )

    h5_root = resolve_path(
        key="h5_root",
        cli_value=args.h5_root,
        config=folders_config,
    ).expanduser().resolve()

    manifest_path = resolve_path(
        key="manifest_path",
        cli_value=args.manifest_path,
        config=folders_config,
    ).expanduser().resolve()

    if not h5_root.is_dir():
        raise ValueError(
            "H5 root does not exist or is not a directory:\n"
            f"{h5_root}"
        )

    if not manifest_path.is_file():
        raise ValueError(
            "Manifest does not exist:\n"
            f"{manifest_path}"
        )

    selected = select_baseline_training_slices(
        manifest_path
    )

    analysis_rows = restrict_to_cases(
        selected,
        args.case,
    )

    radii = tuple(
        range(
            args.min_radius,
            args.max_radius + 1,
        )
    )

    if 0 not in radii:
        raise RuntimeError(
            "Signed-distance radius range must include zero."
        )

    expected_output_rows = (
        len(
            analysis_rows
        )
        * len(
            radii
        )
    )

    print(
        "============================================================"
    )
    print(
        "BASELINE MASK SIGNED-DISTANCE PERTURBATIONS"
    )
    print(
        "============================================================"
    )
    print(
        f"H5 root:               {h5_root}"
    )
    print(
        f"Manifest:              {manifest_path}"
    )
    print(
        f"Verified cohort size:  {len(selected):,}"
    )
    print(
        f"Slices to analyze:     {len(analysis_rows):,}"
    )
    print(
        f"Signed radii:          {args.min_radius:+d} .. "
        f"{args.max_radius:+d} pixels"
    )
    print(
        f"Radii per slice:       {len(radii)}"
    )
    print(
        f"Expected output rows:  {expected_output_rows:,}"
    )
    print(
        "Distance convention:   scipy.ndimage.distance_transform_edt"
    )
    print(
        "Erosion rule:          keep foreground where EDT > |radius|"
    )
    print(
        "Dilation rule:         add background where EDT <= radius"
    )
    print(
        "Model inference:       none"
    )
    print()

    results: list[dict[str, object]] = []

    for position, row in analysis_rows.iterrows():
        volume = int(
            row["volume"]
        )

        slice_index = int(
            row["slice"]
        )

        selection_pixels = int(
            row["selection_pixels"]
        )

        slice_path = str(
            row["slice_path"]
        )

        h5_path = resolve_h5_path(
            h5_root,
            slice_path,
        )

        summaries = analyze_mask(
            h5_path=h5_path,
            selection_pixels=selection_pixels,
            radii=radii,
        )

        if len(summaries) != len(radii):
            raise RuntimeError(
                "Unexpected number of perturbation summaries for "
                f"{h5_path}."
            )

        for summary in summaries:
            result = {
                "volume": volume,
                "slice": slice_index,
                "slice_path": slice_path,
                "selection_pixels": selection_pixels,
                **summary,
            }

            results.append(
                result
            )

            if args.case:
                print(
                    f"volume={volume:3d} "
                    f"slice={slice_index:3d} "
                    f"radius={int(summary['radius_pixels']):+3d} "
                    f"type={str(summary['perturbation']):8s} "
                    f"area={int(summary['perturbed_mask_pixels']):5d} "
                    f"relative_area={float(summary['relative_area']):.4f} "
                    f"retained={float(summary['retained_original_fraction']):.4f} "
                    f"added={float(summary['added_fraction_of_original']):.4f} "
                    f"IoU={float(summary['iou_with_original']):.4f} "
                    f"Dice={float(summary['dice_with_original']):.4f} "
                    f"empty={bool(summary['is_empty'])}"
                )

        if (
            not args.case
            and (
                (position + 1) % 1000 == 0
                or position + 1 == len(analysis_rows)
            )
        ):
            print(
                f"Processed {position + 1:,} / "
                f"{len(analysis_rows):,}"
            )

    if len(results) != expected_output_rows:
        raise RuntimeError(
            "Signed-distance catalog row count does not match expectation.\n"
            f"Expected: {expected_output_rows:,}\n"
            f"Observed: {len(results):,}"
        )

    output_path = args.output.expanduser().resolve()

    write_catalog(
        results,
        output_path,
    )

    print()
    print(
        f"Wrote {len(results):,} rows:"
    )
    print(
        output_path
    )


if __name__ == "__main__":
    main()
