#!/usr/bin/env python3

"""
Analyze geometry and topology of baseline-training tumor masks.

This script implements the baseline mask-characterization stage of the
baseline mask-sensitivity analysis. It reproduces the full-training baseline
slice-selection contract and computes geometry, topology, and
distance-transform persistent-homology descriptors from the exact whole-tumor
mask returned by ``load_h5_full()``.

Persistent homology uses the negative interior Euclidean distance transform as
a vertex filtration. Deep tumor-interior locations therefore enter first, with
progressively more peripheral locations entering as the filtration approaches
zero. Positive finite H0 and H1 intervals are summarized separately; the
essential H0 interval of the full rectangular cubical complex is excluded.

No model inference or mask perturbation is performed here.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import sys
from typing import Optional

import gudhi
import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt
from skimage.measure import (
    euler_number,
    label,
    perimeter_crofton,
)

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
    / "results"
    / "baseline_mask_sensitivity"
    / "mask_geometry_topology"
    / "mask_geometry_topology_catalog.csv"
)

MIN_TUMOR_PIXELS = 300
EXPECTED_ELIGIBLE_SLICES = 19_941
TOPOLOGY_CONNECTIVITY = 2
CROFTON_DIRECTIONS = 4
PERSISTENCE_HOMOLOGY_COEFF_FIELD = 2
PERSISTENCE_TOLERANCE = 1e-12

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
    "mask_pixels",
    "perimeter_crofton",
    "perimeter_area_ratio",
    "compactness",
    "beta_0",
    "beta_1",
    "euler_characteristic",
    "max_edt",
    "h0_positive_finite_count",
    "h0_max_persistence",
    "h0_total_persistence",
    "h1_positive_finite_count",
    "h1_max_persistence",
    "h1_total_persistence",
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
            "Compute geometry and topology descriptors for the exact "
            "whole-tumor masks used by baseline training."
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

    return parser.parse_args()


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


def summarize_positive_finite_persistence(
    intervals: np.ndarray,
) -> tuple[int, float, float]:
    """
    Summarize positive finite persistence intervals.

    Essential intervals are excluded. Finite intervals with persistence at
    or below PERSISTENCE_TOLERANCE are treated as zero-persistence features.
    """
    if intervals.ndim != 2 or intervals.shape[1] != 2:
        raise RuntimeError(
            "GUDHI returned persistence intervals with an unexpected shape: "
            f"{intervals.shape}"
        )

    if len(intervals) == 0:
        return 0, 0.0, 0.0

    finite = intervals[
        np.isfinite(
            intervals[:, 1]
        )
    ]

    if len(finite) == 0:
        return 0, 0.0, 0.0

    persistence = (
        finite[:, 1]
        - finite[:, 0]
    )

    if np.any(
        persistence < -PERSISTENCE_TOLERANCE
    ):
        raise RuntimeError(
            "GUDHI returned a finite persistence interval with death "
            "strictly below birth."
        )

    positive = persistence[
        persistence > PERSISTENCE_TOLERANCE
    ]

    if len(positive) == 0:
        return 0, 0.0, 0.0

    return (
        int(len(positive)),
        float(positive.max()),
        float(positive.sum()),
    )


def compute_distance_persistence(
    binary_mask: np.ndarray,
) -> dict[str, int | float]:
    """
    Compute distance-transform cubical persistence for one binary tumor mask.

    The filtration is the negative Euclidean distance transform evaluated on
    image vertices. Deep tumor-interior locations therefore enter first, with
    progressively more peripheral locations entering as the filtration
    approaches zero.

    Essential H0 persistence is not included in the finite-bar summaries.
    """
    distance = distance_transform_edt(
        binary_mask
    )

    max_edt = float(
        distance.max()
    )

    if not math.isfinite(max_edt) or max_edt <= 0:
        raise RuntimeError(
            f"Invalid maximum Euclidean distance transform: {max_edt}"
        )

    filtration = -distance

    cubical_complex = gudhi.CubicalComplex(
        vertices=filtration
    )

    cubical_complex.compute_persistence(
        homology_coeff_field=PERSISTENCE_HOMOLOGY_COEFF_FIELD,
        min_persistence=-1.0,
    )

    h0_intervals = cubical_complex.persistence_intervals_in_dimension(
        0
    )

    h1_intervals = cubical_complex.persistence_intervals_in_dimension(
        1
    )

    (
        h0_positive_finite_count,
        h0_max_persistence,
        h0_total_persistence,
    ) = summarize_positive_finite_persistence(
        h0_intervals
    )

    (
        h1_positive_finite_count,
        h1_max_persistence,
        h1_total_persistence,
    ) = summarize_positive_finite_persistence(
        h1_intervals
    )

    h0_essential_count = int(
        np.sum(
            ~np.isfinite(
                h0_intervals[:, 1]
            )
        )
    )

    h1_essential_count = int(
        np.sum(
            ~np.isfinite(
                h1_intervals[:, 1]
            )
        )
    )

    if h0_essential_count != 1:
        raise RuntimeError(
            "Expected exactly one essential H0 interval from the full "
            "rectangular cubical complex, but observed "
            f"{h0_essential_count}."
        )

    if h1_essential_count != 0:
        raise RuntimeError(
            "Expected no essential H1 intervals from the full rectangular "
            "cubical complex, but observed "
            f"{h1_essential_count}."
        )

    return {
        "max_edt": max_edt,
        "h0_positive_finite_count": h0_positive_finite_count,
        "h0_max_persistence": h0_max_persistence,
        "h0_total_persistence": h0_total_persistence,
        "h1_positive_finite_count": h1_positive_finite_count,
        "h1_max_persistence": h1_max_persistence,
        "h1_total_persistence": h1_total_persistence,
    }


def analyze_mask(
    *,
    h5_path: Path,
    selection_pixels: int,
) -> dict[str, int | float]:
    """
    Load and characterize one exact baseline whole-tumor mask.
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

    perimeter = float(
        perimeter_crofton(
            binary_mask,
            directions=CROFTON_DIRECTIONS,
        )
    )

    if not math.isfinite(perimeter) or perimeter <= 0:
        raise RuntimeError(
            f"Invalid Crofton perimeter for {h5_path}: {perimeter}"
        )

    perimeter_area_ratio = (
        perimeter
        / mask_pixels
    )

    compactness = (
        4.0
        * math.pi
        * mask_pixels
        / (perimeter ** 2)
    )

    _, beta_0 = label(
        binary_mask,
        connectivity=TOPOLOGY_CONNECTIVITY,
        return_num=True,
    )

    beta_0 = int(
        beta_0
    )

    euler_characteristic = int(
        euler_number(
            binary_mask,
            connectivity=TOPOLOGY_CONNECTIVITY,
        )
    )

    beta_1 = (
        beta_0
        - euler_characteristic
    )

    if beta_1 < 0:
        raise RuntimeError(
            "Computed beta_1 is negative for "
            f"{h5_path}: {beta_1}"
        )

    if euler_characteristic != beta_0 - beta_1:
        raise RuntimeError(
            "Euler/Betti identity failed for "
            f"{h5_path}."
        )

    persistence_metrics = compute_distance_persistence(
        binary_mask
    )

    return {
        "mask_pixels": mask_pixels,
        "perimeter_crofton": perimeter,
        "perimeter_area_ratio": perimeter_area_ratio,
        "compactness": compactness,
        "beta_0": beta_0,
        "beta_1": beta_1,
        "euler_characteristic": euler_characteristic,
        **persistence_metrics,
    }


def write_catalog(
    rows: list[dict[str, object]],
    output_path: Path,
) -> None:
    """
    Atomically write the completed mask-characterization catalog.
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

    print(
        "============================================================"
    )
    print(
        "BASELINE MASK GEOMETRY / TOPOLOGY"
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
        f"Topology connectivity: {TOPOLOGY_CONNECTIVITY} "
        "(8-connected foreground)"
    )
    print(
        f"Crofton directions:    {CROFTON_DIRECTIONS}"
    )
    print(
        "PH filtration:         negative interior Euclidean distance "
        "on vertices"
    )
    print(
        f"PH coefficient field:  {PERSISTENCE_HOMOLOGY_COEFF_FIELD}"
    )
    print(
        f"PH positive tolerance: {PERSISTENCE_TOLERANCE:g}"
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

        metrics = analyze_mask(
            h5_path=h5_path,
            selection_pixels=selection_pixels,
        )

        result = {
            "volume": volume,
            "slice": slice_index,
            "slice_path": slice_path,
            "selection_pixels": selection_pixels,
            **metrics,
        }

        results.append(
            result
        )

        if args.case:
            print(
                f"volume={volume:3d} "
                f"slice={slice_index:3d} "
                f"area={metrics['mask_pixels']:5d} "
                f"perimeter={metrics['perimeter_crofton']:.3f} "
                f"perim/area={metrics['perimeter_area_ratio']:.5f} "
                f"compactness={metrics['compactness']:.5f} "
                f"beta_0={metrics['beta_0']} "
                f"beta_1={metrics['beta_1']} "
                f"euler={metrics['euler_characteristic']} "
                f"max_edt={metrics['max_edt']:.3f} "
                f"H0+={metrics['h0_positive_finite_count']} "
                f"H0max={metrics['h0_max_persistence']:.3f} "
                f"H1+={metrics['h1_positive_finite_count']} "
                f"H1max={metrics['h1_max_persistence']:.3f}"
            )
        elif (
            (position + 1) % 1000 == 0
            or position + 1 == len(analysis_rows)
        ):
            print(
                f"Processed {position + 1:,} / "
                f"{len(analysis_rows):,}"
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
