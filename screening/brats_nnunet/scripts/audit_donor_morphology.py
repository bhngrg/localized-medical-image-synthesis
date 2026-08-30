#!/usr/bin/env python3
"""
Audit morphology of the training-donor pool used for external BR-LoRA evaluation.

This script is descriptive only.

It consumes the donors already retained by the external pair-space audit and
computes 2-D whole-tumor morphology from the same binary mask semantics used by
the synthesis pipeline:

    whole_tumor = h5["mask"][:].max(axis=-1) > 0

No donor selection, external-base selection, donor assignment, or BR-LoRA
inference is performed.

Primary outputs
---------------
donor_morphology.csv
donor_volume_summary.csv
donor_morphology_summary.json

figures/
    lesion_area_distribution.png
    lesion_area_fraction_distribution.png
    bounding_box_area_distribution.png
    lesion_centroid_distribution.png
    connected_component_distribution.png
    largest_component_fraction_distribution.png
    lesion_area_vs_compatible_bases.png
    bounding_box_area_vs_compatible_bases.png
    component_count_vs_compatible_bases.png
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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


IMAGE_HEIGHT = 240
IMAGE_WIDTH = 240
IMAGE_AREA = IMAGE_HEIGHT * IMAGE_WIDTH

OUTPUT_TABLE_NAME = "donor_morphology.csv"
OUTPUT_VOLUME_SUMMARY_NAME = "donor_volume_summary.csv"
OUTPUT_SUMMARY_NAME = "donor_morphology_summary.json"

FIGURE_DIR_NAME = "figures"

FIGURE_NAMES = (
    "lesion_area_distribution.png",
    "lesion_area_fraction_distribution.png",
    "bounding_box_area_distribution.png",
    "lesion_centroid_distribution.png",
    "connected_component_distribution.png",
    "largest_component_fraction_distribution.png",
    "lesion_area_vs_compatible_bases.png",
    "bounding_box_area_vs_compatible_bases.png",
    "component_count_vs_compatible_bases.png",
)


class DonorMorphologyAuditError(
    RuntimeError
):
    """Raised when donor-morphology auditing fails."""


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Audit morphology of margin-valid BraTS training donors "
            "used in the external BR-LoRA compatibility space."
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
        "--donor-counts-csv",
        type=Path,
        default=None,
        help=(
            "external_donor_compatibility_counts.csv from the completed "
            "external pair-space audit. If omitted, uses "
            "<nnunet_run_root>/external_pair_space_audit/"
            "external_donor_compatibility_counts.csv."
        ),
    )

    parser.add_argument(
        "--pair-space-summary",
        type=Path,
        default=None,
        help=(
            "external_pair_space_summary.json from the completed "
            "external pair-space audit. If omitted, uses "
            "<nnunet_run_root>/external_pair_space_audit/"
            "external_pair_space_summary.json."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory for donor morphology tables, summary, and figures. "
            "If omitted, uses <nnunet_run_root>/donor_morphology_audit."
        ),
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Figure resolution. Default: 300.",
    )

    return parser.parse_args()


def resolve_existing_file(
    path: Path,
    *,
    name: str,
) -> Path:
    """Resolve and require one existing file."""

    resolved = path.expanduser().resolve()

    if not resolved.is_file():
        raise FileNotFoundError(
            f"{name} does not exist:\n{resolved}"
        )

    return resolved


def sha256_file(
    path: Path,
) -> str:
    """Return SHA-256 hash of one file."""

    digest = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

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

    return result.stdout.strip() == ""


def require_columns(
    table: pd.DataFrame,
    required: set[str],
    *,
    name: str,
) -> None:
    """Require table columns."""

    missing = sorted(
        required
        - set(table.columns)
    )

    if missing:
        raise DonorMorphologyAuditError(
            f"{name} is missing required column(s): "
            + ", ".join(missing)
        )


def refuse_existing_outputs(
    output_dir: Path,
) -> None:
    """Refuse to overwrite existing donor-morphology outputs."""

    figure_dir = (
        output_dir
        / FIGURE_DIR_NAME
    )

    expected = [
        output_dir
        / OUTPUT_TABLE_NAME,

        output_dir
        / OUTPUT_VOLUME_SUMMARY_NAME,

        output_dir
        / OUTPUT_SUMMARY_NAME,
    ]

    expected.extend(
        figure_dir
        / name
        for name in FIGURE_NAMES
    )

    existing = [
        path
        for path in expected
        if path.exists()
    ]

    if existing:
        raise DonorMorphologyAuditError(
            "Refusing to overwrite existing donor-morphology output(s):\n"
            + "\n".join(
                str(path)
                for path in existing
            )
        )


def describe_numeric(
    values: pd.Series | np.ndarray,
) -> dict[str, float | int]:
    """Return standard descriptive statistics."""

    array = np.asarray(
        values,
        dtype=np.float64,
    )

    array = array[
        np.isfinite(array)
    ]

    if array.size == 0:
        raise DonorMorphologyAuditError(
            "Cannot summarize an empty numeric vector."
        )

    return {
        "count":
            int(array.size),

        "minimum":
            float(array.min()),

        "q1":
            float(
                np.quantile(
                    array,
                    0.25,
                )
            ),

        "median":
            float(
                np.median(array)
            ),

        "mean":
            float(array.mean()),

        "standard_deviation":
            float(
                array.std(ddof=1)
                if array.size > 1
                else 0.0
            ),

        "q3":
            float(
                np.quantile(
                    array,
                    0.75,
                )
            ),

        "maximum":
            float(array.max()),

        "p05":
            float(
                np.quantile(
                    array,
                    0.05,
                )
            ),

        "p95":
            float(
                np.quantile(
                    array,
                    0.95,
                )
            ),
    }


def load_binary_whole_tumor_mask(
    path: Path,
) -> np.ndarray:
    """Load one H5 mask and collapse it to the synthesis whole-tumor mask."""

    if not path.is_file():
        raise FileNotFoundError(
            f"Donor H5 file does not exist:\n{path}"
        )

    try:
        with h5py.File(
            path,
            "r",
        ) as file:
            if "mask" not in file:
                raise DonorMorphologyAuditError(
                    "Donor H5 file is missing dataset 'mask'.\n"
                    f"{path}"
                )

            raw = file[
                "mask"
            ][:]

    except DonorMorphologyAuditError:
        raise

    except Exception as exc:
        raise DonorMorphologyAuditError(
            "Unable to load donor H5 mask.\n"
            f"File:\n{path}\n"
            f"Error:\n{exc}"
        ) from exc

    if raw.shape != (
        IMAGE_HEIGHT,
        IMAGE_WIDTH,
        3,
    ):
        raise DonorMorphologyAuditError(
            "Unexpected donor mask shape.\n"
            f"File: {path}\n"
            f"Expected: {(IMAGE_HEIGHT, IMAGE_WIDTH, 3)}\n"
            f"Observed: {raw.shape}"
        )

    if not np.isin(
        np.unique(raw),
        [
            0,
            1,
        ],
    ).all():
        raise DonorMorphologyAuditError(
            "Donor mask contains values outside {0, 1}.\n"
            f"File: {path}"
        )

    whole = (
        raw.max(axis=-1)
        > 0
    )

    return whole


def connected_components_8(
    mask: np.ndarray,
) -> list[int]:
    """
    Return areas of all 8-connected components in one binary 2-D mask.

    A small explicit flood-fill is used so this audit has no SciPy dependency.
    """

    if mask.shape != (
        IMAGE_HEIGHT,
        IMAGE_WIDTH,
    ):
        raise DonorMorphologyAuditError(
            "Connected-component mask has an unexpected shape."
        )

    visited = np.zeros_like(
        mask,
        dtype=bool,
    )

    component_areas: list[int] = []

    offsets = (
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    )

    positive_coordinates = np.argwhere(
        mask
    )

    for y_value, x_value in positive_coordinates:
        y = int(y_value)
        x = int(x_value)

        if visited[
            y,
            x,
        ]:
            continue

        stack = [
            (
                y,
                x,
            )
        ]

        visited[
            y,
            x,
        ] = True

        area = 0

        while stack:
            current_y, current_x = stack.pop()

            area += 1

            for delta_y, delta_x in offsets:
                next_y = (
                    current_y
                    + delta_y
                )

                next_x = (
                    current_x
                    + delta_x
                )

                if not (
                    0
                    <= next_y
                    < IMAGE_HEIGHT
                    and 0
                    <= next_x
                    < IMAGE_WIDTH
                ):
                    continue

                if (
                    mask[
                        next_y,
                        next_x,
                    ]
                    and not visited[
                        next_y,
                        next_x,
                    ]
                ):
                    visited[
                        next_y,
                        next_x,
                    ] = True

                    stack.append(
                        (
                            next_y,
                            next_x,
                        )
                    )

        component_areas.append(
            area
        )

    return component_areas


def classify_centroid_laterality(
    centroid_x_normalized: float,
) -> str:
    """
    Classify centroid location descriptively.

    The central 10% of image width is called midline.
    This is descriptive only and is not a selection criterion.
    """

    if centroid_x_normalized < 0.45:
        return "left"

    if centroid_x_normalized > 0.55:
        return "right"

    return "midline"


def compute_morphology(
    mask: np.ndarray,
) -> dict[str, Any]:
    """Compute morphology for one binary whole-tumor mask."""

    tumor_pixels = int(
        mask.sum()
    )

    if tumor_pixels <= 0:
        raise DonorMorphologyAuditError(
            "Donor whole-tumor mask is empty."
        )

    coordinates = np.argwhere(
        mask
    )

    y_values = coordinates[
        :,
        0,
    ]

    x_values = coordinates[
        :,
        1,
    ]

    y_min = int(
        y_values.min()
    )

    y_max = int(
        y_values.max()
    )

    x_min = int(
        x_values.min()
    )

    x_max = int(
        x_values.max()
    )

    bbox_height = (
        y_max
        - y_min
        + 1
    )

    bbox_width = (
        x_max
        - x_min
        + 1
    )

    bbox_area = int(
        bbox_height
        * bbox_width
    )

    centroid_y = float(
        y_values.mean()
    )

    centroid_x = float(
        x_values.mean()
    )

    centroid_y_normalized = (
        centroid_y
        / float(
            IMAGE_HEIGHT
            - 1
        )
    )

    centroid_x_normalized = (
        centroid_x
        / float(
            IMAGE_WIDTH
            - 1
        )
    )

    component_areas = connected_components_8(
        mask
    )

    component_count = len(
        component_areas
    )

    largest_component_pixels = int(
        max(
            component_areas
        )
    )

    largest_component_fraction = float(
        largest_component_pixels
        / tumor_pixels
    )

    return {
        "tumor_area_pixels":
            tumor_pixels,

        "tumor_area_fraction":
            float(
                tumor_pixels
                / IMAGE_AREA
            ),

        "bbox_y_min":
            y_min,

        "bbox_y_max":
            y_max,

        "bbox_x_min":
            x_min,

        "bbox_x_max":
            x_max,

        "bbox_height":
            bbox_height,

        "bbox_width":
            bbox_width,

        "bbox_area":
            bbox_area,

        "bbox_fill_fraction":
            float(
                tumor_pixels
                / bbox_area
            ),

        "centroid_y":
            centroid_y,

        "centroid_x":
            centroid_x,

        "centroid_y_normalized":
            centroid_y_normalized,

        "centroid_x_normalized":
            centroid_x_normalized,

        "centroid_laterality":
            classify_centroid_laterality(
                centroid_x_normalized
            ),

        "connected_component_count":
            component_count,

        "largest_component_pixels":
            largest_component_pixels,

        "largest_component_fraction":
            largest_component_fraction,
    }


def build_volume_summary(
    donor_morphology: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize donor slices by original BraTS training volume."""

    result = (
        donor_morphology
        .groupby(
            "volume",
            as_index=False,
        )
        .agg(
            donor_slice_count=(
                "slice_path",
                "size",
            ),

            minimum_donor_slice=(
                "slice",
                "min",
            ),

            maximum_donor_slice=(
                "slice",
                "max",
            ),

            mean_tumor_area_pixels=(
                "tumor_area_pixels",
                "mean",
            ),

            median_tumor_area_pixels=(
                "tumor_area_pixels",
                "median",
            ),

            mean_component_count=(
                "connected_component_count",
                "mean",
            ),

            max_component_count=(
                "connected_component_count",
                "max",
            ),

            mean_compatible_external_bases=(
                "compatible_external_base_count",
                "mean",
            ),

            median_compatible_external_bases=(
                "compatible_external_base_count",
                "median",
            ),
        )
        .sort_values(
            "volume",
            kind="stable",
        )
        .reset_index(
            drop=True
        )
    )

    return result


def save_histogram(
    values: pd.Series,
    *,
    xlabel: str,
    title: str,
    path: Path,
    dpi: int,
    bins: int = 50,
) -> None:
    """Save one histogram."""

    figure, axis = plt.subplots(
        figsize=(
            8,
            5,
        )
    )

    axis.hist(
        values,
        bins=bins,
    )

    axis.set_xlabel(
        xlabel
    )

    axis.set_ylabel(
        "Donor slice count"
    )

    axis.set_title(
        title
    )

    figure.tight_layout()

    figure.savefig(
        path,
        dpi=dpi,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )


def save_scatter(
    *,
    x: pd.Series,
    y: pd.Series,
    xlabel: str,
    ylabel: str,
    title: str,
    path: Path,
    dpi: int,
) -> None:
    """Save one scatterplot."""

    figure, axis = plt.subplots(
        figsize=(
            8,
            5,
        )
    )

    axis.scatter(
        x,
        y,
        s=8,
        alpha=0.35,
    )

    axis.set_xlabel(
        xlabel
    )

    axis.set_ylabel(
        ylabel
    )

    axis.set_title(
        title
    )

    figure.tight_layout()

    figure.savefig(
        path,
        dpi=dpi,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )


def save_centroid_plot(
    table: pd.DataFrame,
    path: Path,
    *,
    dpi: int,
) -> None:
    """Plot donor lesion centroids in normalized image coordinates."""

    figure, axis = plt.subplots(
        figsize=(
            7,
            7,
        )
    )

    axis.scatter(
        table[
            "centroid_x_normalized"
        ],
        table[
            "centroid_y_normalized"
        ],
        s=8,
        alpha=0.3,
    )

    axis.set_xlim(
        0,
        1,
    )

    axis.set_ylim(
        1,
        0,
    )

    axis.set_xlabel(
        "Normalized centroid x"
    )

    axis.set_ylabel(
        "Normalized centroid y"
    )

    axis.set_title(
        "Whole-Tumor Centroid Distribution"
    )

    axis.set_aspect(
        "equal",
        adjustable="box",
    )

    figure.tight_layout()

    figure.savefig(
        path,
        dpi=dpi,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )


def main() -> None:
    """Run donor-morphology audit."""

    args = parse_args()

    folders_config = load_folders_config(
        args.folders_file
    )

    nnunet_run_root = None

    if (
        args.donor_counts_csv is None
        or args.pair_space_summary is None
        or args.output_dir is None
    ):
        nnunet_run_root = resolve_path(
            key="nnunet_run_root",
            cli_value=None,
            config=folders_config,
            selector=None,
        ).expanduser().resolve()

    if args.donor_counts_csv is not None:
        donor_counts_path = resolve_existing_file(
            args.donor_counts_csv,
            name="Donor compatibility CSV",
        )
    else:
        donor_counts_path = resolve_existing_file(
            nnunet_run_root
            / "external_pair_space_audit"
            / "external_donor_compatibility_counts.csv",
            name="Donor compatibility CSV",
        )

    if args.pair_space_summary is not None:
        pair_space_summary_path = resolve_existing_file(
            args.pair_space_summary,
            name="Pair-space summary JSON",
        )
    else:
        pair_space_summary_path = resolve_existing_file(
            nnunet_run_root
            / "external_pair_space_audit"
            / "external_pair_space_summary.json",
            name="Pair-space summary JSON",
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
            / "donor_morphology_audit"
        )

    save_folders_config(
        args.folders_file,
        folders_config,
    )

    if args.dpi <= 0:
        raise DonorMorphologyAuditError(
            "--dpi must be positive."
        )

    donors = pd.read_csv(
        donor_counts_path
    )

    require_columns(
        donors,
        {
            "slice_path",
            "target",
            "volume",
            "slice",
            "label0_pxl_cnt",
            "label1_pxl_cnt",
            "label2_pxl_cnt",
            "background_ratio",
            "whole_tumor_pixels",
            "donor_h5_path",
            "mask_has_margin",
            "loaded_mask_pixels",
            "compatible_external_base_count",
        },
        name="Donor compatibility CSV",
    )

    with pair_space_summary_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        pair_space_summary = json.load(
            file
        )

    expected_donor_count = int(
        pair_space_summary[
            "training_donors_after_margin"
        ]
    )

    if len(donors) != expected_donor_count:
        raise DonorMorphologyAuditError(
            "Donor count disagrees with pair-space summary.\n"
            f"Observed: {len(donors)}\n"
            f"Expected: {expected_donor_count}"
        )

    if not (
        donors[
            "target"
        ]
        == 1
    ).all():
        raise DonorMorphologyAuditError(
            "Donor table contains a non-target row."
        )

    if not donors[
        "mask_has_margin"
    ].astype(
        bool
    ).all():
        raise DonorMorphologyAuditError(
            "Donor table unexpectedly contains a mask that failed "
            "the configured margin rule."
        )

    if (
        donors[
            "whole_tumor_pixels"
        ]
        < 300
    ).any():
        raise DonorMorphologyAuditError(
            "Donor table contains a whole-tumor mask below 300 pixels."
        )

    if not (
        donors[
            "whole_tumor_pixels"
        ]
        == donors[
            "loaded_mask_pixels"
        ]
    ).all():
        raise DonorMorphologyAuditError(
            "Manifest whole-tumor counts and loaded-mask counts disagree."
        )

    if (
        donors[
            "compatible_external_base_count"
        ]
        <= 0
    ).any():
        raise DonorMorphologyAuditError(
            "Donor table contains a donor with no compatible external base."
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure_dir = (
        output_dir
        / FIGURE_DIR_NAME
    )

    figure_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    refuse_existing_outputs(
        output_dir
    )

    print()
    print(
        "=" * 78
    )

    print(
        "DONOR MORPHOLOGY AUDIT"
    )

    print(
        "=" * 78
    )

    print(
        f"Donor slices             : "
        f"{len(donors):,}"
    )

    print(
        "Whole-tumor definition   : "
        "mask.max(axis=-1) > 0"
    )

    print(
        "Minimum donor pixels     : 300"
    )

    print(
        "Margin-valid donors only : True"
    )

    print(
        "=" * 78
    )

    rows: list[
        dict[str, Any]
    ] = []

    for index, row in donors.iterrows():
        donor_path = (
            Path(
                str(
                    row[
                        "donor_h5_path"
                    ]
                )
            )
            .expanduser()
            .resolve()
        )

        mask = load_binary_whole_tumor_mask(
            donor_path
        )

        morphology = compute_morphology(
            mask
        )

        loaded_pixels = int(
            morphology[
                "tumor_area_pixels"
            ]
        )

        expected_pixels = int(
            row[
                "loaded_mask_pixels"
            ]
        )

        if loaded_pixels != expected_pixels:
            raise DonorMorphologyAuditError(
                "Morphology-loaded tumor area disagrees with "
                "pair-space donor table.\n"
                f"Donor: {row['slice_path']}\n"
                f"Morphology: {loaded_pixels}\n"
                f"Pair-space table: {expected_pixels}"
            )

        output_row = row.to_dict()

        output_row.update(
            morphology
        )

        rows.append(
            output_row
        )

        completed = (
            index
            + 1
        )

        if (
            completed % 1000 == 0
            or completed == len(donors)
        ):
            print(
                f"  Donors audited: "
                f"{completed:,} / "
                f"{len(donors):,}",
                flush=True,
            )

    morphology_table = pd.DataFrame(
        rows
    )

    if len(morphology_table) != len(
        donors
    ):
        raise DonorMorphologyAuditError(
            "Unexpected donor morphology row count."
        )

    donor_morphology_path = (
        output_dir
        / OUTPUT_TABLE_NAME
    )

    donor_volume_summary_path = (
        output_dir
        / OUTPUT_VOLUME_SUMMARY_NAME
    )

    donor_summary_path = (
        output_dir
        / OUTPUT_SUMMARY_NAME
    )

    morphology_table.to_csv(
        donor_morphology_path,
        index=False,
    )

    volume_summary = build_volume_summary(
        morphology_table
    )

    volume_summary.to_csv(
        donor_volume_summary_path,
        index=False,
    )

    save_histogram(
        morphology_table[
            "tumor_area_pixels"
        ],
        xlabel="Whole-tumor area (pixels)",
        title="Donor Whole-Tumor Area Distribution",
        path=(
            figure_dir
            / "lesion_area_distribution.png"
        ),
        dpi=args.dpi,
    )

    save_histogram(
        morphology_table[
            "tumor_area_fraction"
        ],
        xlabel="Whole-tumor slice-area fraction",
        title="Donor Whole-Tumor Area Fraction",
        path=(
            figure_dir
            / "lesion_area_fraction_distribution.png"
        ),
        dpi=args.dpi,
    )

    save_histogram(
        morphology_table[
            "bbox_area"
        ],
        xlabel="Tumor bounding-box area (pixels)",
        title="Donor Tumor Bounding-Box Area",
        path=(
            figure_dir
            / "bounding_box_area_distribution.png"
        ),
        dpi=args.dpi,
    )

    save_centroid_plot(
        morphology_table,
        figure_dir
        / "lesion_centroid_distribution.png",
        dpi=args.dpi,
    )

    save_histogram(
        morphology_table[
            "connected_component_count"
        ],
        xlabel="8-connected tumor component count",
        title="Donor Tumor Connected Components",
        path=(
            figure_dir
            / "connected_component_distribution.png"
        ),
        dpi=args.dpi,
        bins=min(
            50,
            max(
                10,
                int(
                    morphology_table[
                        "connected_component_count"
                    ].max()
                ),
            ),
        ),
    )

    save_histogram(
        morphology_table[
            "largest_component_fraction"
        ],
        xlabel="Largest-component fraction of tumor area",
        title="Largest Connected-Component Fraction",
        path=(
            figure_dir
            / "largest_component_fraction_distribution.png"
        ),
        dpi=args.dpi,
    )

    save_scatter(
        x=morphology_table[
            "tumor_area_pixels"
        ],
        y=morphology_table[
            "compatible_external_base_count"
        ],
        xlabel="Whole-tumor area (pixels)",
        ylabel="Compatible external bases",
        title="Tumor Area and External-Base Compatibility",
        path=(
            figure_dir
            / "lesion_area_vs_compatible_bases.png"
        ),
        dpi=args.dpi,
    )

    save_scatter(
        x=morphology_table[
            "bbox_area"
        ],
        y=morphology_table[
            "compatible_external_base_count"
        ],
        xlabel="Tumor bounding-box area (pixels)",
        ylabel="Compatible external bases",
        title="Bounding-Box Area and External-Base Compatibility",
        path=(
            figure_dir
            / "bounding_box_area_vs_compatible_bases.png"
        ),
        dpi=args.dpi,
    )

    save_scatter(
        x=morphology_table[
            "connected_component_count"
        ],
        y=morphology_table[
            "compatible_external_base_count"
        ],
        xlabel="8-connected tumor component count",
        ylabel="Compatible external bases",
        title="Tumor Components and External-Base Compatibility",
        path=(
            figure_dir
            / "component_count_vs_compatible_bases.png"
        ),
        dpi=args.dpi,
    )

    laterality_counts = (
        morphology_table[
            "centroid_laterality"
        ]
        .value_counts()
        .sort_index()
    )

    summary = {
        "audit_role":
            (
                "Descriptive morphology audit of the margin-valid training "
                "donor pool retained by the external pair-space audit. "
                "No donor selection or assignment is performed."
            ),

        "whole_tumor_definition":
            "mask.max(axis=-1) > 0",

        "donor_count":
            int(
                len(
                    morphology_table
                )
            ),

        "training_volume_count":
            int(
                morphology_table[
                    "volume"
                ].nunique()
            ),

        "slice_index":
            describe_numeric(
                morphology_table[
                    "slice"
                ]
            ),

        "tumor_area_pixels":
            describe_numeric(
                morphology_table[
                    "tumor_area_pixels"
                ]
            ),

        "tumor_area_fraction":
            describe_numeric(
                morphology_table[
                    "tumor_area_fraction"
                ]
            ),

        "bounding_box_width":
            describe_numeric(
                morphology_table[
                    "bbox_width"
                ]
            ),

        "bounding_box_height":
            describe_numeric(
                morphology_table[
                    "bbox_height"
                ]
            ),

        "bounding_box_area":
            describe_numeric(
                morphology_table[
                    "bbox_area"
                ]
            ),

        "bounding_box_fill_fraction":
            describe_numeric(
                morphology_table[
                    "bbox_fill_fraction"
                ]
            ),

        "centroid_x_normalized":
            describe_numeric(
                morphology_table[
                    "centroid_x_normalized"
                ]
            ),

        "centroid_y_normalized":
            describe_numeric(
                morphology_table[
                    "centroid_y_normalized"
                ]
            ),

        "centroid_laterality_counts":
            {
                str(key):
                    int(value)
                for key, value in (
                    laterality_counts
                    .items()
                )
            },

        "connected_component_count":
            describe_numeric(
                morphology_table[
                    "connected_component_count"
                ]
            ),

        "largest_component_fraction":
            describe_numeric(
                morphology_table[
                    "largest_component_fraction"
                ]
            ),

        "compatible_external_base_count":
            describe_numeric(
                morphology_table[
                    "compatible_external_base_count"
                ]
            ),

        "selection_status": {
            "donors_selected":
                False,

            "donors_assigned_to_external_bases":
                False,

            "definitive_external_manifest_created":
                False,
        },

        "source_artifacts": {
            "donor_counts_csv":
                str(
                    donor_counts_path
                ),

            "pair_space_summary":
                str(
                    pair_space_summary_path
                ),
        },

        "output_artifacts": {
            "donor_morphology_csv":
                str(
                    donor_morphology_path
                ),

            "donor_volume_summary_csv":
                str(
                    donor_volume_summary_path
                ),

            "figure_directory":
                str(
                    figure_dir
                ),
        },

        "provenance": {
            "git_commit":
                resolve_git_commit(),

            "git_worktree_clean":
                resolve_git_worktree_clean(),

            "script_path":
                str(
                    Path(
                        __file__
                    ).resolve()
                ),

            "script_sha256":
                sha256_file(
                    Path(
                        __file__
                    ).resolve()
                ),

            "donor_counts_csv_sha256":
                sha256_file(
                    donor_counts_path
                ),

            "pair_space_summary_sha256":
                sha256_file(
                    pair_space_summary_path
                ),

            "audited_at_utc":
                datetime.now(
                    timezone.utc
                ).isoformat(),
        },
    }

    with donor_summary_path.open(
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
        "DONOR MORPHOLOGY AUDIT: PASS"
    )

    print(
        "=" * 78
    )

    print(
        f"Donors audited           : "
        f"{len(morphology_table):,}"
    )

    print(
        f"Training volumes         : "
        f"{morphology_table['volume'].nunique():,}"
    )

    print(
        f"Morphology table         : "
        f"{donor_morphology_path}"
    )

    print(
        f"Volume summary           : "
        f"{donor_volume_summary_path}"
    )

    print(
        f"Audit summary            : "
        f"{donor_summary_path}"
    )

    print(
        f"Figures                  : "
        f"{figure_dir}"
    )

    print()
    print(
        "No donors were selected."
    )

    print(
        "No donor-base assignments were made."
    )

    print(
        "=" * 78
    )


if __name__ == "__main__":
    main()
