#!/usr/bin/env python3

"""
Summarize the finalized baseline mask geometry/topology catalog.

This script reads the machine-readable catalog produced by
``analyze_baseline_mask_geometry_topology.py`` and writes a deterministic,
human-readable audit and descriptive summary.

No H5 data are read and no geometry, distance transforms, or persistent
homology are recomputed.
"""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_CATALOG_PATH = (
    PROJECT_ROOT
    / "results"
    / "baseline_mask_sensitivity"
    / "mask_geometry_topology"
    / "mask_geometry_topology_catalog.csv"
)

DEFAULT_OUTPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "baseline_mask_sensitivity"
    / "mask_geometry_topology"
    / "mask_geometry_topology_summary.txt"
)

EXPECTED_ROWS = 19_941
EXPECTED_VOLUMES = 369

EXPECTED_COLUMNS = [
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

SUMMARY_COLUMNS = [
    "mask_pixels",
    "perimeter_area_ratio",
    "compactness",
    "beta_0",
    "beta_1",
    "max_edt",
    "h0_positive_finite_count",
    "h0_max_persistence",
    "h0_total_persistence",
    "h1_positive_finite_count",
    "h1_max_persistence",
    "h1_total_persistence",
]

EXTREME_COLUMNS = [
    "volume",
    "slice",
    "mask_pixels",
    "compactness",
    "beta_0",
    "beta_1",
    "max_edt",
    "h0_positive_finite_count",
    "h0_max_persistence",
    "h0_total_persistence",
    "h1_positive_finite_count",
    "h1_max_persistence",
    "h1_total_persistence",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate and summarize the finalized baseline mask "
            "geometry/topology catalog."
        )
    )

    parser.add_argument(
        "--catalog",
        type=Path,
        default=DEFAULT_CATALOG_PATH,
        help=f"Input catalog CSV. Default: {DEFAULT_CATALOG_PATH}",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Output summary text file. Default: {DEFAULT_OUTPUT_PATH}",
    )

    return parser.parse_args()


def load_and_validate_catalog(path: Path) -> pd.DataFrame:
    path = path.expanduser().resolve()

    if not path.is_file():
        raise FileNotFoundError(
            "Catalog does not exist:\n"
            f"{path}"
        )

    df = pd.read_csv(path)

    if list(df.columns) != EXPECTED_COLUMNS:
        raise RuntimeError(
            "Catalog column contract does not match the expected schema."
        )

    if len(df) != EXPECTED_ROWS:
        raise RuntimeError(
            f"Expected {EXPECTED_ROWS:,} rows, observed {len(df):,}."
        )

    if df["volume"].nunique() != EXPECTED_VOLUMES:
        raise RuntimeError(
            f"Expected {EXPECTED_VOLUMES} unique volumes, observed "
            f"{df['volume'].nunique()}."
        )

    if df.duplicated(["volume", "slice"]).any():
        raise RuntimeError(
            "Catalog contains duplicate (volume, slice) pairs."
        )

    if df["slice_path"].duplicated().any():
        raise RuntimeError(
            "Catalog contains duplicate slice_path values."
        )

    if df.isna().any().any():
        raise RuntimeError(
            "Catalog contains missing values."
        )

    numeric = df.drop(
        columns=["slice_path"]
    )

    if not np.isfinite(
        numeric.to_numpy(
            dtype=float
        )
    ).all():
        raise RuntimeError(
            "Catalog contains non-finite numeric values."
        )

    if not (
        df["selection_pixels"]
        == df["mask_pixels"]
    ).all():
        raise RuntimeError(
            "selection_pixels and mask_pixels disagree."
        )

    if (df["beta_0"] < 1).any():
        raise RuntimeError(
            "Catalog contains beta_0 < 1."
        )

    if (df["beta_1"] < 0).any():
        raise RuntimeError(
            "Catalog contains beta_1 < 0."
        )

    if not (
        df["euler_characteristic"]
        == df["beta_0"] - df["beta_1"]
    ).all():
        raise RuntimeError(
            "Euler/Betti identity failed."
        )

    nonnegative_columns = [
        "max_edt",
        "h0_positive_finite_count",
        "h0_max_persistence",
        "h0_total_persistence",
        "h1_positive_finite_count",
        "h1_max_persistence",
        "h1_total_persistence",
    ]

    if (df[nonnegative_columns] < 0).any().any():
        raise RuntimeError(
            "Catalog contains negative PH/EDT summary values."
        )

    if (df["max_edt"] <= 0).any():
        raise RuntimeError(
            "Catalog contains non-positive max_edt."
        )

    if (
        df["h0_max_persistence"]
        > df["h0_total_persistence"] + 1e-12
    ).any():
        raise RuntimeError(
            "H0 maximum persistence exceeds H0 total persistence."
        )

    if (
        df["h1_max_persistence"]
        > df["h1_total_persistence"] + 1e-12
    ).any():
        raise RuntimeError(
            "H1 maximum persistence exceeds H1 total persistence."
        )

    return df


def print_extremes(
    df: pd.DataFrame,
    column: str,
    title: str,
    n: int = 15,
) -> None:
    print()
    print(f"===== {title} =====")
    print(
        df.nlargest(
            n,
            column,
        )[EXTREME_COLUMNS].to_string(
            index=False
        )
    )


def print_summary(df: pd.DataFrame) -> None:
    print("============================================================")
    print("BASELINE MASK GEOMETRY / TOPOLOGY SUMMARY")
    print("============================================================")
    print()

    print("===== BASIC CONTRACT =====")
    print(f"rows:                    {len(df):,}")
    print(f"columns:                 {len(df.columns)}")
    print(f"unique volumes:          {df['volume'].nunique():,}")
    print(
        "duplicate volume/slice:  "
        f"{int(df.duplicated(['volume', 'slice']).sum()):,}"
    )
    print(
        "duplicate slice_path:    "
        f"{int(df['slice_path'].duplicated().sum()):,}"
    )
    print(
        "missing values:          "
        f"{int(df.isna().sum().sum()):,}"
    )

    print()
    print("===== DESCRIPTIVE SUMMARY =====")

    description = df[
        SUMMARY_COLUMNS
    ].describe(
        percentiles=[
            0.01,
            0.025,
            0.25,
            0.50,
            0.75,
            0.975,
            0.99,
        ]
    ).T

    print(
        description.to_string()
    )

    h0_zero = int(
        (
            df["h0_positive_finite_count"] == 0
        ).sum()
    )
    h1_zero = int(
        (
            df["h1_positive_finite_count"] == 0
        ).sum()
    )

    print()
    print("===== PH ZERO / POSITIVE PREVALENCE =====")
    print(
        "H0 positive finite count = 0: "
        f"{h0_zero:,} ({100.0 * h0_zero / len(df):.3f}%)"
    )
    print(
        "H0 positive finite count > 0: "
        f"{len(df) - h0_zero:,} "
        f"({100.0 * (len(df) - h0_zero) / len(df):.3f}%)"
    )
    print(
        "H1 positive finite count = 0: "
        f"{h1_zero:,} ({100.0 * h1_zero / len(df):.3f}%)"
    )
    print(
        "H1 positive finite count > 0: "
        f"{len(df) - h1_zero:,} "
        f"({100.0 * (len(df) - h1_zero) / len(df):.3f}%)"
    )

    print()
    print("===== SIMPLE TOPOLOGY VS PH PRESENCE =====")
    print("beta_0 = 1 / H0+ = 0:")
    print(
        pd.crosstab(
            df["beta_0"] == 1,
            df["h0_positive_finite_count"] == 0,
            margins=True,
        ).to_string()
    )

    print()
    print("beta_1 = 0 / H1+ = 0:")
    print(
        pd.crosstab(
            df["beta_1"] == 0,
            df["h1_positive_finite_count"] == 0,
            margins=True,
        ).to_string()
    )

    print()
    print("===== SPEARMAN CORRELATIONS =====")
    print(
        df[
            SUMMARY_COLUMNS
        ].corr(
            method="spearman"
        ).round(
            3
        ).to_string()
    )

    print_extremes(
        df,
        "h0_positive_finite_count",
        "LARGEST H0 POSITIVE-FINITE COUNTS",
    )

    print_extremes(
        df,
        "h0_max_persistence",
        "LARGEST H0 MAX PERSISTENCE",
    )

    print_extremes(
        df,
        "h1_positive_finite_count",
        "LARGEST H1 POSITIVE-FINITE COUNTS",
    )

    print_extremes(
        df,
        "h1_max_persistence",
        "LARGEST H1 MAX PERSISTENCE",
    )

    print_extremes(
        df,
        "max_edt",
        "LARGEST MAX EDT",
    )

    connected_no_holes_h0 = df[
        (df["beta_0"] == 1)
        & (df["beta_1"] == 0)
        & (df["h0_positive_finite_count"] > 0)
    ]

    no_binary_holes_h1 = df[
        (df["beta_1"] == 0)
        & (df["h1_positive_finite_count"] > 0)
    ]

    binary_holes_no_h1 = df[
        (df["beta_1"] > 0)
        & (df["h1_positive_finite_count"] == 0)
    ]

    print()
    print("===== TOPOLOGY / PH CONTRASTS =====")
    print(
        "connected/no holes but positive H0 distance structure: "
        f"{len(connected_no_holes_h0):,}"
    )
    print(
        "no binary holes but positive H1 persistence: "
        f"{len(no_binary_holes_h1):,}"
    )
    print(
        "binary holes but no positive H1 persistence: "
        f"{len(binary_holes_no_h1):,}"
    )

    print()
    print("===== BINARY HOLES / NO POSITIVE H1 PH =====")
    print(f"n: {len(binary_holes_no_h1):,}")

    if len(binary_holes_no_h1) > 0:
        print()
        print("beta_1 distribution:")
        print(
            binary_holes_no_h1[
                "beta_1"
            ].value_counts().sort_index().to_string()
        )

        print()
        print("largest beta_1 cases:")
        print(
            binary_holes_no_h1.nlargest(
                15,
                "beta_1",
            )[EXTREME_COLUMNS].to_string(
                index=False
            )
        )


def main() -> None:
    args = parse_args()

    catalog_path = args.catalog.expanduser().resolve()
    output_path = args.output.expanduser().resolve()

    df = load_and_validate_catalog(
        catalog_path
    )

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
            encoding="utf-8",
        ) as file:
            with redirect_stdout(file):
                try:
                    displayed_catalog_path = catalog_path.relative_to(
                        PROJECT_ROOT
                    )
                except ValueError:
                    displayed_catalog_path = catalog_path

                print(f"Catalog: {displayed_catalog_path}")
                print()
                print_summary(df)

        summary_text = temporary_path.read_text(
            encoding="utf-8"
        )

        normalized_summary = "\n".join(
            line.rstrip()
            for line in summary_text.splitlines()
        ) + "\n"

        temporary_path.write_text(
            normalized_summary,
            encoding="utf-8",
        )

        temporary_path.replace(
            output_path
        )
    except Exception:
        if temporary_path.exists():
            temporary_path.unlink()
        raise

    print(
        f"Wrote summary:\n{output_path}"
    )


if __name__ == "__main__":
    main()
