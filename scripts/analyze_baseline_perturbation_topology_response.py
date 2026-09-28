#!/usr/bin/env python3

"""
Analyze whether topology-changing mask perturbations produce larger baseline
generator responses than topology-preserving perturbations.

This analysis consumes the completed Stage-7 controlled perturbation-response
catalog.  For each prespecified nonzero perturbation radius and response
metric, the primary effect is the difference in median response between
topology-changing and topology-preserving perturbations.

Uncertainty is estimated with a BraTS-volume cluster bootstrap.  Training
volumes are sampled with replacement and all eligible slice-level conditions
belonging to each sampled volume are retained, including repeated copies when
a volume is sampled more than once.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

EXPECTED_SLICES = 19_941
EXPECTED_VOLUMES = 369
EXPECTED_ROWS = 139_587
EXPECTED_SKIPPED = 59

ALL_RADII = (
    -4,
    -2,
    -1,
    0,
    1,
    2,
    4,
)

ANALYSIS_RADII = (
    -4,
    -2,
    -1,
    1,
    2,
    4,
)

DEFAULT_BOOTSTRAP_REPLICATES = 10_000
DEFAULT_BOOTSTRAP_SEED = 2026
CI_LEVEL = 0.95

DEFAULT_RESPONSE_CATALOG = (
    PROJECT_ROOT
    / "outputs"
    / "baseline_mask_sensitivity"
    / "perturbation_response"
    / "baseline_perturbation_response_catalog.csv"
)

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "baseline_mask_sensitivity"
    / "perturbation_topology_response"
)

KEY_COLUMNS = [
    "volume",
    "slice",
    "radius_pixels",
]

RESPONSE_COLUMNS = [
    "composite_mae_global",
    "composite_mae_original_boundary",
    "composite_mae_changed_region",
    "composite_abs_response_near_change_fraction",
]

REQUIRED_COLUMNS = {
    *KEY_COLUMNS,
    "slice_path",
    "perturbation",
    "original_mask_pixels",
    "perturbed_mask_pixels",
    "is_empty",
    "inference_skipped",
    "topology_changed",
    *RESPONSE_COLUMNS,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare Stage-7 baseline perturbation responses between "
            "topology-changing and topology-preserving mask perturbations."
        )
    )

    parser.add_argument(
        "--response-catalog",
        type=Path,
        default=DEFAULT_RESPONSE_CATALOG,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--bootstrap-replicates",
        type=int,
        default=DEFAULT_BOOTSTRAP_REPLICATES,
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=DEFAULT_BOOTSTRAP_SEED,
    )

    args = parser.parse_args()

    if args.bootstrap_replicates <= 0:
        parser.error("--bootstrap-replicates must be positive.")

    return args


def resolve_existing_file(
    path: Path,
    label: str,
) -> Path:
    resolved = path.expanduser()

    if not resolved.is_absolute():
        resolved = PROJECT_ROOT / resolved

    resolved = resolved.resolve()

    if not resolved.is_file():
        raise FileNotFoundError(
            f"{label} does not exist:\n{resolved}"
        )

    return resolved


def resolve_output_dir(path: Path) -> Path:
    resolved = path.expanduser()

    if not resolved.is_absolute():
        resolved = PROJECT_ROOT / resolved

    return resolved.resolve()


def validate_columns(
    df: pd.DataFrame,
) -> None:
    missing = sorted(
        REQUIRED_COLUMNS - set(df.columns)
    )

    if missing:
        raise RuntimeError(
            f"Response catalog is missing required columns: {missing}"
        )


def validate_catalog(
    df: pd.DataFrame,
) -> None:
    if len(df) != EXPECTED_ROWS:
        raise RuntimeError(
            f"Response catalog contains {len(df):,} rows; expected "
            f"{EXPECTED_ROWS:,}."
        )

    if df["volume"].nunique() != EXPECTED_VOLUMES:
        raise RuntimeError(
            f"Response catalog contains "
            f"{df['volume'].nunique():,} volumes; expected "
            f"{EXPECTED_VOLUMES:,}."
        )

    slice_count = (
        df[
            [
                "volume",
                "slice",
            ]
        ]
        .drop_duplicates()
        .shape[0]
    )

    if slice_count != EXPECTED_SLICES:
        raise RuntimeError(
            f"Response catalog contains {slice_count:,} unique slices; "
            f"expected {EXPECTED_SLICES:,}."
        )

    duplicate_count = int(
        df.duplicated(KEY_COLUMNS).sum()
    )

    if duplicate_count:
        raise RuntimeError(
            f"Response catalog contains {duplicate_count:,} duplicate "
            "(volume, slice, radius_pixels) keys."
        )

    observed_radii = tuple(
        sorted(
            int(value)
            for value in df["radius_pixels"].unique()
        )
    )

    if observed_radii != ALL_RADII:
        raise RuntimeError(
            "Unexpected perturbation radii.\n"
            f"Observed: {observed_radii}\n"
            f"Expected: {ALL_RADII}"
        )

    counts = (
        df.groupby(
            "radius_pixels",
            sort=True,
        )
        .size()
    )

    for radius in ALL_RADII:
        count = int(
            counts.get(
                radius,
                0,
            )
        )

        if count != EXPECTED_SLICES:
            raise RuntimeError(
                f"Radius {radius:+d} contains {count:,} rows; expected "
                f"{EXPECTED_SLICES:,}."
            )

    if df["slice_path"].isna().any():
        raise RuntimeError(
            "Response catalog contains missing slice_path values."
        )

    skipped = df[
        df["inference_skipped"].astype(bool)
    ]

    if len(skipped) != EXPECTED_SKIPPED:
        raise RuntimeError(
            f"Observed {len(skipped):,} skipped conditions; expected "
            f"{EXPECTED_SKIPPED:,}."
        )

    if not (
        (skipped["radius_pixels"] == -4).all()
        and skipped["is_empty"].astype(bool).all()
    ):
        raise RuntimeError(
            "Skipped conditions are not exactly the expected empty "
            "radius -4 erosions."
        )

    empty = df[
        df["is_empty"].astype(bool)
    ]

    if len(empty) != EXPECTED_SKIPPED:
        raise RuntimeError(
            f"Observed {len(empty):,} empty conditions; expected "
            f"{EXPECTED_SKIPPED:,}."
        )

    radius_zero = df[
        df["radius_pixels"] == 0
    ]

    if radius_zero["inference_skipped"].astype(bool).any():
        raise RuntimeError(
            "Radius 0 contains skipped inference conditions."
        )

    if radius_zero["topology_changed"].astype(bool).any():
        raise RuntimeError(
            "Radius 0 unexpectedly contains topology changes."
        )

    zero_check_columns = [
        "composite_mae_global",
        "composite_mae_original_boundary",
        "composite_abs_response_near_change_fraction",
    ]

    for column in zero_check_columns:
        values = radius_zero[
            column
        ].to_numpy(
            dtype=np.float64
        )

        if not np.all(
            np.isfinite(values)
            & (values == 0.0)
        ):
            raise RuntimeError(
                f"Radius-0 invariant failed for {column}."
            )

    changed_region = radius_zero[
        "composite_mae_changed_region"
    ]

    if changed_region.notna().any():
        raise RuntimeError(
            "Radius-0 composite_mae_changed_region must be undefined "
            "because no pixels changed."
        )


def load_catalog(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    validate_columns(df)
    validate_catalog(df)

    df = df.sort_values(
        KEY_COLUMNS,
        kind="stable",
    ).reset_index(drop=True)

    return df


def finite_values(
    values: pd.Series | np.ndarray,
) -> np.ndarray:
    array = np.asarray(
        values,
        dtype=np.float64,
    )

    return array[
        np.isfinite(array)
    ]


def distribution_summary(
    values: pd.Series | np.ndarray,
) -> dict[str, float | int]:
    array = finite_values(values)

    if len(array) == 0:
        return {
            "n": 0,
            "median": np.nan,
            "q25": np.nan,
            "q75": np.nan,
            "mean": np.nan,
        }

    q25, median, q75 = np.quantile(
        array,
        [
            0.25,
            0.50,
            0.75,
        ],
    )

    return {
        "n": int(len(array)),
        "median": float(median),
        "q25": float(q25),
        "q75": float(q75),
        "mean": float(array.mean()),
    }


def radius_summary_table(
    df: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict] = []

    for radius in ALL_RADII:
        subset = df[
            df["radius_pixels"] == radius
        ]

        valid = subset[
            ~subset["inference_skipped"].astype(bool)
        ]

        row: dict[str, float | int] = {
            "radius_pixels": radius,
            "conditions": int(len(subset)),
            "inference_conditions": int(len(valid)),
            "skipped_conditions": int(
                subset["inference_skipped"].astype(bool).sum()
            ),
            "empty_conditions": int(
                subset["is_empty"].astype(bool).sum()
            ),
            "topology_changed": int(
                subset["topology_changed"].astype(bool).sum()
            ),
            "topology_changed_fraction": float(
                subset["topology_changed"].astype(bool).mean()
            ),
        }

        for response in RESPONSE_COLUMNS:
            summary = distribution_summary(
                valid[response]
            )

            row[f"{response}_n"] = summary["n"]
            row[f"{response}_median"] = summary["median"]
            row[f"{response}_q25"] = summary["q25"]
            row[f"{response}_q75"] = summary["q75"]
            row[f"{response}_mean"] = summary["mean"]

        rows.append(row)

    return pd.DataFrame(rows)


def build_volume_indices(
    df: pd.DataFrame,
) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    volumes = np.asarray(
        sorted(df["volume"].unique()),
        dtype=np.int64,
    )

    if len(volumes) != EXPECTED_VOLUMES:
        raise RuntimeError(
            f"Expected {EXPECTED_VOLUMES:,} volumes but found "
            f"{len(volumes):,}."
        )

    volume_values = df[
        "volume"
    ].to_numpy()

    indices = {
        int(volume): np.flatnonzero(
            volume_values == volume
        )
        for volume in volumes
    }

    return volumes, indices


def median_difference(
    values: np.ndarray,
    topology_changed: np.ndarray,
) -> tuple[
    int,
    int,
    float,
    float,
    float,
]:
    finite = np.isfinite(values)

    preserved = values[
        finite
        & ~topology_changed
    ]

    changed = values[
        finite
        & topology_changed
    ]

    if (
        len(preserved) == 0
        or len(changed) == 0
    ):
        return (
            int(len(preserved)),
            int(len(changed)),
            np.nan,
            np.nan,
            np.nan,
        )

    preserved_median = float(
        np.median(preserved)
    )
    changed_median = float(
        np.median(changed)
    )

    return (
        int(len(preserved)),
        int(len(changed)),
        preserved_median,
        changed_median,
        changed_median - preserved_median,
    )


def cluster_bootstrap_median_difference(
    df: pd.DataFrame,
    *,
    response: str,
    volumes: np.ndarray,
    volume_indices: dict[int, np.ndarray],
    replicates: int,
    rng: np.random.Generator,
) -> tuple[float, float, int]:
    values = df[
        response
    ].to_numpy(
        dtype=np.float64
    )

    topology_changed = df[
        "topology_changed"
    ].astype(bool).to_numpy()

    bootstrap_difference = np.empty(
        replicates,
        dtype=np.float64,
    )

    bootstrap_difference.fill(
        np.nan
    )

    for replicate in range(replicates):
        sampled_volumes = rng.choice(
            volumes,
            size=len(volumes),
            replace=True,
        )

        sampled_indices = np.concatenate(
            [
                volume_indices[int(volume)]
                for volume in sampled_volumes
            ]
        )

        (
            _,
            _,
            _,
            _,
            difference,
        ) = median_difference(
            values[sampled_indices],
            topology_changed[sampled_indices],
        )

        bootstrap_difference[
            replicate
        ] = difference

    finite = bootstrap_difference[
        np.isfinite(
            bootstrap_difference
        )
    ]

    if len(finite) == 0:
        return np.nan, np.nan, 0

    alpha = 1.0 - CI_LEVEL

    lower, upper = np.quantile(
        finite,
        [
            alpha / 2.0,
            1.0 - alpha / 2.0,
        ],
    )

    return (
        float(lower),
        float(upper),
        int(len(finite)),
    )


def topology_response_table(
    df: pd.DataFrame,
    *,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> pd.DataFrame:
    rows: list[dict] = []
    comparison_index = 0

    for radius in ANALYSIS_RADII:
        radius_df = df[
            (df["radius_pixels"] == radius)
            & ~df["inference_skipped"].astype(bool)
        ].copy()

        volumes, volume_indices = (
            build_volume_indices(radius_df)
        )

        topology = radius_df[
            "topology_changed"
        ].astype(bool)

        preserved_volumes = int(
            radius_df.loc[
                ~topology,
                "volume",
            ].nunique()
        )

        changed_volumes = int(
            radius_df.loc[
                topology,
                "volume",
            ].nunique()
        )

        for response in RESPONSE_COLUMNS:
            values = radius_df[
                response
            ].to_numpy(
                dtype=np.float64
            )

            topology_values = topology.to_numpy()

            (
                n_preserved,
                n_changed,
                median_preserved,
                median_changed,
                difference,
            ) = median_difference(
                values,
                topology_values,
            )

            seed_sequence = np.random.SeedSequence(
                [
                    bootstrap_seed,
                    comparison_index,
                ]
            )

            rng = np.random.default_rng(
                seed_sequence
            )

            (
                ci_lower,
                ci_upper,
                valid_bootstrap_replicates,
            ) = cluster_bootstrap_median_difference(
                radius_df,
                response=response,
                volumes=volumes,
                volume_indices=volume_indices,
                replicates=bootstrap_replicates,
                rng=rng,
            )

            rows.append(
                {
                    "radius_pixels": radius,
                    "response": response,
                    "n_topology_preserved": n_preserved,
                    "n_topology_changed": n_changed,
                    "n_volumes_topology_preserved": preserved_volumes,
                    "n_volumes_topology_changed": changed_volumes,
                    "median_topology_preserved": median_preserved,
                    "median_topology_changed": median_changed,
                    "median_difference_changed_minus_preserved": difference,
                    "cluster_bootstrap_ci_lower": ci_lower,
                    "cluster_bootstrap_ci_upper": ci_upper,
                    "ci_level": CI_LEVEL,
                    "bootstrap_replicates_requested": (
                        bootstrap_replicates
                    ),
                    "bootstrap_replicates_valid": (
                        valid_bootstrap_replicates
                    ),
                }
            )

            comparison_index += 1

    result = pd.DataFrame(rows)

    expected_comparisons = (
        len(ANALYSIS_RADII)
        * len(RESPONSE_COLUMNS)
    )

    if len(result) != expected_comparisons:
        raise RuntimeError(
            f"Produced {len(result):,} topology-response comparisons; "
            f"expected {expected_comparisons:,}."
        )

    return result


def write_summary(
    *,
    path: Path,
    catalog_path: Path,
    df: pd.DataFrame,
    radius_summary: pd.DataFrame,
    topology_response: pd.DataFrame,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> None:
    skipped = int(
        df["inference_skipped"].astype(bool).sum()
    )

    lines = [
        "BASELINE PERTURBATION / TOPOLOGY RESPONSE ANALYSIS",
        "=" * 64,
        "",
        "INPUT",
        f"Stage-7 response catalog: {catalog_path}",
        "",
        "CATALOG CONTRACT",
        f"Rows:                      {len(df):,}",
        f"Unique slices:             {EXPECTED_SLICES:,}",
        f"Unique volumes:            {df['volume'].nunique():,}",
        (
            "Radii:                     "
            + ", ".join(
                f"{radius:+d}"
                for radius in ALL_RADII
            )
        ),
        f"Skipped inference:         {skipped:,}",
        "Skipped condition:         empty radius -4 erosions only",
        "Radius-0 control:          exact identity response",
        "",
        "PRIMARY TOPOLOGY ANALYSIS",
        (
            f"Radii:                     "
            f"{len(ANALYSIS_RADII):,} nonzero radii"
        ),
        (
            f"Responses:                 "
            f"{len(RESPONSE_COLUMNS):,}"
        ),
        (
            f"Comparisons:               "
            f"{len(topology_response):,}"
        ),
        (
            "Effect:                    median(topology changed) - "
            "median(topology preserved)"
        ),
        (
            "Uncertainty:               BraTS-volume cluster bootstrap "
            "percentile CI"
        ),
        (
            f"Bootstrap replicates:      {bootstrap_replicates:,}"
        ),
        (
            f"Bootstrap seed:            {bootstrap_seed}"
        ),
        f"CI level:                  {CI_LEVEL:.2f}",
        "P-values / multiplicity:    not used",
        "",
        "RADIUS DESCRIPTIVE SUMMARY",
        radius_summary.to_string(index=False),
        "",
        "TOPOLOGY-STRATIFIED RESPONSE",
        topology_response.to_string(index=False),
        "",
    ]

    path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()

    catalog_path = resolve_existing_file(
        args.response_catalog,
        "Stage-7 response catalog",
    )

    output_dir = resolve_output_dir(
        args.output_dir
    )

    df = load_catalog(
        catalog_path
    )

    print("=" * 64)
    print("STAGE 8: PERTURBATION / TOPOLOGY RESPONSE ANALYSIS")
    print("=" * 64)
    print(f"Response catalog: {catalog_path}")
    print(f"Rows:             {len(df):,}")
    print(f"Unique slices:    {EXPECTED_SLICES:,}")
    print(
        f"Unique volumes:   "
        f"{df['volume'].nunique():,}"
    )
    print(
        "Analysis radii:   "
        + ", ".join(
            f"{radius:+d}"
            for radius in ANALYSIS_RADII
        )
    )
    print(
        f"Responses:        {len(RESPONSE_COLUMNS):,}"
    )
    print(
        f"Comparisons:      "
        f"{len(ANALYSIS_RADII) * len(RESPONSE_COLUMNS):,}"
    )
    print(
        f"Bootstrap:        {args.bootstrap_replicates:,} "
        "volume-cluster replicates"
    )
    print(f"Seed:             {args.bootstrap_seed}")
    print("Training:         none")
    print("Model inference:  none")
    print()

    print(
        "Computing radius-level descriptive summary..."
    )

    radius_summary = radius_summary_table(
        df
    )

    print(
        "Computing topology-stratified response comparisons..."
    )

    topology_response = topology_response_table(
        df,
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed,
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    radius_path = (
        output_dir
        / "perturbation_response_by_radius.csv"
    )

    topology_path = (
        output_dir
        / "topology_stratified_perturbation_response.csv"
    )

    summary_path = (
        output_dir
        / "perturbation_topology_response_summary.txt"
    )

    radius_summary.to_csv(
        radius_path,
        index=False,
    )

    topology_response.to_csv(
        topology_path,
        index=False,
    )

    write_summary(
        path=summary_path,
        catalog_path=catalog_path,
        df=df,
        radius_summary=radius_summary,
        topology_response=topology_response,
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed,
    )

    print()
    print("===== VALIDATION =====")
    print(
        f"Radius summary rows:       "
        f"{len(radius_summary):,}"
    )
    print(
        f"Topology comparisons:      "
        f"{len(topology_response):,}"
    )
    print(
        f"Skipped Stage-7 conditions:"
        f" {df['inference_skipped'].astype(bool).sum():,}"
    )
    print()
    print("===== OUTPUTS =====")
    print(f"Radius summary: {radius_path}")
    print(f"Topology:       {topology_path}")
    print(f"Summary:        {summary_path}")
    print()
    print(
        "PASS: Stage-8 perturbation/topology response analysis completed."
    )


if __name__ == "__main__":
    main()
