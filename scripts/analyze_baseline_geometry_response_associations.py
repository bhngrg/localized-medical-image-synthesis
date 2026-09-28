#!/usr/bin/env python3

"""
Analyze associations between baseline mask geometry/topology and synthesis
boundary response.

This Stage-5 analysis joins the previously generated mask geometry/topology
catalog and baseline boundary-response catalog for all eligible baseline
training slices. No model inference, mask processing, or image synthesis is
performed here.

Primary analysis
----------------
For each prespecified mask descriptor and each prespecified excess-boundary
response, compute slice-level Spearman rank correlation. Uncertainty is
estimated with a subject-cluster bootstrap: BraTS training volumes are sampled
with replacement and all eligible slices belonging to each sampled volume are
retained, including repeated copies when a volume is sampled more than once.

Inference is based on the BraTS-volume cluster-bootstrap percentile
confidence interval for Spearman rho. Slice-level nominal p-values and
multiple-testing adjustments are intentionally not used because eligible
slices are clustered within training volumes.

Secondary analysis
------------------
The same geometry/topology descriptors are associated with prediction MAE in
the near-boundary tumor interior (signed distances 1--4 pixels) and deep tumor
interior (signed distance >= 5 pixels). Composite MAE is not repeated because
hard compositing makes prediction and composite values identical inside the
mask.

Volume-level sensitivity analysis
---------------------------------
Each predictor and response is median-aggregated within BraTS training volume,
and Spearman correlation is recomputed across volumes. This prevents volumes
with more eligible slices from contributing greater weight.

The two input catalogs may refer to the same H5 slices through different
filesystem roots. Identity is therefore enforced by exact (volume, slice)
key agreement and H5 basename agreement rather than full absolute-path
equality.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


PROJECT_ROOT = Path(__file__).resolve().parents[1]

EXPECTED_SLICES = 19_941
EXPECTED_VOLUMES = 369
DEFAULT_BOOTSTRAP_REPLICATES = 10_000
DEFAULT_BOOTSTRAP_SEED = 2026
CI_LEVEL = 0.95

DEFAULT_GEOMETRY_CATALOG = (
    PROJECT_ROOT
    / "results"
    / "baseline_mask_sensitivity"
    / "mask_geometry_topology"
    / "mask_geometry_topology_catalog.csv"
)

DEFAULT_RESPONSE_CATALOG = (
    PROJECT_ROOT
    / "outputs"
    / "baseline_mask_sensitivity"
    / "boundary_response"
    / "baseline_boundary_response_catalog.csv"
)

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "baseline_mask_sensitivity"
    / "geometry_response_associations"
)

KEY_COLUMNS = [
    "volume",
    "slice",
]

GEOMETRY_COLUMNS = [
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

PRIMARY_RESPONSE_COLUMNS = [
    "pred_excess_jump_signed",
    "composite_excess_jump_signed",
    "pred_excess_boundary_gradient",
    "composite_excess_boundary_gradient",
]

SECONDARY_RESPONSE_COLUMNS = [
    "pred_mae_inside_1_4",
    "pred_mae_inside_deep",
]

REQUIRED_GEOMETRY_COLUMNS = {
    *KEY_COLUMNS,
    "slice_path",
    *GEOMETRY_COLUMNS,
}

REQUIRED_RESPONSE_COLUMNS = {
    *KEY_COLUMNS,
    "slice_path",
    "mask_pixels",
    *PRIMARY_RESPONSE_COLUMNS,
    *SECONDARY_RESPONSE_COLUMNS,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Associate baseline mask geometry/topology descriptors with "
            "baseline synthesis boundary-response metrics."
        )
    )

    parser.add_argument(
        "--geometry-catalog",
        type=Path,
        default=DEFAULT_GEOMETRY_CATALOG,
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


def resolve_existing_file(path: Path, label: str) -> Path:
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


def repository_display_path(path: Path) -> str:
    resolved = path.expanduser().resolve()

    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def validate_columns(
    df: pd.DataFrame,
    required: set[str],
    label: str,
) -> None:
    missing = sorted(required - set(df.columns))

    if missing:
        raise RuntimeError(
            f"{label} is missing required columns: {missing}"
        )


def validate_catalog(
    df: pd.DataFrame,
    *,
    label: str,
) -> None:
    if len(df) != EXPECTED_SLICES:
        raise RuntimeError(
            f"{label} contains {len(df):,} rows; expected "
            f"{EXPECTED_SLICES:,}."
        )

    if df["volume"].nunique() != EXPECTED_VOLUMES:
        raise RuntimeError(
            f"{label} contains {df['volume'].nunique():,} volumes; "
            f"expected {EXPECTED_VOLUMES:,}."
        )

    duplicate_count = int(
        df.duplicated(KEY_COLUMNS).sum()
    )

    if duplicate_count:
        raise RuntimeError(
            f"{label} contains {duplicate_count:,} duplicate "
            "(volume, slice) keys."
        )

    if df["slice_path"].isna().any():
        raise RuntimeError(
            f"{label} contains missing slice_path values."
        )


def load_and_join(
    geometry_path: Path,
    response_path: Path,
) -> pd.DataFrame:
    geometry = pd.read_csv(geometry_path)
    response = pd.read_csv(response_path)

    validate_columns(
        geometry,
        REQUIRED_GEOMETRY_COLUMNS,
        "Geometry catalog",
    )
    validate_columns(
        response,
        REQUIRED_RESPONSE_COLUMNS,
        "Response catalog",
    )

    validate_catalog(
        geometry,
        label="Geometry catalog",
    )
    validate_catalog(
        response,
        label="Response catalog",
    )

    geometry_keys = set(
        map(
            tuple,
            geometry[KEY_COLUMNS].itertuples(
                index=False,
                name=None,
            ),
        )
    )
    response_keys = set(
        map(
            tuple,
            response[KEY_COLUMNS].itertuples(
                index=False,
                name=None,
            ),
        )
    )

    if geometry_keys != response_keys:
        raise RuntimeError(
            "Geometry and response catalogs do not contain identical "
            "(volume, slice) key sets."
        )

    joined = geometry.merge(
        response,
        on=KEY_COLUMNS,
        how="inner",
        suffixes=("_geometry", "_response"),
        validate="one_to_one",
    )

    if len(joined) != EXPECTED_SLICES:
        raise RuntimeError(
            f"Joined catalog contains {len(joined):,} rows; expected "
            f"{EXPECTED_SLICES:,}."
        )

    geometry_basename = joined[
        "slice_path_geometry"
    ].map(lambda value: Path(str(value)).name)

    response_basename = joined[
        "slice_path_response"
    ].map(lambda value: Path(str(value)).name)

    basename_mismatch = int(
        (geometry_basename != response_basename).sum()
    )

    if basename_mismatch:
        raise RuntimeError(
            f"Geometry/response H5 basename mismatch in "
            f"{basename_mismatch:,} joined rows."
        )

    geometry_mask = joined["mask_pixels_geometry"]
    response_mask = joined["mask_pixels_response"]

    mask_mismatch = int(
        (geometry_mask != response_mask).sum()
    )

    if mask_mismatch:
        raise RuntimeError(
            f"Geometry/response mask_pixels mismatch in "
            f"{mask_mismatch:,} joined rows."
        )

    # The two catalogs independently carry mask_pixels.  Their exact
    # agreement was validated above, so retain the geometry value as the
    # canonical Stage-5 predictor and remove the redundant merged columns.
    joined["mask_pixels"] = geometry_mask
    joined = joined.drop(
        columns=[
            "mask_pixels_geometry",
            "mask_pixels_response",
        ]
    )

    joined = joined.sort_values(
        KEY_COLUMNS,
        kind="stable",
    ).reset_index(drop=True)

    return joined


def finite_pair(
    x: pd.Series | np.ndarray,
    y: pd.Series | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    x_array = np.asarray(x, dtype=np.float64)
    y_array = np.asarray(y, dtype=np.float64)

    valid = (
        np.isfinite(x_array)
        & np.isfinite(y_array)
    )

    return (
        x_array[valid],
        y_array[valid],
    )


def spearman_pair(
    x: pd.Series | np.ndarray,
    y: pd.Series | np.ndarray,
) -> tuple[int, float]:
    x_valid, y_valid = finite_pair(x, y)

    n = int(len(x_valid))

    if n < 2:
        return n, np.nan

    if np.unique(x_valid).size < 2:
        return n, np.nan

    if np.unique(y_valid).size < 2:
        return n, np.nan

    result = spearmanr(
        x_valid,
        y_valid,
        nan_policy="omit",
    )

    return (
        n,
        float(result.statistic),
    )


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

    volume_values = df["volume"].to_numpy()

    indices = {
        int(volume): np.flatnonzero(
            volume_values == volume
        )
        for volume in volumes
    }

    return volumes, indices


def cluster_bootstrap_ci(
    df: pd.DataFrame,
    *,
    predictor: str,
    response: str,
    volumes: np.ndarray,
    volume_indices: dict[int, np.ndarray],
    replicates: int,
    rng: np.random.Generator,
) -> tuple[float, float, int]:
    x = df[predictor].to_numpy(
        dtype=np.float64
    )
    y = df[response].to_numpy(
        dtype=np.float64
    )

    bootstrap_rho = np.empty(
        replicates,
        dtype=np.float64,
    )

    bootstrap_rho.fill(np.nan)

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

        _, rho = spearman_pair(
            x[sampled_indices],
            y[sampled_indices],
        )

        bootstrap_rho[replicate] = rho

    finite = bootstrap_rho[
        np.isfinite(bootstrap_rho)
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


def association_table(
    df: pd.DataFrame,
    *,
    responses: list[str],
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> pd.DataFrame:
    volumes, volume_indices = (
        build_volume_indices(df)
    )

    rows: list[dict[str, object]] = []

    association_index = 0

    for predictor in GEOMETRY_COLUMNS:
        for response in responses:
            n, rho = spearman_pair(
                df[predictor],
                df[response],
            )

            # Give each association an independent deterministic random stream.
            # This keeps results stable if implementation order is refactored.
            seed_sequence = np.random.SeedSequence(
                [
                    bootstrap_seed,
                    association_index,
                ]
            )
            rng = np.random.default_rng(
                seed_sequence
            )

            (
                ci_lower,
                ci_upper,
                valid_bootstrap_replicates,
            ) = cluster_bootstrap_ci(
                df,
                predictor=predictor,
                response=response,
                volumes=volumes,
                volume_indices=volume_indices,
                replicates=bootstrap_replicates,
                rng=rng,
            )

            rows.append(
                {
                    "predictor": predictor,
                    "response": response,
                    "n_slices": n,
                    "n_volumes": int(
                        df.loc[
                            np.isfinite(
                                df[predictor].to_numpy(
                                    dtype=np.float64
                                )
                            )
                            & np.isfinite(
                                df[response].to_numpy(
                                    dtype=np.float64
                                )
                            ),
                            "volume",
                        ].nunique()
                    ),
                    "spearman_rho": rho,
                    "cluster_bootstrap_ci_lower": ci_lower,
                    "cluster_bootstrap_ci_upper": ci_upper,
                    "bootstrap_replicates_requested": (
                        bootstrap_replicates
                    ),
                    "bootstrap_replicates_valid": (
                        valid_bootstrap_replicates
                    ),
                }
            )

            association_index += 1

    return pd.DataFrame(rows)


def volume_median_table(
    df: pd.DataFrame,
    *,
    responses: list[str],
) -> pd.DataFrame:
    columns = [
        *GEOMETRY_COLUMNS,
        *responses,
    ]

    aggregated = (
        df.groupby(
            "volume",
            sort=True,
        )[columns]
        .median()
        .reset_index()
    )

    if len(aggregated) != EXPECTED_VOLUMES:
        raise RuntimeError(
            f"Volume-median table contains {len(aggregated):,} rows; "
            f"expected {EXPECTED_VOLUMES:,}."
        )

    rows: list[dict[str, object]] = []

    for predictor in GEOMETRY_COLUMNS:
        for response in responses:
            n, rho = spearman_pair(
                aggregated[predictor],
                aggregated[response],
            )

            rows.append(
                {
                    "predictor": predictor,
                    "response": response,
                    "n_volumes": n,
                    "spearman_rho": rho,
                }
            )

    return pd.DataFrame(rows)


def write_summary(
    *,
    path: Path,
    joined: pd.DataFrame,
    primary: pd.DataFrame,
    secondary: pd.DataFrame,
    sensitivity: pd.DataFrame,
    geometry_path: Path,
    response_path: Path,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> None:
    deep_missing = int(
        joined[
            "pred_mae_inside_deep"
        ].isna().sum()
    )

    lines = [
        "BASELINE MASK GEOMETRY / BOUNDARY RESPONSE ASSOCIATIONS",
        "=" * 64,
        "",
        "INPUTS",
        f"Geometry catalog: {repository_display_path(geometry_path)}",
        f"Response catalog: {repository_display_path(response_path)}",
        "",
        "JOIN CONTRACT",
        f"Joined slices:              {len(joined):,}",
        f"Unique volumes:             {joined['volume'].nunique():,}",
        "Join key:                   (volume, slice)",
        "H5 identity:                basename equality",
        (
            "Geometry/response masks:   exact mask_pixels equality"
        ),
        "",
        "PRIMARY ANALYSIS",
        (
            f"Associations:              {len(primary):,}"
        ),
        (
            "Predictors:                "
            f"{len(GEOMETRY_COLUMNS):,}"
        ),
        (
            "Responses:                 "
            f"{len(PRIMARY_RESPONSE_COLUMNS):,}"
        ),
        "Effect size:                Spearman rho",
        (
            "Uncertainty:              subject/volume-cluster "
            "bootstrap percentile CI"
        ),
        (
            f"Bootstrap replicates:      {bootstrap_replicates:,}"
        ),
        (
            f"Bootstrap seed:            {bootstrap_seed}"
        ),
        (
            "Inference:                cluster-bootstrap 95% CI; "
            "slice-level p-values not used"
        ),
        "",
        "SECONDARY MAE ANALYSIS",
        (
            f"Associations:              {len(secondary):,}"
        ),
        (
            "Responses:                 pred_mae_inside_1_4, "
            "pred_mae_inside_deep"
        ),
        (
            f"Deep-interior missing:     {deep_missing:,} "
            f"({100.0 * deep_missing / len(joined):.3f}%)"
        ),
        (
            "Inference:                cluster-bootstrap 95% CI; "
            "slice-level p-values not used"
        ),
        "",
        "VOLUME-MEDIAN SENSITIVITY ANALYSIS",
        (
            f"Associations:              {len(sensitivity):,}"
        ),
        (
            f"Volumes:                   {EXPECTED_VOLUMES:,}"
        ),
        (
            "Aggregation:               median within volume before "
            "Spearman correlation"
        ),
        "",
        "PRIMARY ASSOCIATIONS",
        primary.to_string(index=False),
        "",
        "SECONDARY MAE ASSOCIATIONS",
        secondary.to_string(index=False),
        "",
        "VOLUME-MEDIAN PRIMARY SENSITIVITY",
        sensitivity.to_string(index=False),
        "",
    ]

    path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()

    geometry_path = resolve_existing_file(
        args.geometry_catalog,
        "Geometry catalog",
    )
    response_path = resolve_existing_file(
        args.response_catalog,
        "Response catalog",
    )
    output_dir = resolve_output_dir(
        args.output_dir
    )

    joined = load_and_join(
        geometry_path,
        response_path,
    )

    print("=" * 64)
    print("STAGE 5: GEOMETRY / BOUNDARY RESPONSE ASSOCIATIONS")
    print("=" * 64)
    print(f"Geometry catalog: {geometry_path}")
    print(f"Response catalog: {response_path}")
    print(f"Joined slices:    {len(joined):,}")
    print(
        f"Unique volumes:   "
        f"{joined['volume'].nunique():,}"
    )
    print(
        "H5 identity:      exact (volume, slice) + basename match"
    )
    print(
        f"Bootstrap:        {args.bootstrap_replicates:,} "
        "volume-cluster replicates"
    )
    print(f"Seed:             {args.bootstrap_seed}")
    print()

    print(
        f"Computing {len(GEOMETRY_COLUMNS) * len(PRIMARY_RESPONSE_COLUMNS):,} "
        "primary associations..."
    )

    primary = association_table(
        joined,
        responses=PRIMARY_RESPONSE_COLUMNS,
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed,
    )

    print(
        f"Computing {len(GEOMETRY_COLUMNS) * len(SECONDARY_RESPONSE_COLUMNS):,} "
        "secondary MAE associations..."
    )

    secondary = association_table(
        joined,
        responses=SECONDARY_RESPONSE_COLUMNS,
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed + 1,
    )

    print(
        "Computing volume-median primary sensitivity analysis..."
    )

    sensitivity = volume_median_table(
        joined,
        responses=PRIMARY_RESPONSE_COLUMNS,
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    primary_path = (
        output_dir
        / "primary_geometry_boundary_associations.csv"
    )
    secondary_path = (
        output_dir
        / "secondary_geometry_mae_associations.csv"
    )
    sensitivity_path = (
        output_dir
        / "volume_median_primary_associations.csv"
    )
    summary_path = (
        output_dir
        / "geometry_response_association_summary.txt"
    )

    primary.to_csv(
        primary_path,
        index=False,
    )
    secondary.to_csv(
        secondary_path,
        index=False,
    )
    sensitivity.to_csv(
        sensitivity_path,
        index=False,
    )

    write_summary(
        path=summary_path,
        joined=joined,
        primary=primary,
        secondary=secondary,
        sensitivity=sensitivity,
        geometry_path=geometry_path,
        response_path=response_path,
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed,
    )

    print()
    print("===== OUTPUTS =====")
    print(f"Primary:     {primary_path}")
    print(f"Secondary:   {secondary_path}")
    print(f"Sensitivity: {sensitivity_path}")
    print(f"Summary:     {summary_path}")
    print()
    print(
        "PASS: Stage-5 geometry/response association analysis completed."
    )


if __name__ == "__main__":
    main()
