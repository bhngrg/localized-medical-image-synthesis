#!/usr/bin/env python3
"""
Audit topology changes induced by deterministic signed-distance perturbations.

This Stage-6 diagnostic reconstructs the exact whole-tumor masks used by the
baseline-training cohort, applies the same signed Euclidean-distance
erosion/dilation definition used by analyze_baseline_mask_signed_distance.py,
and measures ordinary 2D topology before and after perturbation.

The existing signed-distance perturbation catalog is treated as a validated
input artifact and is never modified. Reconstructed perturbation areas must
agree exactly with that catalog before topology results are accepted.

No model inference or training is performed.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt
from skimage.measure import euler_number, label

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )

from src.config import load_folders_config, resolve_path
from src.data.preprocessing import load_h5_full


DEFAULT_FOLDERS_FILE = PROJECT_ROOT / "data" / "folders.yaml"

DEFAULT_INPUT_CATALOG = (
    PROJECT_ROOT
    / "outputs"
    / "baseline_mask_sensitivity"
    / "signed_distance"
    / "signed_distance_perturbation_catalog.csv"
)

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "baseline_mask_sensitivity"
    / "signed_distance"
)

DEFAULT_SUMMARY_DIR = (
    PROJECT_ROOT
    / "results"
    / "baseline_mask_sensitivity"
    / "signed_distance"
)

DEFAULT_AUDIT_PATH = (
    DEFAULT_OUTPUT_DIR
    / "signed_distance_topology_audit.csv"
)

DEFAULT_SUMMARY_CSV = (
    DEFAULT_SUMMARY_DIR
    / "signed_distance_topology_summary.csv"
)

DEFAULT_SUMMARY_TXT = (
    DEFAULT_SUMMARY_DIR
    / "signed_distance_topology_summary.txt"
)

EXPECTED_ELIGIBLE_SLICES = 19_941
EXPECTED_VOLUMES = 369
TOPOLOGY_CONNECTIVITY = 2

REQUIRED_COLUMNS = {
    "volume",
    "slice",
    "slice_path",
    "selection_pixels",
    "original_mask_pixels",
    "original_max_edt",
    "radius_pixels",
    "perturbation",
    "perturbed_mask_pixels",
    "is_empty",
}

AUDIT_COLUMNS = [
    "volume",
    "slice",
    "slice_path",
    "radius_pixels",
    "perturbation",
    "original_mask_pixels",
    "perturbed_mask_pixels",
    "original_beta_0",
    "perturbed_beta_0",
    "delta_beta_0",
    "original_beta_1",
    "perturbed_beta_1",
    "delta_beta_1",
    "original_euler_characteristic",
    "perturbed_euler_characteristic",
    "delta_euler_characteristic",
    "beta_0_changed",
    "beta_1_changed",
    "euler_changed",
    "topology_changed",
    "is_empty",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit ordinary topology changes induced by the validated "
            "signed-distance baseline-mask perturbations."
        )
    )

    parser.add_argument(
        "--folders-file",
        type=Path,
        default=DEFAULT_FOLDERS_FILE,
    )
    parser.add_argument(
        "--h5-root",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--input-catalog",
        type=Path,
        default=DEFAULT_INPUT_CATALOG,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for the full topology-audit catalog.",
    )
    parser.add_argument(
        "--summary-dir",
        type=Path,
        default=DEFAULT_SUMMARY_DIR,
        help="Directory for compact curated topology summaries.",
    )
    parser.add_argument(
        "--case",
        type=str,
        default=None,
        help="Optional diagnostic case encoded as VOLUME,SLICE.",
    )

    return parser.parse_args()


def resolve_project_path(path: Path) -> Path:
    path = path.expanduser()

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    return path.resolve()


def repository_display_path(path: Path) -> str:
    resolved = path.expanduser().resolve()

    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def parse_case(value: str | None) -> tuple[int, int] | None:
    if value is None:
        return None

    parts = value.split(",")

    if len(parts) != 2:
        raise ValueError(
            "--case must use VOLUME,SLICE format."
        )

    try:
        volume = int(parts[0])
        slice_index = int(parts[1])
    except ValueError as exc:
        raise ValueError(
            "--case volume and slice must be integers."
        ) from exc

    if volume < 0 or slice_index < 0:
        raise ValueError(
            "--case volume and slice must be non-negative."
        )

    return volume, slice_index


def resolve_h5_path(
    h5_root: Path,
    slice_path: str,
) -> Path:
    candidate = Path(slice_path).expanduser()

    if candidate.is_absolute() and candidate.is_file():
        return candidate.resolve()

    basename_candidate = h5_root / candidate.name

    if basename_candidate.is_file():
        return basename_candidate.resolve()

    direct_candidate = h5_root / candidate

    if direct_candidate.is_file():
        return direct_candidate.resolve()

    raise FileNotFoundError(
        "Could not resolve H5 slice path.\n"
        f"Catalog path: {slice_path}\n"
        f"H5 root:      {h5_root}"
    )


def load_verified_mask(
    *,
    h5_path: Path,
    expected_pixels: int,
) -> np.ndarray:
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

    unique_values = np.unique(mask)

    if not np.all(np.isin(unique_values, [0.0, 1.0])):
        raise RuntimeError(
            "load_h5_full() returned a non-binary mask for "
            f"{h5_path}: {unique_values.tolist()}"
        )

    binary_mask = mask.astype(bool)

    observed_pixels = int(binary_mask.sum())

    if observed_pixels <= 0:
        raise RuntimeError(
            f"Original baseline mask is empty: {h5_path}"
        )

    if observed_pixels != int(expected_pixels):
        raise RuntimeError(
            "Original mask area disagrees with signed-distance catalog.\n"
            f"Path:     {h5_path}\n"
            f"Expected: {expected_pixels}\n"
            f"Observed: {observed_pixels}"
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
    Exact perturbation convention from
    analyze_baseline_mask_signed_distance.py.
    """
    if radius == 0:
        return binary_mask.copy()

    if radius < 0:
        erosion_distance = abs(radius)

        return (
            binary_mask
            & (
                interior_distance
                > float(erosion_distance)
            )
        )

    return (
        binary_mask
        | (
            (~binary_mask)
            & (
                exterior_distance
                <= float(radius)
            )
        )
    )


def ordinary_topology(
    binary_mask: np.ndarray,
) -> tuple[int, int, int]:
    """
    Compute beta_0, beta_1, and Euler characteristic using the exact
    Stage-1 connectivity convention.

    The empty mask is assigned beta_0 = beta_1 = Euler = 0.
    """
    if binary_mask.dtype != np.bool_:
        raise TypeError(
            "Topology input must be a boolean mask."
        )

    if binary_mask.ndim != 2:
        raise ValueError(
            "Topology input must be two-dimensional."
        )

    if not binary_mask.any():
        return 0, 0, 0

    _, beta_0 = label(
        binary_mask,
        connectivity=TOPOLOGY_CONNECTIVITY,
        return_num=True,
    )

    beta_0 = int(beta_0)

    euler_characteristic = int(
        euler_number(
            binary_mask,
            connectivity=TOPOLOGY_CONNECTIVITY,
        )
    )

    beta_1 = int(
        beta_0 - euler_characteristic
    )

    if beta_1 < 0:
        raise RuntimeError(
            f"Computed negative beta_1: {beta_1}"
        )

    if euler_characteristic != beta_0 - beta_1:
        raise RuntimeError(
            "Euler/Betti identity failed."
        )

    return (
        beta_0,
        beta_1,
        euler_characteristic,
    )


def boolean_column(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)

    normalized = (
        series.astype(str)
        .str.strip()
        .str.lower()
    )

    allowed = {"true", "false"}

    unexpected = sorted(
        set(normalized.unique()) - allowed
    )

    if unexpected:
        raise RuntimeError(
            "Unexpected boolean values in is_empty: "
            f"{unexpected}"
        )

    return normalized.map(
        {"true": True, "false": False}
    ).astype(bool)


def write_csv_atomic(
    df: pd.DataFrame,
    path: Path,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_suffix(
        path.suffix + ".tmp"
    )

    try:
        df.to_csv(
            temporary,
            index=False,
        )
        temporary.replace(path)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise


def main() -> None:
    args = parse_args()

    folders_file = resolve_project_path(
        args.folders_file
    )

    folders_config = load_folders_config(
        folders_file
    )

    h5_root = resolve_path(
        key="h5_root",
        cli_value=args.h5_root,
        config=folders_config,
    ).expanduser().resolve()

    input_catalog = resolve_project_path(
        args.input_catalog
    )

    output_dir = resolve_project_path(
        args.output_dir
    )

    summary_dir = resolve_project_path(
        args.summary_dir
    )

    if not h5_root.is_dir():
        raise RuntimeError(
            f"H5 root does not exist: {h5_root}"
        )

    if not input_catalog.is_file():
        raise RuntimeError(
            "Signed-distance catalog does not exist:\n"
            f"{input_catalog}"
        )

    catalog = pd.read_csv(
        input_catalog
    )

    missing = sorted(
        REQUIRED_COLUMNS - set(catalog.columns)
    )

    if missing:
        raise RuntimeError(
            "Signed-distance catalog is missing required columns: "
            f"{missing}"
        )

    catalog["is_empty"] = boolean_column(
        catalog["is_empty"]
    )

    duplicate_key = catalog.duplicated(
        subset=[
            "volume",
            "slice",
            "radius_pixels",
        ],
        keep=False,
    )

    if duplicate_key.any():
        raise RuntimeError(
            "Signed-distance catalog contains duplicate "
            "(volume, slice, radius) rows."
        )

    unique_slices = (
        catalog[
            ["volume", "slice"]
        ]
        .drop_duplicates()
    )

    unique_volumes = int(
        catalog["volume"].nunique()
    )

    if len(unique_slices) != EXPECTED_ELIGIBLE_SLICES:
        raise RuntimeError(
            f"Expected {EXPECTED_ELIGIBLE_SLICES:,} unique slices "
            f"but found {len(unique_slices):,}."
        )

    if unique_volumes != EXPECTED_VOLUMES:
        raise RuntimeError(
            f"Expected {EXPECTED_VOLUMES:,} volumes "
            f"but found {unique_volumes:,}."
        )

    case = parse_case(
        args.case
    )

    if case is not None:
        volume, slice_index = case

        catalog = catalog[
            (catalog["volume"] == volume)
            & (catalog["slice"] == slice_index)
        ].copy()

        if catalog.empty:
            raise RuntimeError(
                f"Diagnostic case {case} is absent from the catalog."
            )

    catalog = catalog.sort_values(
        ["volume", "slice", "radius_pixels"]
    ).reset_index(drop=True)

    radii = tuple(
        sorted(
            int(value)
            for value in catalog["radius_pixels"].unique()
        )
    )

    if 0 not in radii:
        raise RuntimeError(
            "Input catalog does not contain radius zero."
        )

    print(
        "================================================================"
    )
    print(
        "STAGE 6: CONTROLLED GEOMETRIC PERTURBATION TOPOLOGY AUDIT"
    )
    print(
        "================================================================"
    )
    print(
        f"Input catalog:        {input_catalog}"
    )
    print(
        f"H5 root:              {h5_root}"
    )
    print(
        f"Rows to audit:        {len(catalog):,}"
    )
    print(
        f"Unique slices:        "
        f"{catalog[['volume', 'slice']].drop_duplicates().shape[0]:,}"
    )
    print(
        f"Radii:                "
        f"{', '.join(f'{radius:+d}' for radius in radii)}"
    )
    print(
        f"Topology connectivity:{TOPOLOGY_CONNECTIVITY}"
    )
    print(
        "Model inference:       none"
    )
    print()

    results: list[dict[str, object]] = []

    grouped = catalog.groupby(
        ["volume", "slice"],
        sort=True,
    )

    total_groups = grouped.ngroups

    for position, ((volume, slice_index), group) in enumerate(
        grouped,
        start=1,
    ):
        first = group.iloc[0]

        slice_paths = group["slice_path"].astype(str).unique()

        if len(slice_paths) != 1:
            raise RuntimeError(
                "slice_path is inconsistent within one slice group."
            )

        original_pixel_values = (
            group["original_mask_pixels"]
            .astype(int)
            .unique()
        )

        if len(original_pixel_values) != 1:
            raise RuntimeError(
                "original_mask_pixels is inconsistent within one slice."
            )

        slice_path = str(slice_paths[0])
        original_pixels = int(
            original_pixel_values[0]
        )

        h5_path = resolve_h5_path(
            h5_root,
            slice_path,
        )

        original_mask = load_verified_mask(
            h5_path=h5_path,
            expected_pixels=original_pixels,
        )

        interior_distance = distance_transform_edt(
            original_mask
        )

        exterior_distance = distance_transform_edt(
            ~original_mask
        )

        original_beta_0, original_beta_1, original_euler = (
            ordinary_topology(
                original_mask
            )
        )

        for _, row in group.iterrows():
            radius = int(
                row["radius_pixels"]
            )

            perturbed_mask = perturb_mask(
                binary_mask=original_mask,
                radius=radius,
                interior_distance=interior_distance,
                exterior_distance=exterior_distance,
            )

            reconstructed_pixels = int(
                perturbed_mask.sum()
            )

            expected_perturbed_pixels = int(
                row["perturbed_mask_pixels"]
            )

            if reconstructed_pixels != expected_perturbed_pixels:
                raise RuntimeError(
                    "Reconstructed perturbation disagrees with the "
                    "validated signed-distance catalog.\n"
                    f"Volume:   {int(volume)}\n"
                    f"Slice:    {int(slice_index)}\n"
                    f"Radius:   {radius:+d}\n"
                    f"Expected: {expected_perturbed_pixels}\n"
                    f"Observed: {reconstructed_pixels}"
                )

            reconstructed_empty = bool(
                reconstructed_pixels == 0
            )

            if reconstructed_empty != bool(row["is_empty"]):
                raise RuntimeError(
                    "Reconstructed empty-mask status disagrees with "
                    "the validated signed-distance catalog."
                )

            perturbed_beta_0, perturbed_beta_1, perturbed_euler = (
                ordinary_topology(
                    perturbed_mask
                )
            )

            delta_beta_0 = (
                perturbed_beta_0 - original_beta_0
            )

            delta_beta_1 = (
                perturbed_beta_1 - original_beta_1
            )

            delta_euler = (
                perturbed_euler - original_euler
            )

            beta_0_changed = bool(
                delta_beta_0 != 0
            )

            beta_1_changed = bool(
                delta_beta_1 != 0
            )

            euler_changed = bool(
                delta_euler != 0
            )

            topology_changed = bool(
                beta_0_changed
                or beta_1_changed
            )

            if delta_euler != delta_beta_0 - delta_beta_1:
                raise RuntimeError(
                    "Delta Euler/Betti identity failed."
                )

            if radius == 0:
                if (
                    delta_beta_0 != 0
                    or delta_beta_1 != 0
                    or delta_euler != 0
                    or reconstructed_empty
                ):
                    raise RuntimeError(
                        "Radius-zero topology does not reproduce "
                        "the original mask."
                    )

            results.append(
                {
                    "volume": int(volume),
                    "slice": int(slice_index),
                    "slice_path": slice_path,
                    "radius_pixels": radius,
                    "perturbation": str(row["perturbation"]),
                    "original_mask_pixels": original_pixels,
                    "perturbed_mask_pixels": reconstructed_pixels,
                    "original_beta_0": original_beta_0,
                    "perturbed_beta_0": perturbed_beta_0,
                    "delta_beta_0": delta_beta_0,
                    "original_beta_1": original_beta_1,
                    "perturbed_beta_1": perturbed_beta_1,
                    "delta_beta_1": delta_beta_1,
                    "original_euler_characteristic": original_euler,
                    "perturbed_euler_characteristic": perturbed_euler,
                    "delta_euler_characteristic": delta_euler,
                    "beta_0_changed": beta_0_changed,
                    "beta_1_changed": beta_1_changed,
                    "euler_changed": euler_changed,
                    "topology_changed": topology_changed,
                    "is_empty": reconstructed_empty,
                }
            )

        if (
            case is not None
            or position % 1000 == 0
            or position == total_groups
        ):
            print(
                f"Processed {position:,} / {total_groups:,} slices"
            )

    audit = pd.DataFrame(
        results,
        columns=AUDIT_COLUMNS,
    )

    if len(audit) != len(catalog):
        raise RuntimeError(
            "Topology-audit row count does not match input catalog."
        )

    summary_rows = []

    for radius, group in audit.groupby(
        "radius_pixels",
        sort=True,
    ):
        n = len(group)

        summary_rows.append(
            {
                "radius_pixels": int(radius),
                "perturbation": str(
                    group["perturbation"].iloc[0]
                ),
                "n_slices": n,
                "beta_0_changed_n": int(
                    group["beta_0_changed"].sum()
                ),
                "beta_0_changed_fraction": float(
                    group["beta_0_changed"].mean()
                ),
                "beta_1_changed_n": int(
                    group["beta_1_changed"].sum()
                ),
                "beta_1_changed_fraction": float(
                    group["beta_1_changed"].mean()
                ),
                "topology_changed_n": int(
                    group["topology_changed"].sum()
                ),
                "topology_changed_fraction": float(
                    group["topology_changed"].mean()
                ),
                "empty_n": int(
                    group["is_empty"].sum()
                ),
                "empty_fraction": float(
                    group["is_empty"].mean()
                ),
                "median_delta_beta_0": float(
                    group["delta_beta_0"].median()
                ),
                "median_delta_beta_1": float(
                    group["delta_beta_1"].median()
                ),
            }
        )

    summary = pd.DataFrame(
        summary_rows
    )

    audit_path = (
        output_dir
        / DEFAULT_AUDIT_PATH.name
    )

    summary_csv = (
        summary_dir
        / DEFAULT_SUMMARY_CSV.name
    )

    summary_txt = (
        summary_dir
        / DEFAULT_SUMMARY_TXT.name
    )

    write_csv_atomic(
        audit,
        audit_path,
    )

    write_csv_atomic(
        summary,
        summary_csv,
    )

    lines = [
        "CONTROLLED GEOMETRIC PERTURBATION TOPOLOGY AUDIT",
        "=" * 64,
        "",
        f"Input catalog: {repository_display_path(input_catalog)}",
        f"Audited rows:  {len(audit):,}",
        (
            "Unique slices: "
            f"{audit[['volume', 'slice']].drop_duplicates().shape[0]:,}"
        ),
        (
            "Unique volumes: "
            f"{audit['volume'].nunique():,}"
        ),
        (
            "Connectivity:   "
            f"{TOPOLOGY_CONNECTIVITY} (same as Stage 1)"
        ),
        "",
        "RADIUS-LEVEL TOPOLOGY CHANGES",
        "",
        summary.to_string(
            index=False,
        ),
        "",
        "NOTES",
        (
            "- beta_0 and beta_1 use the exact ordinary-topology "
            "definition from Stage 1."
        ),
        (
            "- Empty perturbed masks are assigned beta_0=0, "
            "beta_1=0, Euler characteristic=0."
        ),
        (
            "- Empty-mask destruction is reported separately from "
            "the general topology-changed indicator."
        ),
        (
            "- Every reconstructed perturbation was required to "
            "match the existing signed-distance catalog area exactly."
        ),
        (
            "- This is a geometric perturbation audit; topology "
            "changes here are consequences of erosion/dilation, "
            "not deliberately targeted topology interventions."
        ),
        "",
    ]

    summary_txt.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_summary = summary_txt.with_suffix(
        summary_txt.suffix + ".tmp"
    )

    try:
        temporary_summary.write_text(
            "\n".join(lines),
            encoding="utf-8",
        )
        temporary_summary.replace(
            summary_txt
        )
    except Exception:
        if temporary_summary.exists():
            temporary_summary.unlink()
        raise

    print()
    print("===== OUTPUTS =====")
    print(f"Audit:   {audit_path}")
    print(f"Summary: {summary_csv}")
    print(f"Report:  {summary_txt}")
    print()
    print(
        "PASS: Stage-6 geometric perturbation topology audit completed."
    )


if __name__ == "__main__":
    main()
