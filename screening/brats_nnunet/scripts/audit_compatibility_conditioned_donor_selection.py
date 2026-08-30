#!/usr/bin/env python3
"""
Audit donor assignments against compatibility-conditioned donor references.

Purpose
-------
Candidate donor assignments should not be compared only with the unconditional
pool of all eligible donor slices because external-base compatibility itself
changes which donors can be used.

For each selected external base, this audit defines the reference donor
mechanism as:

    1. select one compatible training volume uniformly;
    2. select one compatible donor slice uniformly within that volume.

Every selected external base contributes equal total probability mass.

The resulting reference distribution is calculated analytically from the exact
base-donor compatibility graph. No Monte Carlo donor resampling is required.

The assigned donors from the nested 125-, 250-, and 625-case candidate designs
are then compared with these compatibility-conditioned reference distributions.

This script is descriptive only. It does not select cases, modify assignments,
freeze a manifest, or run BR-LoRA inference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys
from typing import Any

PROJECT_ROOT = Path(
    __file__
).resolve().parents[3]

if str(
    PROJECT_ROOT
) not in sys.path:
    sys.path.insert(
        0,
        str(
            PROJECT_ROOT
        ),
    )

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.config import (
    load_folders_config,
    resolve_path,
    save_folders_config,
)
from src.data import (
    load_validation_dataset_specification,
)

from screening.brats_nnunet.scripts.audit_external_manifest_design import (
    compute_compatible_donor_indices_for_batch,
    load_and_pack_donor_masks,
    load_selected_base_brain_masks,
    resolve_device,
)


IMAGE_HEIGHT = 240
IMAGE_WIDTH = 240
IMAGE_AREA = (
    IMAGE_HEIGHT
    * IMAGE_WIDTH
)

CANDIDATE_QUOTAS = {
    "candidate_125": 1,
    "candidate_250": 2,
    "candidate_625": 5,
}

NUMERIC_DONOR_METRICS = {
    "tumor_area_pixels":
        "donor_tumor_area_pixels",

    "bbox_area":
        "donor_bbox_area",

    "bbox_fill_fraction":
        "donor_bbox_fill_fraction",

    "connected_component_count":
        "donor_connected_component_count",

    "largest_component_fraction":
        "donor_largest_component_fraction",

    "centroid_x_normalized":
        "donor_centroid_x_normalized",

    "centroid_y_normalized":
        "donor_centroid_y_normalized",

    "compatible_external_base_count":
        "donor_compatible_external_base_count",
}

OUTPUT_COMPARISON = (
    "compatibility_conditioned_donor_comparison.csv"
)

OUTPUT_LATERALITY = (
    "compatibility_conditioned_laterality.csv"
)

OUTPUT_VOLUME = (
    "compatibility_conditioned_volume_distribution.csv"
)

OUTPUT_SUMMARY = (
    "compatibility_conditioned_donor_summary.json"
)

FIGURE_DIR_NAME = "figures"


class ConditionedDonorAuditError(
    RuntimeError
):
    """Raised when compatibility-conditioned donor auditing fails."""


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Compare candidate donor assignments with exact "
            "compatibility-conditioned donor references."
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
        "--validation-dataset",
        type=Path,
        default=None,
        help=(
            "Registered BraTS validation dataset YAML. Overrides "
            "yaml_validation_dataset_path in --folders-file."
        ),
    )

    parser.add_argument(
        "--candidate-assignments",
        type=Path,
        default=None,
        help=(
            "candidate_assignments_all.csv from the completed "
            "external manifest design audit. If omitted, uses "
            "<nnunet_run_root>/external_manifest_design_audit/"
            "candidate_assignments_all.csv."
        ),
    )

    parser.add_argument(
        "--donor-morphology-csv",
        type=Path,
        default=None,
        help=(
            "Donor morphology CSV. If omitted, uses "
            "<nnunet_run_root>/donor_morphology_audit/"
            "donor_morphology.csv."
        ),
    )

    parser.add_argument(
        "--pair-space-summary",
        type=Path,
        default=None,
        help=(
            "Pair-space summary JSON. If omitted, uses "
            "<nnunet_run_root>/external_pair_space_audit/"
            "external_pair_space_summary.json."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory for compatibility-conditioned donor audit "
            "outputs. If omitted, uses "
            "<nnunet_run_root>/compatibility_conditioned_donor_audit."
        ),
    )

    parser.add_argument(
        "--brain-threshold",
        type=float,
        default=0.05,
    )

    parser.add_argument(
        "--min-overlap",
        type=float,
        default=0.80,
    )

    parser.add_argument(
        "--external-modality",
        default="flair",
    )

    parser.add_argument(
        "--device",
        choices=[
            "auto",
            "cpu",
            "mps",
            "cuda",
        ],
        default="auto",
    )

    parser.add_argument(
        "--base-batch-size",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--donor-batch-size",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
    )

    return parser.parse_args()


def resolve_file(
    path: Path,
    *,
    name: str,
) -> Path:
    """Resolve and require one file."""

    resolved = (
        path
        .expanduser()
        .resolve()
    )

    if not resolved.is_file():
        raise FileNotFoundError(
            f"{name} does not exist:\n"
            f"{resolved}"
        )

    return resolved


def sha256_file(
    path: Path,
) -> str:
    """Return file SHA-256."""

    digest = hashlib.sha256()

    with path.open(
        "rb"
    ) as file:
        for block in iter(
            lambda: file.read(
                1024
                * 1024
            ),
            b"",
        ):
            digest.update(
                block
            )

    return digest.hexdigest()


def git_commit() -> str | None:
    """Return current Git commit."""

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


def git_worktree_clean() -> bool | None:
    """Return whether Git worktree is clean."""

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

    return (
        result.stdout.strip()
        == ""
    )


def json_safe(
    value: Any,
) -> Any:
    """Convert values to strict JSON-compatible objects."""

    if isinstance(
        value,
        dict,
    ):
        return {
            str(key): json_safe(
                item
            )
            for key, item in value.items()
        }

    if isinstance(
        value,
        (
            list,
            tuple,
        ),
    ):
        return [
            json_safe(
                item
            )
            for item in value
        ]

    if isinstance(
        value,
        np.integer,
    ):
        return int(
            value
        )

    if isinstance(
        value,
        np.floating,
    ):
        value = float(
            value
        )

    if isinstance(
        value,
        float,
    ):
        if not np.isfinite(
            value
        ):
            return None

        return value

    if isinstance(
        value,
        np.bool_,
    ):
        return bool(
            value
        )

    return value


def require_columns(
    table: pd.DataFrame,
    columns: set[str],
    *,
    name: str,
) -> None:
    """Require table columns."""

    missing = sorted(
        columns
        - set(
            table.columns
        )
    )

    if missing:
        raise ConditionedDonorAuditError(
            f"{name} is missing required column(s): "
            + ", ".join(
                missing
            )
        )


def weighted_mean(
    values: np.ndarray,
    weights: np.ndarray,
) -> float:
    """Weighted mean."""

    return float(
        np.sum(
            values
            * weights
        )
        / np.sum(
            weights
        )
    )


def weighted_variance(
    values: np.ndarray,
    weights: np.ndarray,
) -> float:
    """Population weighted variance."""

    mean = weighted_mean(
        values,
        weights,
    )

    return float(
        np.sum(
            weights
            * (
                values
                - mean
            ) ** 2
        )
        / np.sum(
            weights
        )
    )


def weighted_quantile(
    values: np.ndarray,
    weights: np.ndarray,
    quantile: float,
) -> float:
    """Weighted empirical quantile."""

    order = np.argsort(
        values,
        kind="stable",
    )

    sorted_values = values[
        order
    ]

    sorted_weights = weights[
        order
    ]

    cumulative = np.cumsum(
        sorted_weights
    )

    cumulative = (
        cumulative
        / cumulative[
            -1
        ]
    )

    position = int(
        np.searchsorted(
            cumulative,
            quantile,
            side="left",
        )
    )

    position = min(
        position,
        len(
            sorted_values
        )
        - 1,
    )

    return float(
        sorted_values[
            position
        ]
    )


def weighted_cdf_at(
    reference_values: np.ndarray,
    reference_weights: np.ndarray,
    query: np.ndarray,
) -> np.ndarray:
    """Evaluate weighted empirical CDF at query points."""

    order = np.argsort(
        reference_values,
        kind="stable",
    )

    values = reference_values[
        order
    ]

    weights = reference_weights[
        order
    ]

    cumulative = np.cumsum(
        weights
    )

    cumulative = (
        cumulative
        / cumulative[
            -1
        ]
    )

    positions = np.searchsorted(
        values,
        query,
        side="right",
    )

    result = np.zeros(
        len(
            query
        ),
        dtype=np.float64,
    )

    valid = (
        positions
        > 0
    )

    result[
        valid
    ] = cumulative[
        positions[
            valid
        ]
        - 1
    ]

    return result


def weighted_ks_distance(
    reference_values: np.ndarray,
    reference_weights: np.ndarray,
    observed_values: np.ndarray,
) -> float:
    """KS distance between weighted reference and unweighted sample."""

    query = np.unique(
        np.concatenate(
            [
                reference_values,
                observed_values,
            ]
        )
    )

    reference_cdf = weighted_cdf_at(
        reference_values,
        reference_weights,
        query,
    )

    observed_sorted = np.sort(
        observed_values
    )

    observed_cdf = (
        np.searchsorted(
            observed_sorted,
            query,
            side="right",
        )
        / len(
            observed_sorted
        )
    )

    return float(
        np.max(
            np.abs(
                reference_cdf
                - observed_cdf
            )
        )
    )


def categorical_probabilities(
    values: np.ndarray,
    weights: np.ndarray,
) -> dict[str, float]:
    """Weighted categorical probabilities."""

    result: dict[
        str,
        float,
    ] = {}

    total = float(
        weights.sum()
    )

    for category in sorted(
        str(
            value
        )
        for value in np.unique(
            values
        )
    ):
        mask = (
            values.astype(
                str
            )
            == category
        )

        result[
            category
        ] = float(
            weights[
                mask
            ].sum()
            / total
        )

    return result


def total_variation_distance(
    reference: dict[str, float],
    observed: dict[str, float],
) -> float:
    """Categorical total-variation distance."""

    categories = (
        set(
            reference
        )
        | set(
            observed
        )
    )

    return float(
        0.5
        * sum(
            abs(
                reference.get(
                    category,
                    0.0,
                )
                - observed.get(
                    category,
                    0.0,
                )
            )
            for category in categories
        )
    )


def build_reference_weights(
    *,
    base_indices: np.ndarray,
    compatibility_lists: list[np.ndarray],
    donor_volumes: np.ndarray,
    donor_count: int,
) -> np.ndarray:
    """
    Construct exact hierarchical donor-reference weights.

    Each selected external base contributes equal mass.

    Within a base:
      compatible training volumes receive equal probability,
      then compatible donor slices within a selected volume receive
      equal probability.
    """

    weights = np.zeros(
        donor_count,
        dtype=np.float64,
    )

    for base_index in base_indices:
        compatible = compatibility_lists[
            int(
                base_index
            )
        ]

        volumes = donor_volumes[
            compatible
        ]

        unique_volumes, inverse, counts = np.unique(
            volumes,
            return_inverse=True,
            return_counts=True,
        )

        if unique_volumes.size == 0:
            raise ConditionedDonorAuditError(
                "A selected external base has no compatible training volume."
            )

        donor_probability = (
            1.0
            / float(
                unique_volumes.size
            )
            / counts[
                inverse
            ].astype(
                np.float64
            )
        )

        if not np.isclose(
            donor_probability.sum(),
            1.0,
            rtol=0.0,
            atol=1e-12,
        ):
            raise ConditionedDonorAuditError(
                "Per-base hierarchical donor probabilities "
                "do not sum to one."
            )

        weights[
            compatible
        ] += donor_probability

    weights /= float(
        len(
            base_indices
        )
    )

    if not np.isclose(
        weights.sum(),
        1.0,
        rtol=0.0,
        atol=1e-10,
    ):
        raise ConditionedDonorAuditError(
            "Compatibility-conditioned donor weights "
            "do not sum to one."
        )

    return weights


def reconstruct_compatibility_graph(
    *,
    assignments: pd.DataFrame,
    donors: pd.DataFrame,
    validation_dataset: Any,
    packed_donor_masks: np.ndarray,
    device: torch.device,
    brain_threshold: float,
    min_overlap: float,
    external_modality: str,
    base_batch_size: int,
    donor_batch_size: int,
) -> tuple[
    pd.DataFrame,
    np.ndarray,
    list[np.ndarray],
]:
    """Reconstruct exact compatibility lists for all 625 selected bases."""

    selected = (
        assignments.sort_values(
            [
                "base_rank",
                "external_subject_numeric_id",
            ],
            kind="stable",
        )
        .reset_index(
            drop=True
        )
    )

    base_table = pd.DataFrame(
        {
            "case_id":
                selected[
                    "case_id"
                ],

            "base_rank":
                selected[
                    "base_rank"
                ],

            "subject_numeric_id":
                selected[
                    "external_subject_numeric_id"
                ],

            "slice_index":
                selected[
                    "external_slice_index"
                ],

            "brain_pixels":
                selected[
                    "external_brain_pixels"
                ],

            "compatible_donor_count":
                selected[
                    "external_compatible_donor_count"
                ],
        }
    )

    brain_masks = load_selected_base_brain_masks(
        selected_bases=base_table,
        validation_dataset=validation_dataset,
        external_modality=external_modality,
        brain_threshold=brain_threshold,
    )

    donor_areas = donors[
        "tumor_area_pixels"
    ].to_numpy(
        dtype=np.int64
    )

    compatibility_lists: list[
        np.ndarray
    ] = []

    print()
    print(
        "Reconstructing complete selected-base compatibility graph..."
    )

    for base_start in range(
        0,
        len(
            base_table
        ),
        base_batch_size,
    ):
        base_end = min(
            base_start
            + base_batch_size,
            len(
                base_table
            ),
        )

        lists = compute_compatible_donor_indices_for_batch(
            base_masks=brain_masks[
                base_start:base_end
            ],
            packed_donor_masks=packed_donor_masks,
            donor_areas=donor_areas,
            device=device,
            min_overlap=min_overlap,
            donor_batch_size=donor_batch_size,
        )

        for local_index, compatible in enumerate(
            lists
        ):
            global_index = (
                base_start
                + local_index
            )

            expected = int(
                base_table.iloc[
                    global_index
                ][
                    "compatible_donor_count"
                ]
            )

            observed = int(
                compatible.size
            )

            if observed != expected:
                raise ConditionedDonorAuditError(
                    "Recomputed compatibility count disagrees with "
                    "candidate assignment provenance.\n"
                    f"Case: "
                    f"{base_table.iloc[global_index]['case_id']}\n"
                    f"Observed: {observed:,}\n"
                    f"Expected: {expected:,}"
                )

            compatibility_lists.append(
                compatible.astype(
                    np.int64,
                    copy=False,
                )
            )

        print(
            f"  Compatibility graph: "
            f"{base_end:,} / "
            f"{len(base_table):,} bases",
            flush=True,
        )

    if len(
        compatibility_lists
    ) != len(
        base_table
    ):
        raise ConditionedDonorAuditError(
            "Compatibility graph row count mismatch."
        )

    return (
        selected,
        brain_masks,
        compatibility_lists,
    )


def build_numeric_comparison(
    *,
    assignments: pd.DataFrame,
    donors: pd.DataFrame,
    compatibility_lists: list[np.ndarray],
) -> tuple[
    pd.DataFrame,
    dict[str, np.ndarray],
]:
    """Build compatibility-conditioned numeric comparisons."""

    donor_volumes = donors[
        "volume"
    ].to_numpy(
        dtype=np.int64
    )

    rows: list[
        dict[str, Any]
    ] = []

    reference_weights_by_candidate: dict[
        str,
        np.ndarray
    ] = {}

    for candidate_name, quota in (
        CANDIDATE_QUOTAS.items()
    ):
        base_indices = np.flatnonzero(
            assignments[
                "base_rank"
            ].to_numpy(
                dtype=np.int64
            )
            <= quota
        )

        candidate = assignments.iloc[
            base_indices
        ]

        reference_weights = build_reference_weights(
            base_indices=base_indices,
            compatibility_lists=compatibility_lists,
            donor_volumes=donor_volumes,
            donor_count=len(
                donors
            ),
        )

        reference_weights_by_candidate[
            candidate_name
        ] = reference_weights

        positive_reference = (
            reference_weights
            > 0
        )

        for donor_column, assignment_column in (
            NUMERIC_DONOR_METRICS.items()
        ):
            reference_values = donors[
                donor_column
            ].to_numpy(
                dtype=np.float64
            )

            observed_values = candidate[
                assignment_column
            ].to_numpy(
                dtype=np.float64
            )

            reference_mean = weighted_mean(
                reference_values,
                reference_weights,
            )

            reference_variance = weighted_variance(
                reference_values,
                reference_weights,
            )

            reference_sd = float(
                np.sqrt(
                    reference_variance
                )
            )

            observed_mean = float(
                observed_values.mean()
            )

            if reference_sd > 0:
                smd = float(
                    (
                        observed_mean
                        - reference_mean
                    )
                    / reference_sd
                )
            else:
                smd = 0.0

            rows.append(
                {
                    "candidate":
                        candidate_name,

                    "metric":
                        donor_column,

                    "case_count":
                        len(
                            candidate
                        ),

                    "reference_supported_donor_count":
                        int(
                            positive_reference.sum()
                        ),

                    "reference_mean":
                        reference_mean,

                    "observed_mean":
                        observed_mean,

                    "reference_median":
                        weighted_quantile(
                            reference_values,
                            reference_weights,
                            0.5,
                        ),

                    "observed_median":
                        float(
                            np.median(
                                observed_values
                            )
                        ),

                    "reference_standard_deviation":
                        reference_sd,

                    "standardized_mean_difference":
                        smd,

                    "weighted_ks_distance":
                        weighted_ks_distance(
                            reference_values,
                            reference_weights,
                            observed_values,
                        ),
                }
            )

    return (
        pd.DataFrame(
            rows
        ),
        reference_weights_by_candidate,
    )


def build_laterality_comparison(
    *,
    assignments: pd.DataFrame,
    donors: pd.DataFrame,
    reference_weights_by_candidate: dict[str, np.ndarray],
) -> pd.DataFrame:
    """Compare assigned and conditioned-reference laterality."""

    rows = []

    donor_laterality = donors[
        "centroid_laterality"
    ].astype(
        str
    ).to_numpy()

    for candidate_name, quota in (
        CANDIDATE_QUOTAS.items()
    ):
        candidate = assignments.loc[
            assignments[
                "base_rank"
            ]
            <= quota
        ]

        reference = categorical_probabilities(
            donor_laterality,
            reference_weights_by_candidate[
                candidate_name
            ],
        )

        observed_counts = (
            candidate[
                "donor_centroid_laterality"
            ]
            .astype(
                str
            )
            .value_counts(
                normalize=True
            )
            .to_dict()
        )

        tv = total_variation_distance(
            reference,
            observed_counts,
        )

        categories = sorted(
            set(
                reference
            )
            | set(
                observed_counts
            )
        )

        for category in categories:
            rows.append(
                {
                    "candidate":
                        candidate_name,

                    "category":
                        category,

                    "reference_probability":
                        float(
                            reference.get(
                                category,
                                0.0,
                            )
                        ),

                    "observed_probability":
                        float(
                            observed_counts.get(
                                category,
                                0.0,
                            )
                        ),

                    "total_variation_distance":
                        tv,
                }
            )

    return pd.DataFrame(
        rows
    )


def build_volume_comparison(
    *,
    assignments: pd.DataFrame,
    donors: pd.DataFrame,
    reference_weights_by_candidate: dict[str, np.ndarray],
) -> pd.DataFrame:
    """Compare observed and compatibility-conditioned training-volume use."""

    rows = []

    donor_volumes = donors[
        "volume"
    ].to_numpy(
        dtype=np.int64
    )

    all_volumes = np.sort(
        np.unique(
            donor_volumes
        )
    )

    for candidate_name, quota in (
        CANDIDATE_QUOTAS.items()
    ):
        candidate = assignments.loc[
            assignments[
                "base_rank"
            ]
            <= quota
        ]

        reference_weights = (
            reference_weights_by_candidate[
                candidate_name
            ]
        )

        observed_counts = (
            candidate[
                "donor_volume"
            ]
            .value_counts()
        )

        case_count = len(
            candidate
        )

        for volume in all_volumes:
            reference_probability = float(
                reference_weights[
                    donor_volumes
                    == volume
                ].sum()
            )

            observed_count = int(
                observed_counts.get(
                    volume,
                    0,
                )
            )

            rows.append(
                {
                    "candidate":
                        candidate_name,

                    "volume":
                        int(
                            volume
                        ),

                    "reference_probability":
                        reference_probability,

                    "reference_expected_case_count":
                        reference_probability
                        * case_count,

                    "observed_case_count":
                        observed_count,

                    "observed_probability":
                        observed_count
                        / case_count,
                }
            )

    return pd.DataFrame(
        rows
    )


def ecdf_coordinates(
    values: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
]:
    """Ordinary ECDF coordinates."""

    x = np.sort(
        np.asarray(
            values,
            dtype=np.float64,
        )
    )

    y = (
        np.arange(
            1,
            len(
                x
            )
            + 1,
            dtype=np.float64,
        )
        / len(
            x
        )
    )

    return (
        x,
        y,
    )


def weighted_ecdf_coordinates(
    values: np.ndarray,
    weights: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
]:
    """Weighted ECDF coordinates."""

    order = np.argsort(
        values,
        kind="stable",
    )

    x = values[
        order
    ]

    w = weights[
        order
    ]

    positive = (
        w
        > 0
    )

    x = x[
        positive
    ]

    w = w[
        positive
    ]

    y = np.cumsum(
        w
    )

    y /= y[
        -1
    ]

    return (
        x,
        y,
    )


def save_metric_figures(
    *,
    assignments: pd.DataFrame,
    donors: pd.DataFrame,
    reference_weights_by_candidate: dict[str, np.ndarray],
    figure_dir: Path,
    dpi: int,
) -> None:
    """Save conditioned-reference versus observed ECDFs."""

    figure_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    plotted_metrics = (
        "tumor_area_pixels",
        "bbox_area",
        "connected_component_count",
        "largest_component_fraction",
        "centroid_x_normalized",
        "centroid_y_normalized",
        "compatible_external_base_count",
    )

    for candidate_name, quota in (
        CANDIDATE_QUOTAS.items()
    ):
        candidate = assignments.loc[
            assignments[
                "base_rank"
            ]
            <= quota
        ]

        weights = reference_weights_by_candidate[
            candidate_name
        ]

        for donor_column in plotted_metrics:
            assignment_column = (
                NUMERIC_DONOR_METRICS[
                    donor_column
                ]
            )

            reference_values = donors[
                donor_column
            ].to_numpy(
                dtype=np.float64
            )

            observed_values = candidate[
                assignment_column
            ].to_numpy(
                dtype=np.float64
            )

            reference_x, reference_y = (
                weighted_ecdf_coordinates(
                    reference_values,
                    weights,
                )
            )

            observed_x, observed_y = (
                ecdf_coordinates(
                    observed_values
                )
            )

            figure, axis = plt.subplots(
                figsize=(
                    8,
                    5,
                )
            )

            axis.plot(
                reference_x,
                reference_y,
                label=(
                    "Compatibility-conditioned "
                    "reference"
                ),
            )

            axis.plot(
                observed_x,
                observed_y,
                label="Assigned donors",
            )

            axis.set_xlabel(
                donor_column
            )

            axis.set_ylabel(
                "Empirical cumulative probability"
            )

            axis.set_title(
                f"{candidate_name}: {donor_column}"
            )

            axis.legend()

            figure.tight_layout()

            figure.savefig(
                figure_dir
                / (
                    f"{candidate_name}_"
                    f"{donor_column}_ecdf.png"
                ),
                dpi=dpi,
                bbox_inches="tight",
            )

            plt.close(
                figure
            )


def main() -> None:
    """Run audit."""

    args = parse_args()

    folders_config = load_folders_config(
        args.folders_file
    )

    validation_path = resolve_file(
        resolve_path(
            key="yaml_validation_dataset_path",
            cli_value=args.validation_dataset,
            config=folders_config,
            selector=None,
        ),
        name="Validation dataset",
    )

    nnunet_run_root = None

    if (
        args.candidate_assignments is None
        or args.donor_morphology_csv is None
        or args.pair_space_summary is None
        or args.output_dir is None
    ):
        nnunet_run_root = resolve_path(
            key="nnunet_run_root",
            cli_value=None,
            config=folders_config,
            selector=None,
        ).expanduser().resolve()

    if args.candidate_assignments is not None:
        assignments_path = resolve_file(
            args.candidate_assignments,
            name="Candidate assignments",
        )
    else:
        assignments_path = resolve_file(
            nnunet_run_root
            / "external_manifest_design_audit"
            / "candidate_assignments_all.csv",
            name="Candidate assignments",
        )

    if args.donor_morphology_csv is not None:
        donors_path = resolve_file(
            args.donor_morphology_csv,
            name="Donor morphology CSV",
        )
    else:
        donors_path = resolve_file(
            nnunet_run_root
            / "donor_morphology_audit"
            / "donor_morphology.csv",
            name="Donor morphology CSV",
        )

    if args.pair_space_summary is not None:
        pair_summary_path = resolve_file(
            args.pair_space_summary,
            name="Pair-space summary",
        )
    else:
        pair_summary_path = resolve_file(
            nnunet_run_root
            / "external_pair_space_audit"
            / "external_pair_space_summary.json",
            name="Pair-space summary",
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
            / "compatibility_conditioned_donor_audit"
        )

    save_folders_config(
        args.folders_file,
        folders_config,
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_paths = [
        output_dir
        / OUTPUT_COMPARISON,

        output_dir
        / OUTPUT_LATERALITY,

        output_dir
        / OUTPUT_VOLUME,

        output_dir
        / OUTPUT_SUMMARY,
    ]

    existing = [
        path
        for path in output_paths
        if path.exists()
    ]

    if existing:
        raise ConditionedDonorAuditError(
            "Refusing to overwrite existing output(s):\n"
            + "\n".join(
                str(
                    path
                )
                for path in existing
            )
        )

    assignments = pd.read_csv(
        assignments_path
    )

    donors = pd.read_csv(
        donors_path
    )

    require_columns(
        assignments,
        {
            "case_id",
            "base_rank",
            "external_subject_numeric_id",
            "external_slice_index",
            "external_modality",
            "external_brain_pixels",
            "external_compatible_donor_count",
            "donor_index",
            "donor_h5_path",
            "donor_volume",
            "donor_tumor_area_pixels",
            "donor_tumor_area_fraction",
            "donor_bbox_area",
            "donor_bbox_fill_fraction",
            "donor_centroid_x_normalized",
            "donor_centroid_y_normalized",
            "donor_centroid_laterality",
            "donor_connected_component_count",
            "donor_largest_component_fraction",
            "donor_compatible_external_base_count",
        },
        name="Candidate assignments",
    )

    require_columns(
        donors,
        {
            "volume",
            "tumor_area_pixels",
            "tumor_area_fraction",
            "bbox_area",
            "bbox_fill_fraction",
            "centroid_x_normalized",
            "centroid_y_normalized",
            "centroid_laterality",
            "connected_component_count",
            "largest_component_fraction",
            "compatible_external_base_count",
            "donor_h5_path",
        },
        name="Donor morphology CSV",
    )

    if len(
        assignments
    ) != 625:
        raise ConditionedDonorAuditError(
            "Expected exactly 625 candidate assignments."
        )

    if assignments[
        "donor_h5_path"
    ].duplicated().any():
        raise ConditionedDonorAuditError(
            "Candidate assignments contain donor reuse."
        )

    assignments = (
        assignments.sort_values(
            [
                "base_rank",
                "external_subject_numeric_id",
            ],
            kind="stable",
        )
        .reset_index(
            drop=True
        )
    )

    with pair_summary_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        pair_summary = json.load(
            file
        )

    expected_threshold = float(
        pair_summary[
            "audit_definition"
        ][
            "brain_threshold"
        ]
    )

    expected_overlap = float(
        pair_summary[
            "audit_definition"
        ][
            "minimum_mask_inside_brain_fraction"
        ]
    )

    if not np.isclose(
        args.brain_threshold,
        expected_threshold,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ConditionedDonorAuditError(
            "Brain threshold disagrees with pair-space audit."
        )

    if not np.isclose(
        args.min_overlap,
        expected_overlap,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ConditionedDonorAuditError(
            "Minimum overlap disagrees with pair-space audit."
        )

    device = resolve_device(
        args.device
    )

    validation_dataset = (
        load_validation_dataset_specification(
            validation_path
        )
    )

    print()
    print(
        "=" * 78
    )

    print(
        "COMPATIBILITY-CONDITIONED DONOR-SELECTION AUDIT"
    )

    print(
        "=" * 78
    )

    print(
        f"Selected external bases  : "
        f"{len(assignments):,}"
    )

    print(
        f"Eligible donor slices    : "
        f"{len(donors):,}"
    )

    print(
        f"Training volumes         : "
        f"{donors['volume'].nunique():,}"
    )

    print(
        f"Reference mechanism      : "
        "base -> uniform compatible volume -> uniform donor within volume"
    )

    print(
        f"Brain threshold          : "
        f"{args.brain_threshold}"
    )

    print(
        f"Minimum overlap          : "
        f"{args.min_overlap}"
    )

    print(
        f"Device                   : "
        f"{device}"
    )

    print(
        "=" * 78
    )

    packed_donor_masks = (
        load_and_pack_donor_masks(
            donors
        )
    )

    (
        assignments,
        _brain_masks,
        compatibility_lists,
    ) = reconstruct_compatibility_graph(
        assignments=assignments,
        donors=donors,
        validation_dataset=validation_dataset,
        packed_donor_masks=packed_donor_masks,
        device=device,
        brain_threshold=args.brain_threshold,
        min_overlap=args.min_overlap,
        external_modality=str(
            args.external_modality
        ).lower(),
        base_batch_size=args.base_batch_size,
        donor_batch_size=args.donor_batch_size,
    )

    # Confirm every actual matched donor remains an edge in the
    # reconstructed compatibility graph.
    for base_index, compatible in enumerate(
        compatibility_lists
    ):
        donor_index = int(
            assignments.iloc[
                base_index
            ][
                "donor_index"
            ]
        )

        if donor_index not in set(
            compatible.tolist()
        ):
            raise ConditionedDonorAuditError(
                "An assigned donor is not contained in the exact "
                "reconstructed compatibility set."
            )

    (
        comparison,
        reference_weights,
    ) = build_numeric_comparison(
        assignments=assignments,
        donors=donors,
        compatibility_lists=compatibility_lists,
    )

    laterality = build_laterality_comparison(
        assignments=assignments,
        donors=donors,
        reference_weights_by_candidate=reference_weights,
    )

    volume_distribution = (
        build_volume_comparison(
            assignments=assignments,
            donors=donors,
            reference_weights_by_candidate=reference_weights,
        )
    )

    comparison_path = (
        output_dir
        / OUTPUT_COMPARISON
    )

    laterality_path = (
        output_dir
        / OUTPUT_LATERALITY
    )

    volume_path = (
        output_dir
        / OUTPUT_VOLUME
    )

    comparison.to_csv(
        comparison_path,
        index=False,
    )

    laterality.to_csv(
        laterality_path,
        index=False,
    )

    volume_distribution.to_csv(
        volume_path,
        index=False,
    )

    figure_dir = (
        output_dir
        / FIGURE_DIR_NAME
    )

    save_metric_figures(
        assignments=assignments,
        donors=donors,
        reference_weights_by_candidate=reference_weights,
        figure_dir=figure_dir,
        dpi=args.dpi,
    )

    candidate_summary = []

    for candidate_name, quota in (
        CANDIDATE_QUOTAS.items()
    ):
        numeric = comparison.loc[
            comparison[
                "candidate"
            ]
            == candidate_name
        ]

        lateral = laterality.loc[
            laterality[
                "candidate"
            ]
            == candidate_name
        ]

        volume = volume_distribution.loc[
            volume_distribution[
                "candidate"
            ]
            == candidate_name
        ]

        volume_tv = float(
            0.5
            * np.abs(
                volume[
                    "reference_probability"
                ].to_numpy()
                - volume[
                    "observed_probability"
                ].to_numpy()
            ).sum()
        )

        candidate_summary.append(
            {
                "candidate":
                    candidate_name,

                "case_count":
                    int(
                        (
                            assignments[
                                "base_rank"
                            ]
                            <= quota
                        ).sum()
                    ),

                "maximum_absolute_standardized_mean_difference":
                    float(
                        numeric[
                            "standardized_mean_difference"
                        ].abs().max()
                    ),

                "maximum_weighted_ks_distance":
                    float(
                        numeric[
                            "weighted_ks_distance"
                        ].max()
                    ),

                "laterality_total_variation_distance":
                    float(
                        lateral[
                            "total_variation_distance"
                        ].iloc[
                            0
                        ]
                    ),

                "training_volume_total_variation_distance":
                    volume_tv,

                "observed_training_volumes":
                    int(
                        (
                            volume[
                                "observed_case_count"
                            ]
                            > 0
                        ).sum()
                    ),

                "reference_supported_training_volumes":
                    int(
                        (
                            volume[
                                "reference_probability"
                            ]
                            > 0
                        ).sum()
                    ),
            }
        )

    summary = {
        "audit_role":
            (
                "Descriptive audit of actual candidate donor assignments "
                "against compatibility-conditioned hierarchical donor "
                "reference distributions. No assignments are modified."
            ),

        "reference_mechanism":
            (
                "For each selected external base: choose uniformly among "
                "compatible training volumes, then uniformly among compatible "
                "donor slices within the selected volume. Each selected "
                "external base contributes equal probability mass."
            ),

        "candidate_summary":
            candidate_summary,

        "source_artifacts": {
            "validation_dataset":
                str(
                    validation_path
                ),

            "candidate_assignments":
                str(
                    assignments_path
                ),

            "donor_morphology_csv":
                str(
                    donors_path
                ),

            "pair_space_summary":
                str(
                    pair_summary_path
                ),
        },

        "output_artifacts": {
            "numeric_comparison":
                str(
                    comparison_path
                ),

            "laterality_comparison":
                str(
                    laterality_path
                ),

            "training_volume_distribution":
                str(
                    volume_path
                ),

            "figure_directory":
                str(
                    figure_dir
                ),
        },

        "selection_status": {
            "candidate_case_count_selected":
                False,

            "definitive_manifest_frozen":
                False,

            "donor_assignments_modified":
                False,

            "br_lora_external_evaluation_run":
                False,
        },

        "provenance": {
            "audited_at_utc":
                datetime.now(
                    timezone.utc
                ).isoformat(),

            "git_commit":
                git_commit(),

            "git_worktree_clean":
                git_worktree_clean(),

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

            "candidate_assignments_sha256":
                sha256_file(
                    assignments_path
                ),

            "donor_morphology_csv_sha256":
                sha256_file(
                    donors_path
                ),

            "pair_space_summary_sha256":
                sha256_file(
                    pair_summary_path
                ),

            "device":
                str(
                    device
                ),
        },
    }

    summary_path = (
        output_dir
        / OUTPUT_SUMMARY
    )

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            json_safe(
                summary
            ),
            file,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )

        file.write(
            "\n"
        )

    print()
    print(
        "=" * 78
    )

    print(
        "COMPATIBILITY-CONDITIONED DONOR-SELECTION AUDIT: PASS"
    )

    print(
        "=" * 78
    )

    print()
    print(
        pd.DataFrame(
            candidate_summary
        ).to_string(
            index=False
        )
    )

    print()
    print(
        f"Numeric comparison      : "
        f"{comparison_path}"
    )

    print(
        f"Laterality comparison   : "
        f"{laterality_path}"
    )

    print(
        f"Volume comparison       : "
        f"{volume_path}"
    )

    print(
        f"Summary                 : "
        f"{summary_path}"
    )

    print(
        f"Figures                 : "
        f"{figure_dir}"
    )

    print()
    print(
        "No donor assignments were modified."
    )

    print(
        "No definitive external manifest was selected."
    )

    print(
        "=" * 78
    )


if __name__ == "__main__":
    main()
