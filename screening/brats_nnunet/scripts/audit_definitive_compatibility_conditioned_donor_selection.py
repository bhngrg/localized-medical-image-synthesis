#!/usr/bin/env python3
"""
Audit the definitive 250-case external BR-LoRA donor assignments against the
established compatibility-conditioned hierarchical donor reference.

The definitive external cohort contains:

    125 validation subjects
    x
    2 external tumor-free base slices per subject
    =
    250 external cases

The external bases are the frozen rank-1 and rank-2 selections from the
completed nested candidate-design audit.

The donor assignments are the freshly rematched donors obtained from the
250-base-only global one-to-one matching. No donor assignments inherited from
the discarded 625-case candidate are used.

Reference mechanism
-------------------
For each definitive external base:

    1. identify all exactly compatible donor slices;
    2. choose a compatible training volume uniformly;
    3. choose a compatible donor slice uniformly within that volume.

Each external base contributes equal total probability mass.

This is the same compatibility-conditioned hierarchical donor reference used
in the completed candidate-design audit. The statistical definition is not
changed.

This script is descriptive only. It does not modify the frozen external base
selection, change donor assignments, or run BR-LoRA inference.
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

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )

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

import screening.brats_nnunet.scripts.audit_compatibility_conditioned_donor_selection as conditioned


FINAL_CASE_COUNT = 250
FINAL_SUBJECT_COUNT = 125
FINAL_BASES_PER_SUBJECT = 2
FINAL_COHORT_NAME = "definitive_250"

OUTPUT_COMPARISON = (
    "definitive_compatibility_conditioned_donor_comparison.csv"
)

OUTPUT_LATERALITY = (
    "definitive_compatibility_conditioned_laterality.csv"
)

OUTPUT_VOLUME = (
    "definitive_compatibility_conditioned_volume_distribution.csv"
)

OUTPUT_SUMMARY = (
    "definitive_compatibility_conditioned_donor_summary.json"
)

FIGURE_DIR_NAME = "figures"


class DefinitiveConditionedAuditError(
    RuntimeError
):
    """Raised when definitive compatibility-conditioned auditing fails."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit the definitive 250-case donor assignments against "
            "the established compatibility-conditioned donor reference."
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
        "--final-assignments",
        type=Path,
        default=None,
        help=(
            "external_evaluation_manifest_250_assignments.csv from "
            "the definitive 250-case finalization. If omitted, uses "
            "<nnunet_run_root>/definitive_external_manifest_250/"
            "external_evaluation_manifest_250_assignments.csv."
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
            "Directory for definitive compatibility-conditioned audit "
            "outputs. If omitted, uses "
            "<nnunet_run_root>/definitive_external_manifest_250/"
            "compatibility_conditioned_donor_audit."
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
    resolved = (
        path
        .expanduser()
        .resolve()
    )

    if not resolved.is_file():
        raise FileNotFoundError(
            f"{name} does not exist:\n{resolved}"
        )

    return resolved


def require_columns(
    table: pd.DataFrame,
    columns: set[str],
    *,
    name: str,
) -> None:
    missing = sorted(
        columns
        - set(
            table.columns
        )
    )

    if missing:
        raise DefinitiveConditionedAuditError(
            f"{name} is missing required column(s): "
            + ", ".join(
                missing
            )
        )


def sha256_file(
    path: Path,
) -> str:
    digest = hashlib.sha256()

    with path.open(
        "rb"
    ) as file:
        for block in iter(
            lambda: file.read(
                1024 * 1024
            ),
            b"",
        ):
            digest.update(
                block
            )

    return digest.hexdigest()


def resolve_git_commit() -> str | None:
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
    if isinstance(
        value,
        dict,
    ):
        return {
            str(key): json_safe(item)
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
            json_safe(item)
            for item in value
        ]

    if isinstance(
        value,
        np.integer,
    ):
        return int(value)

    if isinstance(
        value,
        np.floating,
    ):
        value = float(value)

    if isinstance(
        value,
        float,
    ):
        if not np.isfinite(value):
            return None

        return value

    if isinstance(
        value,
        np.bool_,
    ):
        return bool(value)

    return value


def normalize_final_assignments(
    *,
    final_assignments: pd.DataFrame,
    donors: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convert the definitive finalization table into the internal column contract
    already used by the established compatibility-conditioned audit.

    Donor morphology values are recovered from the independently audited donor
    morphology table using the frozen final donor H5 path.
    """

    require_columns(
        final_assignments,
        {
            "case_id",
            "base_rank",
            "external_subject_numeric_id",
            "external_slice_index",
            "external_modality",
            "external_brain_pixels",
            "compatible_donor_count",
            "final_donor_index",
            "final_donor_h5_path",
            "final_donor_volume",
            "final_donor_slice_index",
            "final_donor_tumor_area_pixels",
            "final_donor_bbox_area",
            "final_donor_component_count",
            "final_donor_largest_component_fraction",
            "final_donor_centroid_x_normalized",
            "final_donor_centroid_y_normalized",
            "final_donor_centroid_laterality",
        },
        name="Definitive assignment table",
    )

    require_columns(
        donors,
        {
            "donor_h5_path",
            "volume",
            "slice",
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
        },
        name="Donor morphology table",
    )

    if len(final_assignments) != FINAL_CASE_COUNT:
        raise DefinitiveConditionedAuditError(
            "Definitive assignment table must contain exactly "
            f"{FINAL_CASE_COUNT} rows.\n"
            f"Observed: {len(final_assignments)}"
        )

    if (
        final_assignments[
            "external_subject_numeric_id"
        ].nunique()
        != FINAL_SUBJECT_COUNT
    ):
        raise DefinitiveConditionedAuditError(
            "Definitive assignment table must contain exactly "
            f"{FINAL_SUBJECT_COUNT} validation subjects."
        )

    subject_counts = (
        final_assignments.groupby(
            "external_subject_numeric_id"
        )
        .size()
    )

    if not (
        subject_counts
        == FINAL_BASES_PER_SUBJECT
    ).all():
        raise DefinitiveConditionedAuditError(
            "Every validation subject must contribute exactly "
            f"{FINAL_BASES_PER_SUBJECT} final external cases."
        )

    if not set(
        final_assignments[
            "base_rank"
        ].astype(
            int
        ).unique()
    ).issubset(
        {
            1,
            2,
        }
    ):
        raise DefinitiveConditionedAuditError(
            "Definitive assignments contain a base rank outside {1, 2}."
        )

    if final_assignments[
        "final_donor_h5_path"
    ].duplicated().any():
        raise DefinitiveConditionedAuditError(
            "Definitive assignments contain donor-slice reuse."
        )

    morphology = donors.copy()

    if morphology[
        "donor_h5_path"
    ].duplicated().any():
        raise DefinitiveConditionedAuditError(
            "Donor morphology table contains duplicate donor paths."
        )

    merged = final_assignments.merge(
        morphology[
            [
                "donor_h5_path",
                "volume",
                "slice",
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
            ]
        ],
        how="left",
        left_on="final_donor_h5_path",
        right_on="donor_h5_path",
        validate="one_to_one",
    )

    if merged[
        "donor_h5_path"
    ].isna().any():
        raise DefinitiveConditionedAuditError(
            "At least one frozen final donor path could not be found "
            "in the donor morphology audit."
        )

    # Cross-check fields independently retained during finalization.
    integer_checks = {
        "final_donor_volume":
            "volume",

        "final_donor_slice_index":
            "slice",

        "final_donor_tumor_area_pixels":
            "tumor_area_pixels",

        "final_donor_bbox_area":
            "bbox_area",

        "final_donor_component_count":
            "connected_component_count",
    }

    for final_column, morphology_column in (
        integer_checks.items()
    ):
        left = merged[
            final_column
        ].to_numpy()

        right = merged[
            morphology_column
        ].to_numpy()

        if not np.array_equal(
            left,
            right,
        ):
            raise DefinitiveConditionedAuditError(
                "Final assignment provenance disagrees with donor "
                f"morphology for {final_column!r}."
            )

    float_checks = {
        "final_donor_largest_component_fraction":
            "largest_component_fraction",

        "final_donor_centroid_x_normalized":
            "centroid_x_normalized",

        "final_donor_centroid_y_normalized":
            "centroid_y_normalized",
    }

    for final_column, morphology_column in (
        float_checks.items()
    ):
        left = merged[
            final_column
        ].to_numpy(
            dtype=np.float64
        )

        right = merged[
            morphology_column
        ].to_numpy(
            dtype=np.float64
        )

        if not np.allclose(
            left,
            right,
            rtol=0.0,
            atol=1e-12,
        ):
            raise DefinitiveConditionedAuditError(
                "Final assignment provenance disagrees with donor "
                f"morphology for {final_column!r}."
            )

    if not (
        merged[
            "final_donor_centroid_laterality"
        ].astype(
            str
        ).to_numpy()
        ==
        merged[
            "centroid_laterality"
        ].astype(
            str
        ).to_numpy()
    ).all():
        raise DefinitiveConditionedAuditError(
            "Final donor laterality disagrees with the donor morphology audit."
        )

    normalized = pd.DataFrame(
        {
            "case_id":
                merged[
                    "case_id"
                ],

            "base_rank":
                merged[
                    "base_rank"
                ].astype(
                    int
                ),

            "external_subject_numeric_id":
                merged[
                    "external_subject_numeric_id"
                ].astype(
                    int
                ),

            "external_slice_index":
                merged[
                    "external_slice_index"
                ].astype(
                    int
                ),

            "external_modality":
                merged[
                    "external_modality"
                ].astype(
                    str
                ),

            "external_brain_pixels":
                merged[
                    "external_brain_pixels"
                ].astype(
                    int
                ),

            "external_compatible_donor_count":
                merged[
                    "compatible_donor_count"
                ].astype(
                    int
                ),

            "donor_index":
                merged[
                    "final_donor_index"
                ].astype(
                    int
                ),

            "donor_h5_path":
                merged[
                    "final_donor_h5_path"
                ].astype(
                    str
                ),

            "donor_volume":
                merged[
                    "volume"
                ].astype(
                    int
                ),

            "donor_tumor_area_pixels":
                merged[
                    "tumor_area_pixels"
                ].astype(
                    int
                ),

            "donor_tumor_area_fraction":
                merged[
                    "tumor_area_fraction"
                ].astype(
                    float
                ),

            "donor_bbox_area":
                merged[
                    "bbox_area"
                ],

            "donor_bbox_fill_fraction":
                merged[
                    "bbox_fill_fraction"
                ].astype(
                    float
                ),

            "donor_centroid_x_normalized":
                merged[
                    "centroid_x_normalized"
                ].astype(
                    float
                ),

            "donor_centroid_y_normalized":
                merged[
                    "centroid_y_normalized"
                ].astype(
                    float
                ),

            "donor_centroid_laterality":
                merged[
                    "centroid_laterality"
                ].astype(
                    str
                ),

            "donor_connected_component_count":
                merged[
                    "connected_component_count"
                ].astype(
                    int
                ),

            "donor_largest_component_fraction":
                merged[
                    "largest_component_fraction"
                ].astype(
                    float
                ),

            "donor_compatible_external_base_count":
                merged[
                    "compatible_external_base_count"
                ].astype(
                    int
                ),
        }
    )

    normalized = (
        normalized.sort_values(
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

    return normalized


def main() -> None:
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
        args.final_assignments is None
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

    if args.final_assignments is not None:
        assignments_path = resolve_file(
            args.final_assignments,
            name="Definitive assignment table",
        )
    else:
        assignments_path = resolve_file(
            nnunet_run_root
            / "definitive_external_manifest_250"
            / "external_evaluation_manifest_250_assignments.csv",
            name="Definitive assignment table",
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
            / "definitive_external_manifest_250"
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

    summary_path = (
        output_dir
        / OUTPUT_SUMMARY
    )

    figure_dir = (
        output_dir
        / FIGURE_DIR_NAME
    )

    existing = [
        path
        for path in (
            comparison_path,
            laterality_path,
            volume_path,
            summary_path,
        )
        if path.exists()
    ]

    if existing:
        raise DefinitiveConditionedAuditError(
            "Refusing to overwrite existing definitive audit artifact(s):\n"
            + "\n".join(
                str(path)
                for path in existing
            )
        )

    raw_assignments = pd.read_csv(
        assignments_path
    )

    donors = pd.read_csv(
        donors_path
    )

    assignments = normalize_final_assignments(
        final_assignments=raw_assignments,
        donors=donors,
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
        raise DefinitiveConditionedAuditError(
            "Brain threshold disagrees with the completed pair-space audit."
        )

    if not np.isclose(
        args.min_overlap,
        expected_overlap,
        rtol=0.0,
        atol=1e-12,
    ):
        raise DefinitiveConditionedAuditError(
            "Minimum overlap disagrees with the completed pair-space audit."
        )

    external_modality = str(
        args.external_modality
    ).strip().lower()

    if not (
        assignments[
            "external_modality"
        ]
        .str.lower()
        .eq(
            external_modality
        )
        .all()
    ):
        raise DefinitiveConditionedAuditError(
            "Definitive assignments do not all use the requested "
            "external modality."
        )

    device = conditioned.resolve_device(
        args.device
    )

    validation_dataset = (
        load_validation_dataset_specification(
            validation_path
        )
    )

    # The reused comparison routines operate over CANDIDATE_QUOTAS.
    # For the definitive audit there is exactly one cohort and all rows
    # have base_rank <= 2.
    original_quotas = conditioned.CANDIDATE_QUOTAS

    conditioned.CANDIDATE_QUOTAS = {
        FINAL_COHORT_NAME:
            FINAL_BASES_PER_SUBJECT,
    }

    try:
        print()
        print(
            "=" * 78
        )

        print(
            "DEFINITIVE 250 COMPATIBILITY-CONDITIONED DONOR AUDIT"
        )

        print(
            "=" * 78
        )

        print(
            f"External cases           : "
            f"{len(assignments):,}"
        )

        print(
            f"Validation subjects      : "
            f"{assignments['external_subject_numeric_id'].nunique():,}"
        )

        print(
            f"Bases per subject        : "
            f"{FINAL_BASES_PER_SUBJECT}"
        )

        print(
            f"Unique frozen donors     : "
            f"{assignments['donor_h5_path'].nunique():,}"
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
            "Reference mechanism      : "
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
            conditioned.load_and_pack_donor_masks(
                donors
            )
        )

        (
            assignments,
            _brain_masks,
            compatibility_lists,
        ) = conditioned.reconstruct_compatibility_graph(
            assignments=assignments,
            donors=donors,
            validation_dataset=validation_dataset,
            packed_donor_masks=packed_donor_masks,
            device=device,
            brain_threshold=args.brain_threshold,
            min_overlap=args.min_overlap,
            external_modality=external_modality,
            base_batch_size=args.base_batch_size,
            donor_batch_size=args.donor_batch_size,
        )

        # Exact frozen-assignment compatibility audit.
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

            if not np.any(
                compatible
                == donor_index
            ):
                raise DefinitiveConditionedAuditError(
                    "A frozen final donor is not contained in the exact "
                    "reconstructed compatibility set.\n"
                    f"Case: "
                    f"{assignments.iloc[base_index]['case_id']}"
                )

        (
            comparison,
            reference_weights,
        ) = conditioned.build_numeric_comparison(
            assignments=assignments,
            donors=donors,
            compatibility_lists=compatibility_lists,
        )

        laterality = (
            conditioned.build_laterality_comparison(
                assignments=assignments,
                donors=donors,
                reference_weights_by_candidate=reference_weights,
            )
        )

        volume_distribution = (
            conditioned.build_volume_comparison(
                assignments=assignments,
                donors=donors,
                reference_weights_by_candidate=reference_weights,
            )
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

        conditioned.save_metric_figures(
            assignments=assignments,
            donors=donors,
            reference_weights_by_candidate=reference_weights,
            figure_dir=figure_dir,
            dpi=args.dpi,
        )

        numeric = comparison.loc[
            comparison[
                "candidate"
            ]
            == FINAL_COHORT_NAME
        ]

        lateral = laterality.loc[
            laterality[
                "candidate"
            ]
            == FINAL_COHORT_NAME
        ]

        volume = volume_distribution.loc[
            volume_distribution[
                "candidate"
            ]
            == FINAL_COHORT_NAME
        ]

        if numeric.empty:
            raise DefinitiveConditionedAuditError(
                "Definitive numeric comparison is empty."
            )

        if lateral.empty:
            raise DefinitiveConditionedAuditError(
                "Definitive laterality comparison is empty."
            )

        if volume.empty:
            raise DefinitiveConditionedAuditError(
                "Definitive training-volume comparison is empty."
            )

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

        maximum_absolute_smd = float(
            numeric[
                "standardized_mean_difference"
            ].abs().max()
        )

        maximum_ks = float(
            numeric[
                "weighted_ks_distance"
            ].max()
        )

        laterality_tv = float(
            lateral[
                "total_variation_distance"
            ].iloc[
                0
            ]
        )

        observed_training_volumes = int(
            (
                volume[
                    "observed_case_count"
                ]
                > 0
            ).sum()
        )

        reference_supported_training_volumes = int(
            (
                volume[
                    "reference_probability"
                ]
                > 0
            ).sum()
        )

        summary = {
            "audit_role":
                (
                    "Final descriptive audit of the frozen 250-case donor "
                    "assignments against the established compatibility-"
                    "conditioned hierarchical donor reference."
                ),

            "cohort": {
                "name":
                    FINAL_COHORT_NAME,

                "case_count":
                    FINAL_CASE_COUNT,

                "validation_subject_count":
                    FINAL_SUBJECT_COUNT,

                "bases_per_validation_subject":
                    FINAL_BASES_PER_SUBJECT,

                "unique_donor_slice_count":
                    int(
                        assignments[
                            "donor_h5_path"
                        ].nunique()
                    ),
            },

            "reference_mechanism":
                (
                    "For each frozen external base: choose uniformly among "
                    "compatible training volumes, then uniformly among "
                    "compatible donor slices within the selected volume. "
                    "Each external base contributes equal probability mass."
                ),

            "results": {
                "maximum_absolute_standardized_mean_difference":
                    maximum_absolute_smd,

                "maximum_weighted_ks_distance":
                    maximum_ks,

                "laterality_total_variation_distance":
                    laterality_tv,

                "training_volume_total_variation_distance":
                    volume_tv,

                "observed_training_volumes":
                    observed_training_volumes,

                "reference_supported_training_volumes":
                    reference_supported_training_volumes,
            },

            "eligibility_rules": {
                "brain_threshold":
                    float(
                        args.brain_threshold
                    ),

                "minimum_mask_inside_brain_fraction":
                    float(
                        args.min_overlap
                    ),

                "external_modality":
                    external_modality,

                "external_base":
                    (
                        "tumor_free_candidate == True, "
                        "predicted_tumor_pixels == 0, "
                        "compatible_donor_count > 0"
                    ),

                "donor":
                    (
                        "whole_tumor_pixels >= 300 and established "
                        "mask-margin rule already passed"
                    ),
            },

            "audit_checks": {
                "all_250_frozen_donors_are_exact_compatibility_edges":
                    True,

                "unique_donor_slices":
                    True,

                "exactly_two_bases_per_validation_subject":
                    True,

                "donor_morphology_provenance_crosscheck":
                    True,
            },

            "source_artifacts": {
                "validation_dataset":
                    str(
                        validation_path
                    ),

                "definitive_assignments":
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

            "provenance": {
                "audited_at_utc":
                    datetime.now(
                        timezone.utc
                    ).isoformat(),

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

                "definitive_assignments_sha256":
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

            "status": {
                "definitive_base_selection_modified":
                    False,

                "definitive_donor_assignments_modified":
                    False,

                "br_lora_external_evaluation_run":
                    False,

                "audit_passed":
                    True,
            },
        }

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
            "DEFINITIVE 250 COMPATIBILITY-CONDITIONED DONOR AUDIT: PASS"
        )

        print(
            "=" * 78
        )

        print(
            f"Cases                    : "
            f"{FINAL_CASE_COUNT}"
        )

        print(
            f"Validation subjects      : "
            f"{FINAL_SUBJECT_COUNT}"
        )

        print(
            f"Unique donor slices      : "
            f"{assignments['donor_h5_path'].nunique()}"
        )

        print(
            f"All donor edges exact    : True"
        )

        print(
            f"Max |SMD|                : "
            f"{maximum_absolute_smd:.6f}"
        )

        print(
            f"Max weighted KS          : "
            f"{maximum_ks:.6f}"
        )

        print(
            f"Laterality TV            : "
            f"{laterality_tv:.6f}"
        )

        print(
            f"Training-volume TV       : "
            f"{volume_tv:.6f}"
        )

        print(
            f"Observed train volumes   : "
            f"{observed_training_volumes}"
        )

        print()
        print(
            f"Numeric comparison       : "
            f"{comparison_path}"
        )

        print(
            f"Laterality comparison    : "
            f"{laterality_path}"
        )

        print(
            f"Volume comparison        : "
            f"{volume_path}"
        )

        print(
            f"Summary                  : "
            f"{summary_path}"
        )

        print(
            f"Figures                  : "
            f"{figure_dir}"
        )

        print()
        print(
            "No external bases were changed."
        )

        print(
            "No donor assignments were changed."
        )

        print(
            "=" * 78
        )

    finally:
        conditioned.CANDIDATE_QUOTAS = (
            original_quotas
        )


if __name__ == "__main__":
    main()
