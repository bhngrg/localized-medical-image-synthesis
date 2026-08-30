#!/usr/bin/env python3
"""
Analyze the audited external-base / training-donor compatibility space.

This script is descriptive only.

It does not:
- run nnU-Net,
- recompute base-donor compatibility,
- select the definitive external evaluation cohort,
- assign donors to external bases, or
- run BR-LoRA inference.

Inputs
------
1. validation_slice_screening.csv
2. external_base_compatibility_counts.csv
3. external_donor_compatibility_counts.csv
4. external_pair_space_summary.json

Outputs
-------
subject_summary.csv
slice_summary.csv
rejected_base_summary.csv
cohort_balance_summary.csv
analysis_summary.json

figures/
    eligible_bases_per_subject.png
    eligible_bases_by_slice.png
    compatible_donor_histogram.png
    compatible_donor_ecdf.png
    brain_pixels_vs_donors.png
    slice_index_vs_brain_pixels.png
    slice_index_vs_compatible_donors.png
    accepted_vs_rejected_brain_pixels.png
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
    resolve_path as resolve_config_path,
    save_folders_config,
)


SUBJECT_SUMMARY_NAME = "subject_summary.csv"
SLICE_SUMMARY_NAME = "slice_summary.csv"
REJECTED_BASE_SUMMARY_NAME = "rejected_base_summary.csv"
COHORT_BALANCE_SUMMARY_NAME = "cohort_balance_summary.csv"
ANALYSIS_SUMMARY_NAME = "analysis_summary.json"

FIGURE_DIR_NAME = "figures"

FIGURE_NAMES = (
    "eligible_bases_per_subject.png",
    "eligible_bases_by_slice.png",
    "compatible_donor_histogram.png",
    "compatible_donor_ecdf.png",
    "brain_pixels_vs_donors.png",
    "slice_index_vs_brain_pixels.png",
    "slice_index_vs_compatible_donors.png",
    "accepted_vs_rejected_brain_pixels.png",
)

DEFAULT_BALANCE_TARGETS = (
    1,
    2,
    5,
    10,
    20,
    30,
    40,
    50,
)


class PairSpaceAnalysisError(
    RuntimeError
):
    """Raised when external pair-space analysis fails."""


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Analyze the completed external BR-LoRA "
            "base-donor pair-space audit."
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
        "--screening-csv",
        type=Path,
        default=None,
        help=(
            "Slice-level validation screening CSV. If omitted, uses "
            "<nnunet_run_root>/validation_slice_screening/"
            "validation_slice_screening.csv."
        ),
    )

    parser.add_argument(
        "--base-counts-csv",
        type=Path,
        default=None,
        help=(
            "External-base compatibility-count CSV from "
            "audit_external_pair_space.py. If omitted, uses "
            "<nnunet_run_root>/external_pair_space_audit/"
            "external_base_compatibility_counts.csv."
        ),
    )

    parser.add_argument(
        "--donor-counts-csv",
        type=Path,
        default=None,
        help=(
            "Training-donor compatibility-count CSV from "
            "audit_external_pair_space.py. If omitted, uses "
            "<nnunet_run_root>/external_pair_space_audit/"
            "external_donor_compatibility_counts.csv."
        ),
    )

    parser.add_argument(
        "--pair-space-summary",
        type=Path,
        default=None,
        help=(
            "Pair-space summary JSON from audit_external_pair_space.py. "
            "If omitted, uses <nnunet_run_root>/external_pair_space_audit/"
            "external_pair_space_summary.json."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory for descriptive audit tables and figures. "
            "If omitted, uses "
            "<nnunet_run_root>/external_pair_space_analysis."
        ),
    )

    parser.add_argument(
        "--balance-targets",
        type=int,
        nargs="+",
        default=list(
            DEFAULT_BALANCE_TARGETS
        ),
        help=(
            "Per-subject eligible-base targets used "
            "only for feasibility summaries. "
            "Default: 1 2 5 10 20 30 40 50."
        ),
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help=(
            "Figure resolution. Default: 300."
        ),
    )

    return parser.parse_args()


def resolve_path(
    path: Path,
    *,
    kind: str,
) -> Path:
    """Resolve and require one existing file."""

    resolved = (
        path
        .expanduser()
        .resolve()
    )

    if not resolved.is_file():
        raise FileNotFoundError(
            f"{kind} does not exist:\n"
            f"{resolved}"
        )

    return resolved


def sha256_file(
    path: Path,
) -> str:
    """Return SHA-256 hash of one file."""

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
    """Return whether the current worktree is clean."""

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


def require_columns(
    table: pd.DataFrame,
    required: set[str],
    *,
    name: str,
) -> None:
    """Require table columns."""

    missing = sorted(
        required
        - set(
            table.columns
        )
    )

    if missing:
        raise PairSpaceAnalysisError(
            f"{name} is missing required column(s): "
            + ", ".join(
                missing
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
        np.isfinite(
            array
        )
    ]

    if array.size == 0:
        raise PairSpaceAnalysisError(
            "Cannot summarize an empty numeric vector."
        )

    return {
        "count":
            int(
                array.size
            ),

        "minimum":
            float(
                array.min()
            ),

        "q1":
            float(
                np.quantile(
                    array,
                    0.25,
                )
            ),

        "median":
            float(
                np.median(
                    array
                )
            ),

        "mean":
            float(
                array.mean()
            ),

        "standard_deviation":
            float(
                array.std(
                    ddof=1
                )
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
            float(
                array.max()
            ),

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


def refuse_existing_outputs(
    output_dir: Path,
) -> None:
    """Refuse to overwrite prior analysis outputs."""

    expected = [
        output_dir
        / SUBJECT_SUMMARY_NAME,

        output_dir
        / SLICE_SUMMARY_NAME,

        output_dir
        / REJECTED_BASE_SUMMARY_NAME,

        output_dir
        / COHORT_BALANCE_SUMMARY_NAME,

        output_dir
        / ANALYSIS_SUMMARY_NAME,
    ]

    figure_dir = (
        output_dir
        / FIGURE_DIR_NAME
    )

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
        raise PairSpaceAnalysisError(
            "Refusing to overwrite existing "
            "analysis output(s):\n"
            + "\n".join(
                str(
                    path
                )
                for path in existing
            )
        )


def build_subject_summary(
    *,
    screening: pd.DataFrame,
    bases: pd.DataFrame,
) -> pd.DataFrame:
    """Construct one row per validation subject."""

    tumor_free = (
        screening.loc[
            screening[
                "tumor_free_candidate"
            ].astype(
                bool
            )
        ]
        .groupby(
            [
                "subject_numeric_id",
                "subject",
            ],
            as_index=False,
        )
        .size()
        .rename(
            columns={
                "size":
                    "tumor_free_slice_count"
            }
        )
    )

    base_summary = (
        bases.groupby(
            [
                "subject_numeric_id",
                "subject",
            ],
            as_index=False,
        )
        .agg(
            screened_tumor_free_slice_count=(
                "slice_index",
                "size",
            ),

            eligible_base_count=(
                "composition_eligible",
                "sum",
            ),

            mean_compatible_donors=(
                "compatible_donor_count",
                "mean",
            ),

            median_compatible_donors=(
                "compatible_donor_count",
                "median",
            ),

            max_compatible_donors=(
                "compatible_donor_count",
                "max",
            ),

            mean_brain_pixels=(
                "brain_pixels",
                "mean",
            ),

            median_brain_pixels=(
                "brain_pixels",
                "median",
            ),
        )
    )

    result = tumor_free.merge(
        base_summary,
        on=[
            "subject_numeric_id",
            "subject",
        ],
        how="outer",
        validate="one_to_one",
    )

    numeric_fill_zero = (
        "tumor_free_slice_count",
        "screened_tumor_free_slice_count",
        "eligible_base_count",
    )

    for column in numeric_fill_zero:
        result[
            column
        ] = (
            result[
                column
            ]
            .fillna(
                0
            )
            .astype(
                int
            )
        )

    if not (
        result[
            "tumor_free_slice_count"
        ]
        == result[
            "screened_tumor_free_slice_count"
        ]
    ).all():
        raise PairSpaceAnalysisError(
            "Tumor-free counts disagree between "
            "screening and base compatibility tables."
        )

    result[
        "eligibility_rate"
    ] = (
        result[
            "eligible_base_count"
        ]
        / result[
            "tumor_free_slice_count"
        ].replace(
            0,
            np.nan,
        )
    )

    result[
        "rejected_base_count"
    ] = (
        result[
            "tumor_free_slice_count"
        ]
        - result[
            "eligible_base_count"
        ]
    )

    result = result.sort_values(
        "subject_numeric_id",
        kind="stable",
    ).reset_index(
        drop=True
    )

    return result


def build_slice_summary(
    *,
    screening: pd.DataFrame,
    bases: pd.DataFrame,
) -> pd.DataFrame:
    """Construct one row per axial slice index."""

    tumor_free = (
        screening.loc[
            screening[
                "tumor_free_candidate"
            ].astype(
                bool
            )
        ]
        .groupby(
            "slice_index",
            as_index=False,
        )
        .size()
        .rename(
            columns={
                "size":
                    "tumor_free_slice_count"
            }
        )
    )

    eligible = (
        bases.groupby(
            "slice_index",
            as_index=False,
        )
        .agg(
            screened_tumor_free_slice_count=(
                "subject_numeric_id",
                "size",
            ),

            eligible_base_count=(
                "composition_eligible",
                "sum",
            ),

            median_compatible_donors=(
                "compatible_donor_count",
                "median",
            ),

            mean_compatible_donors=(
                "compatible_donor_count",
                "mean",
            ),

            median_brain_pixels=(
                "brain_pixels",
                "median",
            ),

            mean_brain_pixels=(
                "brain_pixels",
                "mean",
            ),
        )
    )

    result = tumor_free.merge(
        eligible,
        on="slice_index",
        how="outer",
        validate="one_to_one",
    ).sort_values(
        "slice_index",
        kind="stable",
    ).reset_index(
        drop=True
    )

    result[
        "tumor_free_slice_count"
    ] = (
        result[
            "tumor_free_slice_count"
        ]
        .fillna(
            0
        )
        .astype(
            int
        )
    )

    result[
        "screened_tumor_free_slice_count"
    ] = (
        result[
            "screened_tumor_free_slice_count"
        ]
        .fillna(
            0
        )
        .astype(
            int
        )
    )

    result[
        "eligible_base_count"
    ] = (
        result[
            "eligible_base_count"
        ]
        .fillna(
            0
        )
        .astype(
            int
        )
    )

    if not (
        result[
            "tumor_free_slice_count"
        ]
        == result[
            "screened_tumor_free_slice_count"
        ]
    ).all():
        raise PairSpaceAnalysisError(
            "Slice-level tumor-free counts disagree "
            "between source tables."
        )

    result[
        "eligibility_rate"
    ] = (
        result[
            "eligible_base_count"
        ]
        / result[
            "tumor_free_slice_count"
        ].replace(
            0,
            np.nan,
        )
    )

    return result


def build_cohort_balance_summary(
    *,
    subject_summary: pd.DataFrame,
    targets: tuple[int, ...],
) -> pd.DataFrame:
    """
    Summarize feasibility of fixed per-subject base quotas.

    This is descriptive only and does not recommend or select a cohort.
    """

    rows: list[
        dict[str, Any]
    ] = []

    total_subjects = int(
        len(
            subject_summary
        )
    )

    eligible_counts = (
        subject_summary[
            "eligible_base_count"
        ].to_numpy(
            dtype=np.int64
        )
    )

    for target in targets:
        subjects_meeting_target = int(
            (
                eligible_counts
                >= target
            ).sum()
        )

        perfectly_balanced_possible = (
            subjects_meeting_target
            == total_subjects
        )

        rows.append(
            {
                "bases_per_subject_target":
                    target,

                "subjects_meeting_target":
                    subjects_meeting_target,

                "total_subjects":
                    total_subjects,

                "subject_fraction_meeting_target":
                    float(
                        subjects_meeting_target
                        / total_subjects
                    ),

                "perfectly_balanced_across_all_subjects":
                    perfectly_balanced_possible,

                "total_cases_if_all_subjects_meet_target":
                    (
                        int(
                            target
                            * total_subjects
                        )
                        if perfectly_balanced_possible
                        else np.nan
                    ),

                "total_cases_using_only_subjects_meeting_target":
                    int(
                        target
                        * subjects_meeting_target
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


def save_eligible_bases_per_subject(
    subject_summary: pd.DataFrame,
    path: Path,
    *,
    dpi: int,
) -> None:
    """Plot eligible external bases per subject."""

    figure, axis = plt.subplots(
        figsize=(
            12,
            5,
        )
    )

    axis.bar(
        subject_summary[
            "subject_numeric_id"
        ],
        subject_summary[
            "eligible_base_count"
        ],
    )

    axis.set_xlabel(
        "Validation subject numeric ID"
    )

    axis.set_ylabel(
        "Composition-eligible tumor-free slices"
    )

    axis.set_title(
        "Eligible External Base Slices per Validation Subject"
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


def save_eligible_bases_by_slice(
    slice_summary: pd.DataFrame,
    path: Path,
    *,
    dpi: int,
) -> None:
    """Plot screened and eligible bases across axial position."""

    figure, axis = plt.subplots(
        figsize=(
            11,
            5,
        )
    )

    axis.plot(
        slice_summary[
            "slice_index"
        ],
        slice_summary[
            "tumor_free_slice_count"
        ],
        label="Tumor-free screened",
    )

    axis.plot(
        slice_summary[
            "slice_index"
        ],
        slice_summary[
            "eligible_base_count"
        ],
        label="Composition-eligible",
    )

    axis.set_xlabel(
        "Axial slice index"
    )

    axis.set_ylabel(
        "Number of validation slices"
    )

    axis.set_title(
        "External Base Eligibility Across Axial Slice Position"
    )

    axis.legend()

    figure.tight_layout()

    figure.savefig(
        path,
        dpi=dpi,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )


def save_compatible_donor_histogram(
    bases: pd.DataFrame,
    path: Path,
    *,
    dpi: int,
) -> None:
    """Plot compatible-donor count distribution."""

    figure, axis = plt.subplots(
        figsize=(
            8,
            5,
        )
    )

    axis.hist(
        bases[
            "compatible_donor_count"
        ],
        bins=50,
    )

    axis.set_xlabel(
        "Compatible training donors per external base"
    )

    axis.set_ylabel(
        "External base count"
    )

    axis.set_title(
        "Distribution of Compatible Donors per External Base"
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


def save_compatible_donor_ecdf(
    bases: pd.DataFrame,
    path: Path,
    *,
    dpi: int,
) -> None:
    """Plot empirical CDF of compatible-donor counts."""

    values = np.sort(
        bases[
            "compatible_donor_count"
        ].to_numpy(
            dtype=np.int64
        )
    )

    probability = (
        np.arange(
            1,
            values.size + 1,
        )
        / values.size
    )

    figure, axis = plt.subplots(
        figsize=(
            8,
            5,
        )
    )

    axis.plot(
        values,
        probability,
    )

    axis.set_xlabel(
        "Compatible training donors per external base"
    )

    axis.set_ylabel(
        "Empirical cumulative probability"
    )

    axis.set_title(
        "ECDF of Compatible Donors per External Base"
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
    """Save one descriptive scatterplot."""

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


def save_accepted_vs_rejected_brain_pixels(
    bases: pd.DataFrame,
    path: Path,
    *,
    dpi: int,
) -> None:
    """Compare brain-pixel distributions by composition eligibility."""

    accepted = bases.loc[
        bases[
            "composition_eligible"
        ],
        "brain_pixels",
    ]

    rejected = bases.loc[
        ~bases[
            "composition_eligible"
        ],
        "brain_pixels",
    ]

    figure, axis = plt.subplots(
        figsize=(
            8,
            5,
        )
    )

    axis.hist(
        [
            accepted,
            rejected,
        ],
        bins=50,
        label=[
            "Eligible",
            "Rejected",
        ],
        alpha=0.65,
    )

    axis.set_xlabel(
        "External base brain pixels"
    )

    axis.set_ylabel(
        "Slice count"
    )

    axis.set_title(
        "Brain-Pixel Distribution by Composition Eligibility"
    )

    axis.legend()

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
    """Run descriptive external pair-space analysis."""

    args = parse_args()

    folders_config = load_folders_config(
        args.folders_file
    )

    nnunet_run_root = None

    if (
        args.screening_csv is None
        or args.base_counts_csv is None
        or args.donor_counts_csv is None
        or args.pair_space_summary is None
        or args.output_dir is None
    ):
        nnunet_run_root = resolve_config_path(
            key="nnunet_run_root",
            cli_value=None,
            config=folders_config,
            selector=None,
        ).expanduser().resolve()

    if args.screening_csv is not None:
        screening_csv = resolve_path(
            args.screening_csv,
            kind="Screening CSV",
        )
    else:
        screening_csv = resolve_path(
            nnunet_run_root
            / "validation_slice_screening"
            / "validation_slice_screening.csv",
            kind="Screening CSV",
        )

    if args.base_counts_csv is not None:
        base_counts_csv = resolve_path(
            args.base_counts_csv,
            kind="Base compatibility CSV",
        )
    else:
        base_counts_csv = resolve_path(
            nnunet_run_root
            / "external_pair_space_audit"
            / "external_base_compatibility_counts.csv",
            kind="Base compatibility CSV",
        )

    if args.donor_counts_csv is not None:
        donor_counts_csv = resolve_path(
            args.donor_counts_csv,
            kind="Donor compatibility CSV",
        )
    else:
        donor_counts_csv = resolve_path(
            nnunet_run_root
            / "external_pair_space_audit"
            / "external_donor_compatibility_counts.csv",
            kind="Donor compatibility CSV",
        )

    if args.pair_space_summary is not None:
        pair_space_summary_path = resolve_path(
            args.pair_space_summary,
            kind="Pair-space summary JSON",
        )
    else:
        pair_space_summary_path = resolve_path(
            nnunet_run_root
            / "external_pair_space_audit"
            / "external_pair_space_summary.json",
            kind="Pair-space summary JSON",
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
            / "external_pair_space_analysis"
        )

    save_folders_config(
        args.folders_file,
        folders_config,
    )

    balance_targets = tuple(
        sorted(
            set(
                int(
                    value
                )
                for value in args.balance_targets
            )
        )
    )

    if not balance_targets:
        raise PairSpaceAnalysisError(
            "--balance-targets must contain at least one value."
        )

    if any(
        value <= 0
        for value in balance_targets
    ):
        raise PairSpaceAnalysisError(
            "--balance-targets values must all be positive."
        )

    if args.dpi <= 0:
        raise PairSpaceAnalysisError(
            "--dpi must be positive."
        )

    screening = pd.read_csv(
        screening_csv
    )

    bases = pd.read_csv(
        base_counts_csv
    )

    donors = pd.read_csv(
        donor_counts_csv
    )

    with pair_space_summary_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        pair_space_summary = json.load(
            file
        )

    require_columns(
        screening,
        {
            "subject",
            "subject_numeric_id",
            "slice_index",
            "predicted_tumor_pixels",
            "tumor_free_candidate",
        },
        name="Screening CSV",
    )

    require_columns(
        bases,
        {
            "subject",
            "subject_numeric_id",
            "slice_index",
            "predicted_tumor_pixels",
            "tumor_free_candidate",
            "brain_pixels",
            "compatible_donor_count",
        },
        name="Base compatibility CSV",
    )

    require_columns(
        donors,
        {
            "slice_path",
            "volume",
            "slice",
            "whole_tumor_pixels",
            "loaded_mask_pixels",
            "compatible_external_base_count",
        },
        name="Donor compatibility CSV",
    )

    bases[
        "tumor_free_candidate"
    ] = bases[
        "tumor_free_candidate"
    ].astype(
        bool
    )

    if not bases[
        "tumor_free_candidate"
    ].all():
        raise PairSpaceAnalysisError(
            "Base compatibility table unexpectedly contains "
            "non-tumor-free rows."
        )

    if not (
        bases[
            "predicted_tumor_pixels"
        ]
        == 0
    ).all():
        raise PairSpaceAnalysisError(
            "Base compatibility table contains a row with "
            "nonzero predicted_tumor_pixels."
        )

    if (
        bases[
            "compatible_donor_count"
        ]
        < 0
    ).any():
        raise PairSpaceAnalysisError(
            "compatible_donor_count contains a negative value."
        )

    bases[
        "composition_eligible"
    ] = (
        bases[
            "compatible_donor_count"
        ]
        > 0
    )

    external_base_count = int(
        len(
            bases
        )
    )

    eligible_base_count = int(
        bases[
            "composition_eligible"
        ].sum()
    )

    rejected_base_count = int(
        (
            ~bases[
                "composition_eligible"
            ]
        ).sum()
    )

    expected_external_bases = int(
        pair_space_summary[
            "external_tumor_free_bases"
        ]
    )

    expected_eligible_bases = int(
        pair_space_summary[
            "external_bases_with_at_least_one_compatible_donor"
        ]
    )

    expected_rejected_bases = int(
        pair_space_summary[
            "external_bases_with_zero_compatible_donors"
        ]
    )

    if (
        external_base_count
        != expected_external_bases
    ):
        raise PairSpaceAnalysisError(
            "External base count disagrees with pair-space summary.\n"
            f"Observed: {external_base_count}\n"
            f"Expected: {expected_external_bases}"
        )

    if (
        eligible_base_count
        != expected_eligible_bases
    ):
        raise PairSpaceAnalysisError(
            "Eligible base count disagrees with pair-space summary.\n"
            f"Observed: {eligible_base_count}\n"
            f"Expected: {expected_eligible_bases}"
        )

    if (
        rejected_base_count
        != expected_rejected_bases
    ):
        raise PairSpaceAnalysisError(
            "Rejected base count disagrees with pair-space summary.\n"
            f"Observed: {rejected_base_count}\n"
            f"Expected: {expected_rejected_bases}"
        )

    total_pairs_from_bases = int(
        bases[
            "compatible_donor_count"
        ].sum()
    )

    total_pairs_from_donors = int(
        donors[
            "compatible_external_base_count"
        ].sum()
    )

    expected_total_pairs = int(
        pair_space_summary[
            "total_compatible_base_donor_pairs"
        ]
    )

    if not (
        total_pairs_from_bases
        == total_pairs_from_donors
        == expected_total_pairs
    ):
        raise PairSpaceAnalysisError(
            "Compatible-pair totals disagree across audit artifacts.\n"
            f"Base table  : {total_pairs_from_bases}\n"
            f"Donor table : {total_pairs_from_donors}\n"
            f"Summary     : {expected_total_pairs}"
        )

    print()
    print(
        "=" * 78
    )

    print(
        "EXTERNAL PAIR-SPACE DESCRIPTIVE ANALYSIS"
    )

    print(
        "=" * 78
    )

    print(
        f"Screened tumor-free bases : "
        f"{external_base_count:,}"
    )

    print(
        f"Composition-eligible bases: "
        f"{eligible_base_count:,}"
    )

    print(
        f"Rejected bases            : "
        f"{rejected_base_count:,}"
    )

    print(
        f"Compatible pairs          : "
        f"{expected_total_pairs:,}"
    )

    print(
        f"Training donors           : "
        f"{len(donors):,}"
    )

    print(
        "=" * 78
    )

    subject_summary = build_subject_summary(
        screening=screening,
        bases=bases,
    )

    slice_summary = build_slice_summary(
        screening=screening,
        bases=bases,
    )

    rejected_bases = (
        bases.loc[
            ~bases[
                "composition_eligible"
            ]
        ]
        .copy()
        .sort_values(
            [
                "subject_numeric_id",
                "slice_index",
            ],
            kind="stable",
        )
        .reset_index(
            drop=True
        )
    )

    cohort_balance_summary = (
        build_cohort_balance_summary(
            subject_summary=subject_summary,
            targets=balance_targets,
        )
    )

    subjects_total = int(
        len(
            subject_summary
        )
    )

    subjects_with_eligible = int(
        (
            subject_summary[
                "eligible_base_count"
            ]
            > 0
        ).sum()
    )

    subjects_without_eligible = (
        subjects_total
        - subjects_with_eligible
    )

    minimum_eligible_per_subject = int(
        subject_summary[
            "eligible_base_count"
        ].min()
    )

    maximum_uniform_quota = (
        minimum_eligible_per_subject
    )

    maximum_uniform_case_count = int(
        maximum_uniform_quota
        * subjects_total
    )

    print()
    print(
        "Subject coverage"
    )

    print(
        "-" * 78
    )

    print(
        f"Validation subjects       : "
        f"{subjects_total:,}"
    )

    print(
        f"Subjects with >=1 eligible: "
        f"{subjects_with_eligible:,}"
    )

    print(
        f"Subjects with zero eligible: "
        f"{subjects_without_eligible:,}"
    )

    print(
        f"Minimum eligible/subject : "
        f"{minimum_eligible_per_subject:,}"
    )

    print(
        f"Maximum uniform quota    : "
        f"{maximum_uniform_quota:,} bases/subject"
    )

    print(
        f"Uniform cohort at max    : "
        f"{maximum_uniform_case_count:,} cases"
    )

    print(
        "-" * 78
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

    subject_summary_path = (
        output_dir
        / SUBJECT_SUMMARY_NAME
    )

    slice_summary_path = (
        output_dir
        / SLICE_SUMMARY_NAME
    )

    rejected_summary_path = (
        output_dir
        / REJECTED_BASE_SUMMARY_NAME
    )

    cohort_balance_path = (
        output_dir
        / COHORT_BALANCE_SUMMARY_NAME
    )

    analysis_summary_path = (
        output_dir
        / ANALYSIS_SUMMARY_NAME
    )

    subject_summary.to_csv(
        subject_summary_path,
        index=False,
    )

    slice_summary.to_csv(
        slice_summary_path,
        index=False,
    )

    rejected_bases.to_csv(
        rejected_summary_path,
        index=False,
    )

    cohort_balance_summary.to_csv(
        cohort_balance_path,
        index=False,
    )

    save_eligible_bases_per_subject(
        subject_summary,
        figure_dir
        / "eligible_bases_per_subject.png",
        dpi=args.dpi,
    )

    save_eligible_bases_by_slice(
        slice_summary,
        figure_dir
        / "eligible_bases_by_slice.png",
        dpi=args.dpi,
    )

    save_compatible_donor_histogram(
        bases,
        figure_dir
        / "compatible_donor_histogram.png",
        dpi=args.dpi,
    )

    save_compatible_donor_ecdf(
        bases,
        figure_dir
        / "compatible_donor_ecdf.png",
        dpi=args.dpi,
    )

    save_scatter(
        x=bases[
            "brain_pixels"
        ],
        y=bases[
            "compatible_donor_count"
        ],
        xlabel="External base brain pixels",
        ylabel="Compatible training donors",
        title=(
            "Brain Content and Donor Compatibility"
        ),
        path=(
            figure_dir
            / "brain_pixels_vs_donors.png"
        ),
        dpi=args.dpi,
    )

    save_scatter(
        x=bases[
            "slice_index"
        ],
        y=bases[
            "brain_pixels"
        ],
        xlabel="Axial slice index",
        ylabel="External base brain pixels",
        title=(
            "Brain Content Across Axial Slice Position"
        ),
        path=(
            figure_dir
            / "slice_index_vs_brain_pixels.png"
        ),
        dpi=args.dpi,
    )

    save_scatter(
        x=bases[
            "slice_index"
        ],
        y=bases[
            "compatible_donor_count"
        ],
        xlabel="Axial slice index",
        ylabel="Compatible training donors",
        title=(
            "Donor Compatibility Across Axial Slice Position"
        ),
        path=(
            figure_dir
            / "slice_index_vs_compatible_donors.png"
        ),
        dpi=args.dpi,
    )

    save_accepted_vs_rejected_brain_pixels(
        bases,
        figure_dir
        / "accepted_vs_rejected_brain_pixels.png",
        dpi=args.dpi,
    )

    eligible_bases = bases.loc[
        bases[
            "composition_eligible"
        ]
    ]

    rejected_subject_count = int(
        rejected_bases[
            "subject_numeric_id"
        ].nunique()
    )

    analysis_summary = {
        "analysis_role":
            (
                "Descriptive audit of the completed "
                "external base-donor compatibility space. "
                "No cohort selection or donor assignment "
                "is performed."
            ),

        "population": {
            "validation_subjects":
                subjects_total,

            "screened_tumor_free_bases":
                external_base_count,

            "composition_eligible_bases":
                eligible_base_count,

            "composition_rejected_bases":
                rejected_base_count,

            "composition_eligibility_fraction":
                float(
                    eligible_base_count
                    / external_base_count
                ),

            "training_donors_after_margin":
                int(
                    len(
                        donors
                    )
                ),

            "compatible_base_donor_pairs":
                expected_total_pairs,
        },

        "subject_coverage": {
            "subjects_with_at_least_one_eligible_base":
                subjects_with_eligible,

            "subjects_with_zero_eligible_bases":
                subjects_without_eligible,

            "subjects_with_at_least_one_rejected_base":
                rejected_subject_count,

            "eligible_bases_per_subject":
                describe_numeric(
                    subject_summary[
                        "eligible_base_count"
                    ]
                ),

            "tumor_free_bases_per_subject":
                describe_numeric(
                    subject_summary[
                        "tumor_free_slice_count"
                    ]
                ),

            "eligibility_rate_per_subject":
                describe_numeric(
                    subject_summary[
                        "eligibility_rate"
                    ]
                ),
        },

        "base_characteristics": {
            "compatible_donors_all_screened_bases":
                describe_numeric(
                    bases[
                        "compatible_donor_count"
                    ]
                ),

            "compatible_donors_eligible_bases_only":
                describe_numeric(
                    eligible_bases[
                        "compatible_donor_count"
                    ]
                ),

            "brain_pixels_all_screened_bases":
                describe_numeric(
                    bases[
                        "brain_pixels"
                    ]
                ),

            "brain_pixels_eligible_bases":
                describe_numeric(
                    eligible_bases[
                        "brain_pixels"
                    ]
                ),

            "brain_pixels_rejected_bases":
                describe_numeric(
                    rejected_bases[
                        "brain_pixels"
                    ]
                ),

            "slice_index_eligible_bases":
                describe_numeric(
                    eligible_bases[
                        "slice_index"
                    ]
                ),

            "slice_index_rejected_bases":
                describe_numeric(
                    rejected_bases[
                        "slice_index"
                    ]
                ),
        },

        "cohort_feasibility": {
            "note":
                (
                    "These quantities describe possible "
                    "per-subject quotas only. They do not "
                    "constitute a cohort recommendation."
                ),

            "maximum_uniform_bases_per_subject":
                maximum_uniform_quota,

            "maximum_uniform_case_count_across_all_subjects":
                maximum_uniform_case_count,

            "evaluated_balance_targets":
                list(
                    balance_targets
                ),

            "target_results":
                (
                    cohort_balance_summary
                    .replace(
                        {
                            np.nan:
                                None
                        }
                    )
                    .to_dict(
                        orient="records"
                    )
                ),
        },

        "selection_status": {
            "definitive_external_manifest_created":
                False,

            "external_cases_selected":
                False,

            "donors_assigned":
                False,

            "recommendation":
                (
                    "No manifest recommendation is made by "
                    "this descriptive analysis. Review the "
                    "subject, slice-position, brain-content, "
                    "and compatibility distributions before "
                    "fixing the evaluation design."
                ),
        },

        "source_artifacts": {
            "screening_csv":
                str(
                    screening_csv
                ),

            "base_counts_csv":
                str(
                    base_counts_csv
                ),

            "donor_counts_csv":
                str(
                    donor_counts_csv
                ),

            "pair_space_summary":
                str(
                    pair_space_summary_path
                ),
        },

        "output_artifacts": {
            "subject_summary":
                str(
                    subject_summary_path
                ),

            "slice_summary":
                str(
                    slice_summary_path
                ),

            "rejected_base_summary":
                str(
                    rejected_summary_path
                ),

            "cohort_balance_summary":
                str(
                    cohort_balance_path
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

            "screening_csv_sha256":
                sha256_file(
                    screening_csv
                ),

            "base_counts_csv_sha256":
                sha256_file(
                    base_counts_csv
                ),

            "donor_counts_csv_sha256":
                sha256_file(
                    donor_counts_csv
                ),

            "pair_space_summary_sha256":
                sha256_file(
                    pair_space_summary_path
                ),

            "analyzed_at_utc":
                datetime.now(
                    timezone.utc
                ).isoformat(),
        },
    }

    with analysis_summary_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            analysis_summary,
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
        "EXTERNAL PAIR-SPACE DESCRIPTIVE ANALYSIS: PASS"
    )

    print(
        "=" * 78
    )

    print(
        f"Subject summary          : "
        f"{subject_summary_path}"
    )

    print(
        f"Slice summary            : "
        f"{slice_summary_path}"
    )

    print(
        f"Rejected-base summary    : "
        f"{rejected_summary_path}"
    )

    print(
        f"Cohort balance summary   : "
        f"{cohort_balance_path}"
    )

    print(
        f"Analysis summary         : "
        f"{analysis_summary_path}"
    )

    print(
        f"Figures                  : "
        f"{figure_dir}"
    )

    print()
    print(
        "No external cohort was selected."
    )

    print(
        "No donor assignments were made."
    )

    print(
        "=" * 78
    )


if __name__ == "__main__":
    main()
