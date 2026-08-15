#!/usr/bin/env python3
"""
Audit candidate external BR-LoRA evaluation-manifest designs.

This script constructs three nested candidate external evaluation designs:

    125 cases = 1 eligible base per validation subject
    250 cases = 2 eligible bases per validation subject
    625 cases = 5 eligible bases per validation subject

The purpose is to compare candidate case counts before freezing a definitive
external evaluation manifest.

Base sampling
-------------
Eligible external bases are exactly those already established by the completed
screening and pair-space audits:

    tumor_free_candidate == True
    predicted_tumor_pixels == 0
    compatible_donor_count > 0

Within each of the 125 validation subjects, five eligible base slices are
sampled uniformly without replacement and given ranks 1-5.

The designs are nested:

    candidate_125: rank <= 1
    candidate_250: rank <= 2
    candidate_625: rank <= 5

Donor assignment
----------------
For each selected external base, exact base-donor compatibility is reconstructed
using the already established composition rules:

    donor whole-tumor pixels >= 300
    donor mask-margin rule already passed
    external brain threshold = 0.05
    mask-inside-brain fraction >= 0.80

Donor assignment is solved globally as a one-to-one bipartite matching
problem over the complete compatibility graph for the 625 selected external
bases.

Every selected external base must receive exactly one compatible donor slice,
and each donor slice may be used at most once. Bases are processed from the
smallest to the largest compatibility set, and augmenting paths are used to
reroute earlier assignments when necessary.

Training-volume diversity is retained only as a donor traversal preference.
Within each base's complete compatible-donor set, compatible training volumes
and donor slices are reproducibly shuffled and donor candidates are interleaved
across volumes. This ordering does not remove compatibility edges, impose a
training-volume quota, or change matching feasibility.

The 125- and 250-case candidates are nested subsets of the 625 selected-base
superset. Their donor assignments are inherited from the completed global
625-case matching.

Important
---------
This is a design audit only.

It does NOT:

- choose a definitive case count,
- freeze the external evaluation manifest,
- modify screening rules,
- introduce morphology strata,
- run BR-LoRA inference, or
- use validation ground-truth tumor labels.

Primary outputs
---------------
candidate_manifest_125.csv
candidate_manifest_250.csv
candidate_manifest_625.csv
candidate_assignments_all.csv
candidate_design_summary.csv
candidate_distribution_comparison.csv
candidate_manifest_design_summary.json
figures/
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
from datetime import datetime, timezone
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

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.data import (
    get_brain_mask,
    load_validation_dataset_specification,
    load_validation_slice,
)


IMAGE_HEIGHT = 240
IMAGE_WIDTH = 240
IMAGE_AREA = (
    IMAGE_HEIGHT
    * IMAGE_WIDTH
)

BASE_RANKS = (
    1,
    2,
    3,
    4,
    5,
)

CANDIDATE_QUOTAS = {
    "candidate_125": 1,
    "candidate_250": 2,
    "candidate_625": 5,
}

CANDIDATE_CASE_COUNTS = {
    "candidate_125": 125,
    "candidate_250": 250,
    "candidate_625": 625,
}

REQUIRED_BASE_COLUMNS = {
    "subject",
    "subject_numeric_id",
    "slice_index",
    "predicted_tumor_pixels",
    "tumor_free_candidate",
    "reference_modality",
    "reference_image_path",
    "brain_pixels",
    "compatible_donor_count",
}

REQUIRED_DONOR_COLUMNS = {
    "slice_path",
    "volume",
    "slice",
    "whole_tumor_pixels",
    "donor_h5_path",
    "mask_has_margin",
    "loaded_mask_pixels",
    "compatible_external_base_count",
    "tumor_area_pixels",
    "tumor_area_fraction",
    "bbox_area",
    "bbox_fill_fraction",
    "centroid_x_normalized",
    "centroid_y_normalized",
    "centroid_laterality",
    "connected_component_count",
    "largest_component_fraction",
}

EVALUATOR_MANIFEST_COLUMNS = [
    "case_id",
    "external_subject_numeric_id",
    "external_slice_index",
    "external_modality",
    "donor_h5_path",
]

OUTPUT_ASSIGNMENTS_NAME = (
    "candidate_assignments_all.csv"
)

OUTPUT_DESIGN_SUMMARY_NAME = (
    "candidate_design_summary.csv"
)

OUTPUT_DISTRIBUTION_NAME = (
    "candidate_distribution_comparison.csv"
)

OUTPUT_JSON_NAME = (
    "candidate_manifest_design_summary.json"
)

FIGURE_DIR_NAME = "figures"

FIGURE_NAMES = (
    "external_slice_index_ecdf.png",
    "external_brain_pixels_ecdf.png",
    "base_compatible_donors_ecdf.png",
    "donor_tumor_area_ecdf.png",
    "donor_component_count_ecdf.png",
    "donor_compatibility_ecdf.png",
)


class ManifestDesignAuditError(
    RuntimeError
):
    """Raised when candidate-manifest design auditing fails."""


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Construct and compare nested candidate external "
            "BR-LoRA evaluation manifests."
        )
    )

    parser.add_argument(
        "--validation-dataset",
        required=True,
        type=Path,
        help=(
            "Registered BraTS 2020 validation_dataset.yaml."
        ),
    )

    parser.add_argument(
        "--base-counts-csv",
        required=True,
        type=Path,
        help=(
            "external_base_compatibility_counts.csv from the "
            "completed pair-space audit."
        ),
    )

    parser.add_argument(
        "--donor-morphology-csv",
        required=True,
        type=Path,
        help=(
            "donor_morphology.csv from the completed donor "
            "morphology audit."
        ),
    )

    parser.add_argument(
        "--pair-space-summary",
        required=True,
        type=Path,
        help=(
            "external_pair_space_summary.json from the completed "
            "pair-space audit."
        ),
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help=(
            "Directory for candidate manifests and design-audit outputs."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help=(
            "Base sampling seed. Default: 42, matching the validated "
            "baseline configuration seed."
        ),
    )

    parser.add_argument(
        "--brain-threshold",
        type=float,
        default=0.05,
        help=(
            "External brain-mask threshold. Default: 0.05."
        ),
    )

    parser.add_argument(
        "--min-overlap",
        type=float,
        default=0.80,
        help=(
            "Minimum donor-mask fraction required inside the external "
            "brain mask. Default: 0.80."
        ),
    )

    parser.add_argument(
        "--external-modality",
        default="flair",
        help=(
            "External validation MRI modality. Default: flair."
        ),
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
        help=(
            "Device for compatibility matrix calculations."
        ),
    )

    parser.add_argument(
        "--base-batch-size",
        type=int,
        default=32,
        help=(
            "Selected-base batch size for compatibility computation. "
            "Default: 32."
        ),
    )

    parser.add_argument(
        "--donor-batch-size",
        type=int,
        default=256,
        help=(
            "Donor-mask batch size for compatibility computation. "
            "Default: 256."
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


def resolve_existing_file(
    path: Path,
    *,
    name: str,
) -> Path:
    """Resolve and require one existing file."""

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
    """Return SHA-256 of one file."""

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
    """Return whether current Git worktree is clean."""

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
    """Require expected table columns."""

    missing = sorted(
        required
        - set(
            table.columns
        )
    )

    if missing:
        raise ManifestDesignAuditError(
            f"{name} is missing required column(s): "
            + ", ".join(
                missing
            )
        )


def resolve_device(
    requested: str,
) -> torch.device:
    """Resolve requested Torch device."""

    if requested == "cpu":
        return torch.device(
            "cpu"
        )

    if requested == "cuda":
        if not torch.cuda.is_available():
            raise ManifestDesignAuditError(
                "CUDA was requested but is unavailable."
            )

        return torch.device(
            "cuda"
        )

    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise ManifestDesignAuditError(
                "MPS was requested but is unavailable."
            )

        return torch.device(
            "mps"
        )

    if torch.cuda.is_available():
        return torch.device(
            "cuda"
        )

    if torch.backends.mps.is_available():
        return torch.device(
            "mps"
        )

    return torch.device(
        "cpu"
    )


def set_seed(
    seed: int,
) -> None:
    """Set process-level random seeds."""

    random.seed(
        seed
    )

    np.random.seed(
        seed
    )

    torch.manual_seed(
        seed
    )


def refuse_existing_outputs(
    output_dir: Path,
) -> None:
    """Refuse to overwrite design-audit outputs."""

    expected = [
        output_dir
        / OUTPUT_ASSIGNMENTS_NAME,

        output_dir
        / OUTPUT_DESIGN_SUMMARY_NAME,

        output_dir
        / OUTPUT_DISTRIBUTION_NAME,

        output_dir
        / OUTPUT_JSON_NAME,
    ]

    for candidate_name in (
        CANDIDATE_QUOTAS
    ):
        expected.append(
            output_dir
            / (
                f"{candidate_name}"
                "_manifest.csv"
            )
        )

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
        raise ManifestDesignAuditError(
            "Refusing to overwrite existing design-audit output(s):\n"
            + "\n".join(
                str(
                    path
                )
                for path in existing
            )
        )


def validate_source_tables(
    *,
    bases: pd.DataFrame,
    donors: pd.DataFrame,
    pair_summary: dict[str, Any],
) -> pd.DataFrame:
    """Validate base and donor source-table contracts."""

    require_columns(
        bases,
        REQUIRED_BASE_COLUMNS,
        name="Base compatibility CSV",
    )

    require_columns(
        donors,
        REQUIRED_DONOR_COLUMNS,
        name="Donor morphology CSV",
    )

    expected_base_count = int(
        pair_summary[
            "external_tumor_free_bases"
        ]
    )

    expected_donor_count = int(
        pair_summary[
            "training_donors_after_margin"
        ]
    )

    if len(
        bases
    ) != expected_base_count:
        raise ManifestDesignAuditError(
            "Base-count table row count disagrees "
            "with pair-space summary.\n"
            f"Observed: {len(bases):,}\n"
            f"Expected: {expected_base_count:,}"
        )

    if len(
        donors
    ) != expected_donor_count:
        raise ManifestDesignAuditError(
            "Donor table row count disagrees "
            "with pair-space summary.\n"
            f"Observed: {len(donors):,}\n"
            f"Expected: {expected_donor_count:,}"
        )

    if not bases[
        "tumor_free_candidate"
    ].astype(
        bool
    ).all():
        raise ManifestDesignAuditError(
            "Base table contains a row that is not "
            "a tumor-free screening candidate."
        )

    if (
        bases[
            "predicted_tumor_pixels"
        ]
        != 0
    ).any():
        raise ManifestDesignAuditError(
            "Base table contains a nonzero predicted "
            "tumor pixel count."
        )

    if not donors[
        "mask_has_margin"
    ].astype(
        bool
    ).all():
        raise ManifestDesignAuditError(
            "Donor morphology table contains a donor "
            "that failed the established margin rule."
        )

    if (
        donors[
            "whole_tumor_pixels"
        ]
        < 300
    ).any():
        raise ManifestDesignAuditError(
            "Donor morphology table contains a donor "
            "below the established 300-pixel threshold."
        )

    if not (
        donors[
            "whole_tumor_pixels"
        ]
        == donors[
            "loaded_mask_pixels"
        ]
    ).all():
        raise ManifestDesignAuditError(
            "Donor manifest and loaded whole-tumor "
            "pixel counts disagree."
        )

    if (
        donors[
            "compatible_external_base_count"
        ]
        <= 0
    ).any():
        raise ManifestDesignAuditError(
            "Donor table contains a donor with no "
            "compatible external base."
        )

    eligible = bases.loc[
        bases[
            "compatible_donor_count"
        ]
        > 0
    ].copy()

    expected_eligible = int(
        pair_summary[
            "external_bases_with_at_least_one_compatible_donor"
        ]
    )

    if len(
        eligible
    ) != expected_eligible:
        raise ManifestDesignAuditError(
            "Composition-eligible base count disagrees "
            "with pair-space summary.\n"
            f"Observed: {len(eligible):,}\n"
            f"Expected: {expected_eligible:,}"
        )

    subject_count = int(
        eligible[
            "subject_numeric_id"
        ].nunique()
    )

    if subject_count != 125:
        raise ManifestDesignAuditError(
            "Expected eligible bases from all 125 validation "
            f"subjects; observed {subject_count}."
        )

    per_subject = (
        eligible.groupby(
            "subject_numeric_id"
        )
        .size()
    )

    if int(
        per_subject.min()
    ) < 5:
        raise ManifestDesignAuditError(
            "At least one validation subject has fewer than "
            "five composition-eligible base slices."
        )

    return (
        eligible.sort_values(
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


def sample_nested_bases(
    eligible_bases: pd.DataFrame,
    *,
    seed: int,
) -> pd.DataFrame:
    """
    Sample five ordered eligible bases per validation subject.

    The same sampled rows support the 125-, 250-, and 625-case candidates.
    """

    rng = np.random.default_rng(
        seed
    )

    selected_rows: list[
        dict[str, Any]
    ] = []

    subject_ids = sorted(
        int(
            value
        )
        for value in (
            eligible_bases[
                "subject_numeric_id"
            ].unique()
        )
    )

    for subject_id in subject_ids:
        subject_table = (
            eligible_bases.loc[
                eligible_bases[
                    "subject_numeric_id"
                ]
                == subject_id
            ]
            .sort_values(
                "slice_index",
                kind="stable",
            )
            .reset_index(
                drop=True
            )
        )

        selected_positions = rng.choice(
            len(
                subject_table
            ),
            size=5,
            replace=False,
        )

        for rank, position in zip(
            BASE_RANKS,
            selected_positions.tolist(),
            strict=True,
        ):
            row = subject_table.iloc[
                int(
                    position
                )
            ].to_dict()

            row[
                "base_rank"
            ] = rank

            row[
                "case_id"
            ] = (
                f"external_s"
                f"{subject_id:03d}"
                f"_r{rank:02d}"
            )

            selected_rows.append(
                row
            )

    selected = pd.DataFrame(
        selected_rows
    )

    selected = (
        selected.sort_values(
            [
                "base_rank",
                "subject_numeric_id",
            ],
            kind="stable",
        )
        .reset_index(
            drop=True
        )
    )

    if len(
        selected
    ) != 625:
        raise ManifestDesignAuditError(
            "Nested base sampling did not produce "
            "exactly 625 rows."
        )

    if selected[
        [
            "subject_numeric_id",
            "slice_index",
        ]
    ].duplicated().any():
        raise ManifestDesignAuditError(
            "Nested base selection contains a duplicated "
            "external subject/slice pair."
        )

    for rank in BASE_RANKS:
        observed = int(
            (
                selected[
                    "base_rank"
                ]
                == rank
            ).sum()
        )

        if observed != 125:
            raise ManifestDesignAuditError(
                f"Base rank {rank} does not contain "
                "exactly 125 subjects."
            )

    return selected


def load_selected_base_brain_masks(
    *,
    selected_bases: pd.DataFrame,
    validation_dataset: Any,
    external_modality: str,
    brain_threshold: float,
) -> np.ndarray:
    """Load selected external bases and return flattened binary brain masks."""

    brain_masks = np.zeros(
        (
            len(
                selected_bases
            ),
            IMAGE_AREA,
        ),
        dtype=np.uint8,
    )

    print()
    print(
        "Loading selected external base images..."
    )

    for position, row in selected_bases.iterrows():
        external = load_validation_slice(
            validation_dataset,
            subject_numeric_id=int(
                row[
                    "subject_numeric_id"
                ]
            ),
            slice_index=int(
                row[
                    "slice_index"
                ]
            ),
            modality=external_modality,
        )

        if tuple(
            external.image.shape
        ) != (
            1,
            IMAGE_HEIGHT,
            IMAGE_WIDTH,
        ):
            raise ManifestDesignAuditError(
                "Selected validation slice has an "
                "unexpected tensor shape."
            )

        brain = get_brain_mask(
            external.image,
            threshold=brain_threshold,
        )

        brain_numpy = (
            brain[
                0
            ]
            .detach()
            .cpu()
            .numpy()
            .astype(
                np.uint8,
                copy=False,
            )
        )

        observed_brain_pixels = int(
            brain_numpy.sum()
        )

        expected_brain_pixels = int(
            row[
                "brain_pixels"
            ]
        )

        if (
            observed_brain_pixels
            != expected_brain_pixels
        ):
            raise ManifestDesignAuditError(
                "Reconstructed external brain-pixel count "
                "disagrees with pair-space audit.\n"
                f"Case: {row['case_id']}\n"
                f"Observed: {observed_brain_pixels}\n"
                f"Expected: {expected_brain_pixels}"
            )

        brain_masks[
            position,
            :,
        ] = brain_numpy.reshape(
            -1
        )

        completed = (
            position
            + 1
        )

        if (
            completed % 100 == 0
            or completed
            == len(
                selected_bases
            )
        ):
            print(
                f"  External bases loaded: "
                f"{completed:,} / "
                f"{len(selected_bases):,}",
                flush=True,
            )

    return brain_masks


def load_and_pack_donor_masks(
    donors: pd.DataFrame,
) -> np.ndarray:
    """
    Load all donor masks once and store them using np.packbits.

    Packing keeps the cached donor-mask population compact while allowing
    donor batches to be reconstructed exactly during compatibility analysis.
    """

    packed_width = (
        IMAGE_AREA
        + 7
    ) // 8

    packed_masks = np.empty(
        (
            len(
                donors
            ),
            packed_width,
        ),
        dtype=np.uint8,
    )

    print()
    print(
        "Loading and packing donor whole-tumor masks..."
    )

    for position, row in donors.iterrows():
        path = (
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

        if not path.is_file():
            raise FileNotFoundError(
                "Donor H5 file is unavailable:\n"
                f"{path}"
            )

        with h5py.File(
            path,
            "r",
        ) as file:
            if "mask" not in file:
                raise ManifestDesignAuditError(
                    "Donor H5 file is missing dataset 'mask'.\n"
                    f"{path}"
                )

            raw_mask = file[
                "mask"
            ][:]

        if raw_mask.shape != (
            IMAGE_HEIGHT,
            IMAGE_WIDTH,
            3,
        ):
            raise ManifestDesignAuditError(
                "Unexpected donor H5 mask shape.\n"
                f"File: {path}\n"
                f"Observed: {raw_mask.shape}"
            )

        whole_tumor = (
            raw_mask.max(
                axis=-1
            )
            > 0
        ).astype(
            np.uint8
        )

        observed_pixels = int(
            whole_tumor.sum()
        )

        expected_pixels = int(
            row[
                "tumor_area_pixels"
            ]
        )

        if (
            observed_pixels
            != expected_pixels
        ):
            raise ManifestDesignAuditError(
                "Loaded donor mask area disagrees with "
                "donor morphology audit.\n"
                f"Donor: {row['slice_path']}\n"
                f"Observed: {observed_pixels}\n"
                f"Expected: {expected_pixels}"
            )

        packed_masks[
            position,
            :,
        ] = np.packbits(
            whole_tumor.reshape(
                -1
            )
        )

        completed = (
            position
            + 1
        )

        if (
            completed % 1000 == 0
            or completed
            == len(
                donors
            )
        ):
            print(
                f"  Donors packed: "
                f"{completed:,} / "
                f"{len(donors):,}",
                flush=True,
            )

    return packed_masks


def unpack_donor_batch(
    packed_masks: np.ndarray,
    *,
    start: int,
    end: int,
) -> np.ndarray:
    """Unpack one donor-mask batch to flattened binary masks."""

    unpacked = np.unpackbits(
        packed_masks[
            start:end
        ],
        axis=1,
        count=IMAGE_AREA,
    )

    return unpacked.astype(
        np.float32,
        copy=False,
    )


def compute_compatible_donor_indices_for_batch(
    *,
    base_masks: np.ndarray,
    packed_donor_masks: np.ndarray,
    donor_areas: np.ndarray,
    device: torch.device,
    min_overlap: float,
    donor_batch_size: int,
) -> list[np.ndarray]:
    """
    Compute exact compatible donor indices for one selected-base batch.
    """

    base_tensor = torch.from_numpy(
        base_masks.astype(
            np.float32,
            copy=False,
        )
    ).to(
        device=device,
    )

    compatible_chunks: list[
        list[np.ndarray]
    ] = [
        []
        for _ in range(
            base_masks.shape[
                0
            ]
        )
    ]

    donor_count = int(
        packed_donor_masks.shape[
            0
        ]
    )

    for donor_start in range(
        0,
        donor_count,
        donor_batch_size,
    ):
        donor_end = min(
            donor_start
            + donor_batch_size,
            donor_count,
        )

        donor_numpy = unpack_donor_batch(
            packed_donor_masks,
            start=donor_start,
            end=donor_end,
        )

        donor_tensor = torch.from_numpy(
            donor_numpy
        ).to(
            device=device,
        )

        overlap = (
            base_tensor
            @ donor_tensor.T
        )

        area_tensor = torch.from_numpy(
            donor_areas[
                donor_start:donor_end
            ].astype(
                np.float32,
                copy=False,
            )
        ).to(
            device=device,
        )

        fractions = (
            overlap
            / area_tensor.unsqueeze(
                0
            )
        )

        compatibility = (
            fractions
            >= min_overlap
        ).detach().cpu().numpy()

        for base_local_index in range(
            compatibility.shape[
                0
            ]
        ):
            local_indices = np.flatnonzero(
                compatibility[
                    base_local_index
                ]
            )

            if local_indices.size:
                compatible_chunks[
                    base_local_index
                ].append(
                    local_indices.astype(
                        np.int64,
                        copy=False,
                    )
                    + donor_start
                )

        del (
            donor_tensor,
            area_tensor,
            overlap,
            fractions,
        )

    result: list[
        np.ndarray
    ] = []

    for chunks in compatible_chunks:
        if chunks:
            indices = np.concatenate(
                chunks
            )
        else:
            indices = np.empty(
                0,
                dtype=np.int64,
            )

        result.append(
            indices
        )

    return result


def build_matching_candidate_order(
    *,
    compatible_indices: np.ndarray,
    donors: pd.DataFrame,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Build a deterministic randomized donor preference order for one base.

    Compatibility determines feasibility.

    Training-volume diversity is used only as an ordering preference:
    compatible training volumes are shuffled, donor slices are shuffled
    within volume, and donors are then interleaved across volumes.

    This ordering does not remove any compatible donor and therefore cannot
    change the feasible bipartite graph.
    """

    compatible_indices = np.asarray(
        compatible_indices,
        dtype=np.int64,
    )

    if compatible_indices.ndim != 1:
        raise ManifestDesignAuditError(
            "compatible_indices must be one-dimensional."
        )

    if compatible_indices.size == 0:
        return compatible_indices

    volumes = donors.iloc[
        compatible_indices
    ][
        "volume"
    ].to_numpy(
        dtype=np.int64
    )

    unique_volumes = np.unique(
        volumes
    )

    unique_volumes = rng.permutation(
        unique_volumes
    )

    by_volume: list[
        np.ndarray
    ] = []

    for volume in unique_volumes:
        indices = compatible_indices[
            volumes
            == volume
        ]

        indices = rng.permutation(
            indices
        ).astype(
            np.int64,
            copy=False,
        )

        by_volume.append(
            indices
        )

    # Interleave donor slices across compatible training volumes.
    # This preserves the historical "volume first" preference without
    # imposing a hard volume-level constraint on the matching problem.
    ordered: list[int] = []

    depth = 0

    while True:
        added = False

        for indices in by_volume:
            if depth < indices.size:
                ordered.append(
                    int(
                        indices[
                            depth
                        ]
                    )
                )

                added = True

        if not added:
            break

        depth += 1

    result = np.asarray(
        ordered,
        dtype=np.int64,
    )

    if result.size != compatible_indices.size:
        raise ManifestDesignAuditError(
            "Donor preference ordering changed the "
            "compatible-donor set size."
        )

    if np.unique(
        result
    ).size != result.size:
        raise ManifestDesignAuditError(
            "Donor preference ordering introduced duplicate indices."
        )

    if not np.array_equal(
        np.sort(
            result
        ),
        np.sort(
            compatible_indices
        ),
    ):
        raise ManifestDesignAuditError(
            "Donor preference ordering changed the "
            "compatible-donor set."
        )

    return result


def solve_unique_donor_matching(
    *,
    compatibility_lists: list[np.ndarray],
    donors: pd.DataFrame,
    seed: int,
) -> np.ndarray:
    """
    Solve the global one-to-one external-base / donor matching problem.

    Every selected external base must receive exactly one compatible donor
    slice, and each donor slice may be used at most once.

    A standard augmenting-path bipartite matching algorithm is used. Bases
    are processed from smallest to largest compatibility set so that the
    most constrained cases are protected first.

    Donor-volume diversity affects candidate traversal order only. It is not
    a feasibility constraint and does not alter the compatibility graph.

    Returns
    -------
    numpy.ndarray
        donor index for every selected external base, in the same order as
        compatibility_lists.
    """

    base_count = len(
        compatibility_lists
    )

    donor_count = len(
        donors
    )

    if base_count <= 0:
        raise ManifestDesignAuditError(
            "Cannot solve matching for an empty base set."
        )

    if donor_count <= 0:
        raise ManifestDesignAuditError(
            "Cannot solve matching for an empty donor set."
        )

    rng = np.random.default_rng(
        seed
        + 1
    )

    ordered_candidates: list[
        np.ndarray
    ] = []

    degrees = np.zeros(
        base_count,
        dtype=np.int64,
    )

    for base_index, compatible_indices in enumerate(
        compatibility_lists
    ):
        compatible_indices = np.asarray(
            compatible_indices,
            dtype=np.int64,
        )

        if compatible_indices.size == 0:
            raise ManifestDesignAuditError(
                "A selected external base has no compatible donors."
            )

        if (
            compatible_indices.min() < 0
            or compatible_indices.max() >= donor_count
        ):
            raise ManifestDesignAuditError(
                "Compatibility list contains an out-of-range "
                "donor index."
            )

        if np.unique(
            compatible_indices
        ).size != compatible_indices.size:
            raise ManifestDesignAuditError(
                "Compatibility list contains duplicate donor indices."
            )

        ordered = build_matching_candidate_order(
            compatible_indices=compatible_indices,
            donors=donors,
            rng=rng,
        )

        ordered_candidates.append(
            ordered
        )

        degrees[
            base_index
        ] = ordered.size

    # Random tie-break values make equal-degree ordering reproducible while
    # avoiding systematic preference for low subject IDs.
    tie_break = rng.random(
        base_count
    )

    base_order = sorted(
        range(
            base_count
        ),
        key=lambda index: (
            int(
                degrees[
                    index
                ]
            ),
            float(
                tie_break[
                    index
                ]
            ),
        ),
    )

    base_to_donor = np.full(
        base_count,
        -1,
        dtype=np.int64,
    )

    donor_to_base = np.full(
        donor_count,
        -1,
        dtype=np.int64,
    )

    def try_assign(
        base_index: int,
        seen_donors: np.ndarray,
    ) -> bool:
        """
        Attempt to assign one base, recursively rerouting earlier matches.

        This is the augmenting-path step that prevents a flexible early base
        from permanently consuming a donor required by a later constrained
        base.
        """

        for donor_index_value in ordered_candidates[
            base_index
        ]:
            donor_index = int(
                donor_index_value
            )

            if seen_donors[
                donor_index
            ]:
                continue

            seen_donors[
                donor_index
            ] = True

            currently_matched_base = int(
                donor_to_base[
                    donor_index
                ]
            )

            if (
                currently_matched_base
                == -1
            ):
                base_to_donor[
                    base_index
                ] = donor_index

                donor_to_base[
                    donor_index
                ] = base_index

                return True

            if try_assign(
                currently_matched_base,
                seen_donors,
            ):
                base_to_donor[
                    base_index
                ] = donor_index

                donor_to_base[
                    donor_index
                ] = base_index

                return True

        return False

    print()
    print(
        "Solving global one-to-one base-donor matching..."
    )

    print(
        f"External bases           : "
        f"{base_count:,}"
    )

    print(
        f"Donor slices             : "
        f"{donor_count:,}"
    )

    print(
        f"Minimum base degree      : "
        f"{int(degrees.min()):,}"
    )

    print(
        f"Median base degree       : "
        f"{float(np.median(degrees)):,.1f}"
    )

    print(
        f"Maximum base degree      : "
        f"{int(degrees.max()):,}"
    )

    matched_count = 0

    for order_position, base_index in enumerate(
        base_order,
        start=1,
    ):
        seen_donors = np.zeros(
            donor_count,
            dtype=bool,
        )

        success = try_assign(
            base_index,
            seen_donors,
        )

        if not success:
            raise ManifestDesignAuditError(
                "Global one-to-one donor matching failed.\n"
                f"Matched bases before failure: {matched_count:,} / "
                f"{base_count:,}\n"
                f"Failed base index: {base_index}\n"
                f"Compatible donors for failed base: "
                f"{int(degrees[base_index]):,}\n\n"
                "This indicates that the selected compatibility graph "
                "does not admit a complete one-to-one assignment under "
                "the no-donor-reuse requirement."
            )

        matched_count += 1

        if (
            order_position % 100 == 0
            or order_position == base_count
        ):
            print(
                f"  Matched bases: "
                f"{order_position:,} / "
                f"{base_count:,}",
                flush=True,
            )

    if (
        base_to_donor
        < 0
    ).any():
        raise ManifestDesignAuditError(
            "Global matching completed with an unmatched external base."
        )

    if np.unique(
        base_to_donor
    ).size != base_count:
        raise ManifestDesignAuditError(
            "Global matching reused at least one donor slice."
        )

    # Verify every final edge belongs to the original compatibility graph.
    for base_index, donor_index_value in enumerate(
        base_to_donor
    ):
        donor_index = int(
            donor_index_value
        )

        if not np.any(
            compatibility_lists[
                base_index
            ]
            == donor_index
        ):
            raise ManifestDesignAuditError(
                "Global matching produced an edge outside "
                "the original compatibility graph."
            )

    print()
    print(
        "Global one-to-one matching: PASS"
    )

    print(
        f"Matched external bases   : "
        f"{base_count:,}"
    )

    print(
        f"Unique donor slices      : "
        f"{np.unique(base_to_donor).size:,}"
    )

    return base_to_donor


def compute_pair_overlap_fraction(
    *,
    base_mask: np.ndarray,
    packed_donor_masks: np.ndarray,
    donor_index: int,
    donor_area: int,
) -> float:
    """Compute exact selected-pair mask-inside-brain fraction."""

    donor_mask = np.unpackbits(
        packed_donor_masks[
            donor_index:
            donor_index
            + 1
        ],
        axis=1,
        count=IMAGE_AREA,
    )[
        0
    ].astype(
        np.uint8,
        copy=False,
    )

    overlap_pixels = int(
        np.logical_and(
            base_mask > 0,
            donor_mask > 0,
        ).sum()
    )

    return float(
        overlap_pixels
        / donor_area
    )


def assign_nested_donors(
    *,
    selected_bases: pd.DataFrame,
    base_brain_masks: np.ndarray,
    donors: pd.DataFrame,
    packed_donor_masks: np.ndarray,
    device: torch.device,
    min_overlap: float,
    base_batch_size: int,
    donor_batch_size: int,
    seed: int,
) -> pd.DataFrame:
    """
    Reconstruct the complete compatibility graph and solve donor assignment.

    Exact compatibility is recomputed for every one of the 625 selected
    external bases and required to reproduce the counts from the completed
    pair-space audit.

    Donor assignments are then solved globally as a one-to-one bipartite
    matching problem. No external base is assigned until the complete
    selected-base compatibility graph has been reconstructed.
    """

    donor_areas = donors[
        "tumor_area_pixels"
    ].to_numpy(
        dtype=np.int64
    )

    print()
    print(
        "Reconstructing selected-base compatibility graph..."
    )

    print(
        f"Device                    : {device}"
    )

    print(
        f"Selected bases            : "
        f"{len(selected_bases):,}"
    )

    print(
        f"Donor population          : "
        f"{len(donors):,}"
    )

    all_compatible_lists: list[
        np.ndarray
    ] = []

    for base_start in range(
        0,
        len(
            selected_bases
        ),
        base_batch_size,
    ):
        base_end = min(
            base_start
            + base_batch_size,
            len(
                selected_bases
            ),
        )

        base_batch = base_brain_masks[
            base_start:base_end
        ]

        compatible_lists = (
            compute_compatible_donor_indices_for_batch(
                base_masks=base_batch,
                packed_donor_masks=packed_donor_masks,
                donor_areas=donor_areas,
                device=device,
                min_overlap=min_overlap,
                donor_batch_size=donor_batch_size,
            )
        )

        if len(
            compatible_lists
        ) != (
            base_end
            - base_start
        ):
            raise ManifestDesignAuditError(
                "Compatibility computation returned an "
                "unexpected number of base rows."
            )

        for local_position, compatible_indices in enumerate(
            compatible_lists
        ):
            global_position = (
                base_start
                + local_position
            )

            base_row = selected_bases.iloc[
                global_position
            ]

            expected_count = int(
                base_row[
                    "compatible_donor_count"
                ]
            )

            observed_count = int(
                compatible_indices.size
            )

            if (
                observed_count
                != expected_count
            ):
                raise ManifestDesignAuditError(
                    "Recomputed base-donor compatibility count "
                    "disagrees with the completed pair-space audit.\n"
                    f"Case: {base_row['case_id']}\n"
                    f"Observed: {observed_count:,}\n"
                    f"Expected: {expected_count:,}"
                )

            all_compatible_lists.append(
                compatible_indices.astype(
                    np.int64,
                    copy=False,
                )
            )

        print(
            f"  Compatibility graph: "
            f"{base_end:,} / "
            f"{len(selected_bases):,} bases",
            flush=True,
        )

    if len(
        all_compatible_lists
    ) != len(
        selected_bases
    ):
        raise ManifestDesignAuditError(
            "Complete compatibility graph has an unexpected "
            "number of selected-base rows."
        )

    print()
    print(
        "Compatibility graph reconstruction: PASS"
    )

    print(
        f"Selected bases            : "
        f"{len(all_compatible_lists):,}"
    )

    print(
        f"Total selected graph edges: "
        f"{sum(x.size for x in all_compatible_lists):,}"
    )

    matched_donor_indices = (
        solve_unique_donor_matching(
            compatibility_lists=all_compatible_lists,
            donors=donors,
            seed=seed,
        )
    )

    if matched_donor_indices.shape != (
        len(
            selected_bases
        ),
    ):
        raise ManifestDesignAuditError(
            "Global matching returned an unexpected result shape."
        )

    assignment_rows: list[
        dict[str, Any]
    ] = []

    for global_position in range(
        len(
            selected_bases
        )
    ):
        base_row = selected_bases.iloc[
            global_position
        ]

        compatible_indices = (
            all_compatible_lists[
                global_position
            ]
        )

        selected_donor_index = int(
            matched_donor_indices[
                global_position
            ]
        )

        donor_row = donors.iloc[
            selected_donor_index
        ]

        if not np.any(
            compatible_indices
            == selected_donor_index
        ):
            raise ManifestDesignAuditError(
                "Matched donor is not in the base's exact "
                "compatibility set."
            )

        overlap_fraction = (
            compute_pair_overlap_fraction(
                base_mask=base_brain_masks[
                    global_position
                ],
                packed_donor_masks=packed_donor_masks,
                donor_index=selected_donor_index,
                donor_area=int(
                    donor_row[
                        "tumor_area_pixels"
                    ]
                ),
            )
        )

        if (
            overlap_fraction
            + 1e-12
            < min_overlap
        ):
            raise ManifestDesignAuditError(
                "Matched donor assignment does not satisfy "
                "minimum mask-inside-brain overlap."
            )

        compatible_volume_count = int(
            donors.iloc[
                compatible_indices
            ][
                "volume"
            ].nunique()
        )

        assignment_rows.append(
            {
                "case_id":
                    str(
                        base_row[
                            "case_id"
                        ]
                    ),

                "base_rank":
                    int(
                        base_row[
                            "base_rank"
                        ]
                    ),

                "external_subject":
                    str(
                        base_row[
                            "subject"
                        ]
                    ),

                "external_subject_numeric_id":
                    int(
                        base_row[
                            "subject_numeric_id"
                        ]
                    ),

                "external_slice_index":
                    int(
                        base_row[
                            "slice_index"
                        ]
                    ),

                "external_modality":
                    str(
                        base_row[
                            "reference_modality"
                        ]
                    ).lower(),

                "external_brain_pixels":
                    int(
                        base_row[
                            "brain_pixels"
                        ]
                    ),

                "external_compatible_donor_count":
                    int(
                        base_row[
                            "compatible_donor_count"
                        ]
                    ),

                "compatible_training_volume_count":
                    compatible_volume_count,

                "donor_index":
                    selected_donor_index,

                "donor_slice_path":
                    str(
                        donor_row[
                            "slice_path"
                        ]
                    ),

                "donor_h5_path":
                    str(
                        donor_row[
                            "donor_h5_path"
                        ]
                    ),

                "donor_volume":
                    int(
                        donor_row[
                            "volume"
                        ]
                    ),

                "donor_slice_index":
                    int(
                        donor_row[
                            "slice"
                        ]
                    ),

                "donor_tumor_area_pixels":
                    int(
                        donor_row[
                            "tumor_area_pixels"
                        ]
                    ),

                "donor_tumor_area_fraction":
                    float(
                        donor_row[
                            "tumor_area_fraction"
                        ]
                    ),

                "donor_bbox_area":
                    int(
                        donor_row[
                            "bbox_area"
                        ]
                    ),

                "donor_bbox_fill_fraction":
                    float(
                        donor_row[
                            "bbox_fill_fraction"
                        ]
                    ),

                "donor_centroid_x_normalized":
                    float(
                        donor_row[
                            "centroid_x_normalized"
                        ]
                    ),

                "donor_centroid_y_normalized":
                    float(
                        donor_row[
                            "centroid_y_normalized"
                        ]
                    ),

                "donor_centroid_laterality":
                    str(
                        donor_row[
                            "centroid_laterality"
                        ]
                    ),

                "donor_connected_component_count":
                    int(
                        donor_row[
                            "connected_component_count"
                        ]
                    ),

                "donor_largest_component_fraction":
                    float(
                        donor_row[
                            "largest_component_fraction"
                        ]
                    ),

                "donor_compatible_external_base_count":
                    int(
                        donor_row[
                            "compatible_external_base_count"
                        ]
                    ),

                "selected_pair_mask_inside_brain_fraction":
                    overlap_fraction,
            }
        )

    assignments = pd.DataFrame(
        assignment_rows
    )

    if len(
        assignments
    ) != len(
        selected_bases
    ):
        raise ManifestDesignAuditError(
            "Unexpected donor-assignment row count."
        )

    if assignments[
        "donor_h5_path"
    ].duplicated().any():
        raise ManifestDesignAuditError(
            "Exact donor-slice reuse occurred after "
            "global bipartite matching."
        )

    if (
        assignments[
            "selected_pair_mask_inside_brain_fraction"
        ]
        < min_overlap
    ).any():
        raise ManifestDesignAuditError(
            "At least one final matched pair violates "
            "the established overlap threshold."
        )

    print()
    print(
        "Final assignment audit: PASS"
    )

    print(
        f"Assigned external bases  : "
        f"{len(assignments):,}"
    )

    print(
        f"Unique donor slices      : "
        f"{assignments['donor_h5_path'].nunique():,}"
    )

    print(
        f"Training volumes used    : "
        f"{assignments['donor_volume'].nunique():,}"
    )

    return assignments


def empirical_ks_distance(
    reference: np.ndarray,
    sample: np.ndarray,
) -> float:
    """Return two-sample empirical Kolmogorov-Smirnov distance."""

    reference = np.asarray(
        reference,
        dtype=np.float64,
    )

    sample = np.asarray(
        sample,
        dtype=np.float64,
    )

    reference = reference[
        np.isfinite(
            reference
        )
    ]

    sample = sample[
        np.isfinite(
            sample
        )
    ]

    if (
        reference.size == 0
        or sample.size == 0
    ):
        return float(
            "nan"
        )

    values = np.sort(
        np.unique(
            np.concatenate(
                [
                    reference,
                    sample,
                ]
            )
        )
    )

    reference_sorted = np.sort(
        reference
    )

    sample_sorted = np.sort(
        sample
    )

    reference_ecdf = (
        np.searchsorted(
            reference_sorted,
            values,
            side="right",
        )
        / reference_sorted.size
    )

    sample_ecdf = (
        np.searchsorted(
            sample_sorted,
            values,
            side="right",
        )
        / sample_sorted.size
    )

    return float(
        np.max(
            np.abs(
                reference_ecdf
                - sample_ecdf
            )
        )
    )


def standardized_mean_difference(
    reference: np.ndarray,
    sample: np.ndarray,
) -> float:
    """Return sample-minus-reference mean difference in reference SD units."""

    reference = np.asarray(
        reference,
        dtype=np.float64,
    )

    sample = np.asarray(
        sample,
        dtype=np.float64,
    )

    reference = reference[
        np.isfinite(
            reference
        )
    ]

    sample = sample[
        np.isfinite(
            sample
        )
    ]

    if (
        reference.size < 2
        or sample.size == 0
    ):
        return float(
            "nan"
        )

    reference_sd = float(
        reference.std(
            ddof=1
        )
    )

    if reference_sd == 0:
        return 0.0

    return float(
        (
            sample.mean()
            - reference.mean()
        )
        / reference_sd
    )


def nearest_within_subject_gaps(
    candidate: pd.DataFrame,
) -> np.ndarray:
    """Return minimum selected slice gap within each subject having >1 case."""

    gaps = []

    for _, group in candidate.groupby(
        "external_subject_numeric_id"
    ):
        slices = np.sort(
            group[
                "external_slice_index"
            ].to_numpy(
                dtype=np.int64
            )
        )

        if slices.size <= 1:
            continue

        gaps.append(
            int(
                np.diff(
                    slices
                ).min()
            )
        )

    return np.asarray(
        gaps,
        dtype=np.float64,
    )


def build_design_summary(
    assignments: pd.DataFrame,
) -> pd.DataFrame:
    """Build one-row summary per nested candidate design."""

    rows = []

    for candidate_name, quota in (
        CANDIDATE_QUOTAS.items()
    ):
        candidate = assignments.loc[
            assignments[
                "base_rank"
            ]
            <= quota
        ].copy()

        expected_cases = (
            CANDIDATE_CASE_COUNTS[
                candidate_name
            ]
        )

        if len(
            candidate
        ) != expected_cases:
            raise ManifestDesignAuditError(
                f"{candidate_name} does not contain "
                f"{expected_cases} cases."
            )

        within_subject_gaps = (
            nearest_within_subject_gaps(
                candidate
            )
        )

        donor_volume_counts = (
            candidate[
                "donor_volume"
            ].value_counts()
        )

        rows.append(
            {
                "candidate":
                    candidate_name,

                "bases_per_validation_subject":
                    quota,

                "case_count":
                    len(
                        candidate
                    ),

                "validation_subject_count":
                    int(
                        candidate[
                            "external_subject_numeric_id"
                        ].nunique()
                    ),

                "unique_external_base_count":
                    int(
                        candidate[
                            [
                                "external_subject_numeric_id",
                                "external_slice_index",
                            ]
                        ]
                        .drop_duplicates()
                        .shape[
                            0
                        ]
                    ),

                "unique_donor_slice_count":
                    int(
                        candidate[
                            "donor_h5_path"
                        ].nunique()
                    ),

                "unique_training_volume_count":
                    int(
                        candidate[
                            "donor_volume"
                        ].nunique()
                    ),

                "maximum_cases_from_one_training_volume":
                    int(
                        donor_volume_counts.max()
                    ),

                "median_cases_per_used_training_volume":
                    float(
                        donor_volume_counts.median()
                    ),

                "mean_cases_per_used_training_volume":
                    float(
                        donor_volume_counts.mean()
                    ),

                "minimum_selected_pair_overlap":
                    float(
                        candidate[
                            "selected_pair_mask_inside_brain_fraction"
                        ].min()
                    ),

                "median_selected_pair_overlap":
                    float(
                        candidate[
                            "selected_pair_mask_inside_brain_fraction"
                        ].median()
                    ),

                "minimum_compatible_donors_per_selected_base":
                    int(
                        candidate[
                            "external_compatible_donor_count"
                        ].min()
                    ),

                "median_compatible_donors_per_selected_base":
                    float(
                        candidate[
                            "external_compatible_donor_count"
                        ].median()
                    ),

                "minimum_compatible_training_volumes_per_selected_base":
                    int(
                        candidate[
                            "compatible_training_volume_count"
                        ].min()
                    ),

                "median_compatible_training_volumes_per_selected_base":
                    float(
                        candidate[
                            "compatible_training_volume_count"
                        ].median()
                    ),

                "minimum_within_subject_selected_slice_gap":
                    (
                        float(
                            within_subject_gaps.min()
                        )
                        if within_subject_gaps.size
                        else np.nan
                    ),

                "median_within_subject_selected_slice_gap":
                    (
                        float(
                            np.median(
                                within_subject_gaps
                            )
                        )
                        if within_subject_gaps.size
                        else np.nan
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


def build_distribution_comparison(
    *,
    assignments: pd.DataFrame,
    eligible_bases: pd.DataFrame,
    donors: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compare each candidate with the complete eligible slice populations.

    These are descriptive diagnostics, not pass/fail thresholds.
    """

    base_metrics = {
        "external_slice_index": (
            eligible_bases[
                "slice_index"
            ].to_numpy(
                dtype=np.float64
            ),
            "external_slice_index",
        ),

        "external_brain_pixels": (
            eligible_bases[
                "brain_pixels"
            ].to_numpy(
                dtype=np.float64
            ),
            "external_brain_pixels",
        ),

        "external_compatible_donor_count": (
            eligible_bases[
                "compatible_donor_count"
            ].to_numpy(
                dtype=np.float64
            ),
            "external_compatible_donor_count",
        ),
    }

    donor_metrics = {
        "donor_tumor_area_pixels": (
            donors[
                "tumor_area_pixels"
            ].to_numpy(
                dtype=np.float64
            ),
            "donor_tumor_area_pixels",
        ),

        "donor_bbox_area": (
            donors[
                "bbox_area"
            ].to_numpy(
                dtype=np.float64
            ),
            "donor_bbox_area",
        ),

        "donor_connected_component_count": (
            donors[
                "connected_component_count"
            ].to_numpy(
                dtype=np.float64
            ),
            "donor_connected_component_count",
        ),

        "donor_largest_component_fraction": (
            donors[
                "largest_component_fraction"
            ].to_numpy(
                dtype=np.float64
            ),
            "donor_largest_component_fraction",
        ),

        "donor_centroid_x_normalized": (
            donors[
                "centroid_x_normalized"
            ].to_numpy(
                dtype=np.float64
            ),
            "donor_centroid_x_normalized",
        ),

        "donor_centroid_y_normalized": (
            donors[
                "centroid_y_normalized"
            ].to_numpy(
                dtype=np.float64
            ),
            "donor_centroid_y_normalized",
        ),

        "donor_compatible_external_base_count": (
            donors[
                "compatible_external_base_count"
            ].to_numpy(
                dtype=np.float64
            ),
            "donor_compatible_external_base_count",
        ),
    }

    rows = []

    for candidate_name, quota in (
        CANDIDATE_QUOTAS.items()
    ):
        candidate = assignments.loc[
            assignments[
                "base_rank"
            ]
            <= quota
        ]

        for metric_name, (
            reference,
            candidate_column,
        ) in {
            **base_metrics,
            **donor_metrics,
        }.items():
            sample = candidate[
                candidate_column
            ].to_numpy(
                dtype=np.float64
            )

            rows.append(
                {
                    "candidate":
                        candidate_name,

                    "metric":
                        metric_name,

                    "reference_count":
                        int(
                            reference.size
                        ),

                    "candidate_count":
                        int(
                            sample.size
                        ),

                    "reference_mean":
                        float(
                            reference.mean()
                        ),

                    "candidate_mean":
                        float(
                            sample.mean()
                        ),

                    "reference_median":
                        float(
                            np.median(
                                reference
                            )
                        ),

                    "candidate_median":
                        float(
                            np.median(
                                sample
                            )
                        ),

                    "standardized_mean_difference":
                        standardized_mean_difference(
                            reference,
                            sample,
                        ),

                    "empirical_ks_distance":
                        empirical_ks_distance(
                            reference,
                            sample,
                        ),
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
    """Return ECDF x/y coordinates."""

    values = np.asarray(
        values,
        dtype=np.float64,
    )

    values = values[
        np.isfinite(
            values
        )
    ]

    x = np.sort(
        values
    )

    y = (
        np.arange(
            1,
            x.size
            + 1,
            dtype=np.float64,
        )
        / x.size
    )

    return (
        x,
        y,
    )


def save_ecdf_figure(
    *,
    reference: np.ndarray,
    candidate_values: dict[str, np.ndarray],
    xlabel: str,
    title: str,
    path: Path,
    dpi: int,
) -> None:
    """Save ECDF comparison figure."""

    figure, axis = plt.subplots(
        figsize=(
            8,
            5,
        )
    )

    x_reference, y_reference = (
        ecdf_coordinates(
            reference
        )
    )

    axis.plot(
        x_reference,
        y_reference,
        label="Eligible population",
    )

    for candidate_name, values in (
        candidate_values.items()
    ):
        x_candidate, y_candidate = (
            ecdf_coordinates(
                values
            )
        )

        axis.plot(
            x_candidate,
            y_candidate,
            label=candidate_name,
        )

    axis.set_xlabel(
        xlabel
    )

    axis.set_ylabel(
        "Empirical cumulative probability"
    )

    axis.set_title(
        title
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


def save_candidate_figures(
    *,
    assignments: pd.DataFrame,
    eligible_bases: pd.DataFrame,
    donors: pd.DataFrame,
    figure_dir: Path,
    dpi: int,
) -> None:
    """Save candidate-distribution comparison figures."""

    candidate_tables = {
        candidate_name:
            assignments.loc[
                assignments[
                    "base_rank"
                ]
                <= quota
            ]
        for candidate_name, quota in (
            CANDIDATE_QUOTAS.items()
        )
    }

    save_ecdf_figure(
        reference=eligible_bases[
            "slice_index"
        ].to_numpy(),
        candidate_values={
            name:
                table[
                    "external_slice_index"
                ].to_numpy()
            for name, table in (
                candidate_tables.items()
            )
        },
        xlabel="External axial slice index",
        title="Candidate External Slice-Index Distributions",
        path=(
            figure_dir
            / "external_slice_index_ecdf.png"
        ),
        dpi=dpi,
    )

    save_ecdf_figure(
        reference=eligible_bases[
            "brain_pixels"
        ].to_numpy(),
        candidate_values={
            name:
                table[
                    "external_brain_pixels"
                ].to_numpy()
            for name, table in (
                candidate_tables.items()
            )
        },
        xlabel="External brain pixels",
        title="Candidate External Brain-Content Distributions",
        path=(
            figure_dir
            / "external_brain_pixels_ecdf.png"
        ),
        dpi=dpi,
    )

    save_ecdf_figure(
        reference=eligible_bases[
            "compatible_donor_count"
        ].to_numpy(),
        candidate_values={
            name:
                table[
                    "external_compatible_donor_count"
                ].to_numpy()
            for name, table in (
                candidate_tables.items()
            )
        },
        xlabel="Compatible donors per external base",
        title="Candidate External Compatibility Distributions",
        path=(
            figure_dir
            / "base_compatible_donors_ecdf.png"
        ),
        dpi=dpi,
    )

    save_ecdf_figure(
        reference=donors[
            "tumor_area_pixels"
        ].to_numpy(),
        candidate_values={
            name:
                table[
                    "donor_tumor_area_pixels"
                ].to_numpy()
            for name, table in (
                candidate_tables.items()
            )
        },
        xlabel="Donor whole-tumor area (pixels)",
        title="Candidate Donor Tumor-Area Distributions",
        path=(
            figure_dir
            / "donor_tumor_area_ecdf.png"
        ),
        dpi=dpi,
    )

    save_ecdf_figure(
        reference=donors[
            "connected_component_count"
        ].to_numpy(),
        candidate_values={
            name:
                table[
                    "donor_connected_component_count"
                ].to_numpy()
            for name, table in (
                candidate_tables.items()
            )
        },
        xlabel="Donor 8-connected component count",
        title="Candidate Donor Component-Count Distributions",
        path=(
            figure_dir
            / "donor_component_count_ecdf.png"
        ),
        dpi=dpi,
    )

    save_ecdf_figure(
        reference=donors[
            "compatible_external_base_count"
        ].to_numpy(),
        candidate_values={
            name:
                table[
                    "donor_compatible_external_base_count"
                ].to_numpy()
            for name, table in (
                candidate_tables.items()
            )
        },
        xlabel="Compatible external bases per donor",
        title="Candidate Donor Compatibility Distributions",
        path=(
            figure_dir
            / "donor_compatibility_ecdf.png"
        ),
        dpi=dpi,
    )


def write_candidate_manifests(
    *,
    assignments: pd.DataFrame,
    output_dir: Path,
) -> dict[str, str]:
    """Write evaluator-compatible nested candidate manifests."""

    paths: dict[
        str,
        str
    ] = {}

    for candidate_name, quota in (
        CANDIDATE_QUOTAS.items()
    ):
        candidate = (
            assignments.loc[
                assignments[
                    "base_rank"
                ]
                <= quota
            ]
            .copy()
            .sort_values(
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

        manifest = candidate.loc[
            :,
            EVALUATOR_MANIFEST_COLUMNS,
        ]

        expected_cases = (
            CANDIDATE_CASE_COUNTS[
                candidate_name
            ]
        )

        if len(
            manifest
        ) != expected_cases:
            raise ManifestDesignAuditError(
                f"{candidate_name} manifest has an "
                "unexpected row count."
            )

        path = (
            output_dir
            / (
                f"{candidate_name}"
                "_manifest.csv"
            )
        )

        manifest.to_csv(
            path,
            index=False,
        )

        paths[
            candidate_name
        ] = str(
            path
        )

    return paths


def json_safe(
    value: Any,
) -> Any:
    """
    Convert NumPy/scalar values to strict JSON-compatible objects.

    Non-finite floating-point values are represented as JSON null rather
    than the non-standard NaN/Infinity literals accepted by Python's
    default JSON encoder.
    """

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


def main() -> None:
    """Run candidate external-manifest design audit."""

    args = parse_args()

    if args.base_batch_size <= 0:
        raise ManifestDesignAuditError(
            "--base-batch-size must be positive."
        )

    if args.donor_batch_size <= 0:
        raise ManifestDesignAuditError(
            "--donor-batch-size must be positive."
        )

    if args.dpi <= 0:
        raise ManifestDesignAuditError(
            "--dpi must be positive."
        )

    if not (
        0.0
        <= args.brain_threshold
        <= 1.0
    ):
        raise ManifestDesignAuditError(
            "--brain-threshold must lie in [0, 1]."
        )

    if not (
        0.0
        < args.min_overlap
        <= 1.0
    ):
        raise ManifestDesignAuditError(
            "--min-overlap must lie in (0, 1]."
        )

    external_modality = (
        str(
            args.external_modality
        )
        .strip()
        .lower()
    )

    if external_modality != "flair":
        raise ManifestDesignAuditError(
            "This audit currently requires FLAIR so that the "
            "external base representation matches the established "
            "channel-0 synthesis configuration."
        )

    validation_dataset_path = (
        resolve_existing_file(
            args.validation_dataset,
            name="Validation dataset specification",
        )
    )

    base_counts_path = (
        resolve_existing_file(
            args.base_counts_csv,
            name="Base compatibility CSV",
        )
    )

    donor_morphology_path = (
        resolve_existing_file(
            args.donor_morphology_csv,
            name="Donor morphology CSV",
        )
    )

    pair_space_summary_path = (
        resolve_existing_file(
            args.pair_space_summary,
            name="Pair-space summary JSON",
        )
    )

    output_dir = (
        args.output_dir
        .expanduser()
        .resolve()
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

    set_seed(
        args.seed
    )

    device = resolve_device(
        args.device
    )

    validation_dataset = (
        load_validation_dataset_specification(
            validation_dataset_path
        )
    )

    bases = pd.read_csv(
        base_counts_path
    )

    donors = pd.read_csv(
        donor_morphology_path
    )

    with pair_space_summary_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        pair_summary = json.load(
            file
        )

    eligible_bases = (
        validate_source_tables(
            bases=bases,
            donors=donors,
            pair_summary=pair_summary,
        )
    )

    expected_brain_threshold = float(
        pair_summary[
            "audit_definition"
        ][
            "brain_threshold"
        ]
    )

    expected_min_overlap = float(
        pair_summary[
            "audit_definition"
        ][
            "minimum_mask_inside_brain_fraction"
        ]
    )

    if not np.isclose(
        args.brain_threshold,
        expected_brain_threshold,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ManifestDesignAuditError(
            "Configured brain threshold disagrees with "
            "completed pair-space audit."
        )

    if not np.isclose(
        args.min_overlap,
        expected_min_overlap,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ManifestDesignAuditError(
            "Configured minimum overlap disagrees with "
            "completed pair-space audit."
        )

    print()
    print(
        "=" * 78
    )

    print(
        "EXTERNAL BR-LoRA CANDIDATE MANIFEST DESIGN AUDIT"
    )

    print(
        "=" * 78
    )

    print(
        f"Eligible external bases  : "
        f"{len(eligible_bases):,}"
    )

    print(
        f"Validation subjects      : "
        f"{eligible_bases['subject_numeric_id'].nunique():,}"
    )

    print(
        f"Eligible donors          : "
        f"{len(donors):,}"
    )

    print(
        f"Training volumes         : "
        f"{donors['volume'].nunique():,}"
    )

    print(
        f"Candidate sizes          : "
        "125, 250, 625"
    )

    print(
        f"Base sampling seed       : "
        f"{args.seed}"
    )

    print(
        f"Donor assignment seed    : "
        f"{args.seed + 1}"
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
        f"External modality        : "
        f"{external_modality}"
    )

    print(
        f"Computation device       : "
        f"{device}"
    )

    print(
        "=" * 78
    )

    selected_bases = (
        sample_nested_bases(
            eligible_bases,
            seed=args.seed,
        )
    )

    print()
    print(
        "Nested base sampling"
    )

    print(
        "-" * 78
    )

    for candidate_name, quota in (
        CANDIDATE_QUOTAS.items()
    ):
        count = int(
            (
                selected_bases[
                    "base_rank"
                ]
                <= quota
            ).sum()
        )

        print(
            f"{candidate_name:<24}: "
            f"{count:,} cases"
        )

    print(
        "-" * 78
    )

    base_brain_masks = (
        load_selected_base_brain_masks(
            selected_bases=selected_bases,
            validation_dataset=validation_dataset,
            external_modality=external_modality,
            brain_threshold=args.brain_threshold,
        )
    )

    packed_donor_masks = (
        load_and_pack_donor_masks(
            donors
        )
    )

    assignments = (
        assign_nested_donors(
            selected_bases=selected_bases,
            base_brain_masks=base_brain_masks,
            donors=donors,
            packed_donor_masks=packed_donor_masks,
            device=device,
            min_overlap=args.min_overlap,
            base_batch_size=args.base_batch_size,
            donor_batch_size=args.donor_batch_size,
            seed=args.seed,
        )
    )

    assignment_path = (
        output_dir
        / OUTPUT_ASSIGNMENTS_NAME
    )

    assignments.to_csv(
        assignment_path,
        index=False,
    )

    manifest_paths = (
        write_candidate_manifests(
            assignments=assignments,
            output_dir=output_dir,
        )
    )

    design_summary = (
        build_design_summary(
            assignments
        )
    )

    design_summary_path = (
        output_dir
        / OUTPUT_DESIGN_SUMMARY_NAME
    )

    design_summary.to_csv(
        design_summary_path,
        index=False,
    )

    distribution_comparison = (
        build_distribution_comparison(
            assignments=assignments,
            eligible_bases=eligible_bases,
            donors=donors,
        )
    )

    distribution_path = (
        output_dir
        / OUTPUT_DISTRIBUTION_NAME
    )

    distribution_comparison.to_csv(
        distribution_path,
        index=False,
    )

    save_candidate_figures(
        assignments=assignments,
        eligible_bases=eligible_bases,
        donors=donors,
        figure_dir=figure_dir,
        dpi=args.dpi,
    )

    summary = {
        "audit_role":
            (
                "Nested candidate-manifest design audit. "
                "No definitive external evaluation manifest "
                "is selected or frozen."
            ),

        "design": {
            "candidate_case_counts":
                CANDIDATE_CASE_COUNTS,

            "bases_per_validation_subject":
                CANDIDATE_QUOTAS,

            "nested":
                True,

            "base_sampling":
                (
                    "Uniform sampling without replacement within "
                    "each validation subject."
                ),

            "donor_assignment":
                (
                    "Global one-to-one bipartite matching over the complete "
                    "selected-base compatibility graph. Every external base "
                    "is assigned exactly one compatible donor and each donor "
                    "slice is used at most once. Bases are processed from "
                    "smallest to largest compatibility set, with augmenting "
                    "paths used to reroute prior assignments when necessary."
                ),

            "donor_preference_order":
                (
                    "Within each base's complete compatible-donor set, "
                    "training volumes and donor slices are reproducibly "
                    "shuffled and donors are interleaved across volumes. "
                    "This affects traversal preference only and does not "
                    "remove compatibility edges or impose a volume quota."
                ),

            "exact_donor_slice_reuse":
                False,

            "assignment_order":
                (
                    "The complete 625-base compatibility graph is solved "
                    "globally. The 125- and 250-case candidate manifests "
                    "inherit the donor assignments of their corresponding "
                    "nested base subsets from that completed matching."
                ),
        },

        "established_eligibility_rules": {
            "external_base":
                (
                    "tumor_free_candidate == True, "
                    "predicted_tumor_pixels == 0, "
                    "compatible_donor_count > 0"
                ),

            "donor":
                (
                    "whole_tumor_pixels >= 300 and "
                    "established mask-margin rule already passed"
                ),

            "brain_threshold":
                args.brain_threshold,

            "minimum_mask_inside_brain_fraction":
                args.min_overlap,

            "external_modality":
                external_modality,

            "donor_image_channel":
                0,
        },

        "population": {
            "composition_eligible_external_bases":
                int(
                    len(
                        eligible_bases
                    )
                ),

            "validation_subjects":
                int(
                    eligible_bases[
                        "subject_numeric_id"
                    ].nunique()
                ),

            "eligible_donor_slices":
                int(
                    len(
                        donors
                    )
                ),

            "training_volumes":
                int(
                    donors[
                        "volume"
                    ].nunique()
                ),
        },

        "candidate_design_summary":
            design_summary.to_dict(
                orient="records"
            ),

        "selection_status": {
            "candidate_manifests_created":
                True,

            "definitive_manifest_selected":
                False,

            "definitive_manifest_frozen":
                False,

            "br_lora_external_evaluation_run":
                False,
        },

        "source_artifacts": {
            "validation_dataset":
                str(
                    validation_dataset_path
                ),

            "base_counts_csv":
                str(
                    base_counts_path
                ),

            "donor_morphology_csv":
                str(
                    donor_morphology_path
                ),

            "pair_space_summary":
                str(
                    pair_space_summary_path
                ),
        },

        "output_artifacts": {
            "candidate_manifests":
                manifest_paths,

            "all_assignments":
                str(
                    assignment_path
                ),

            "design_summary":
                str(
                    design_summary_path
                ),

            "distribution_comparison":
                str(
                    distribution_path
                ),

            "figure_directory":
                str(
                    figure_dir
                ),
        },

        "provenance": {
            "base_sampling_seed":
                args.seed,

            "donor_assignment_seed":
                args.seed
                + 1,

            "device":
                str(
                    device
                ),

            "base_batch_size":
                args.base_batch_size,

            "donor_batch_size":
                args.donor_batch_size,

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

            "base_counts_csv_sha256":
                sha256_file(
                    base_counts_path
                ),

            "donor_morphology_csv_sha256":
                sha256_file(
                    donor_morphology_path
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

    summary_path = (
        output_dir
        / OUTPUT_JSON_NAME
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
        "CANDIDATE MANIFEST DESIGN AUDIT: PASS"
    )

    print(
        "=" * 78
    )

    print()
    print(
        design_summary.to_string(
            index=False
        )
    )

    print()
    print(
        f"Assignments              : "
        f"{assignment_path}"
    )

    print(
        f"Design summary           : "
        f"{design_summary_path}"
    )

    print(
        f"Distribution comparison : "
        f"{distribution_path}"
    )

    print(
        f"Audit summary            : "
        f"{summary_path}"
    )

    print(
        f"Figures                  : "
        f"{figure_dir}"
    )

    print()
    print(
        "Candidate manifests were created for comparison only."
    )

    print(
        "No definitive external evaluation manifest was selected."
    )

    print(
        "No BR-LoRA external evaluation was run."
    )

    print(
        "=" * 78
    )


if __name__ == "__main__":
    main()
