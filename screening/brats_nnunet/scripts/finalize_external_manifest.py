#!/usr/bin/env python3
"""
Finalize the definitive 250-case external BR-LoRA evaluation manifest.

The 250 external base images are inherited exactly from the completed nested
candidate-design audit:

    125 validation subjects
    x
    2 selected external base slices per subject
    =
    250 external cases

Important
---------
Only the external-base selections are inherited from the candidate-design
audit.

Donor assignments are NOT inherited from the 625-case global solution.

Instead, exact base-donor compatibility is reconstructed for the selected
250-base cohort and a new deterministic one-to-one global matching is solved
using only those 250 external bases.

This prevents donor assignments in the definitive 250-case experiment from
being influenced by rank-3, rank-4, or rank-5 external bases that belong only
to the discarded 625-case candidate.

Primary outputs
---------------
external_evaluation_manifest_250.csv
external_evaluation_manifest_250_assignments.csv
external_evaluation_manifest_250_summary.json
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

PROJECT_ROOT = Path(__file__).resolve().parents[3]

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

from src.inference.external_manifest import (
    load_external_evaluation_manifest,
)

from screening.brats_nnunet.scripts.audit_external_manifest_design import (
    build_matching_candidate_order,
    compute_compatible_donor_indices_for_batch,
    load_and_pack_donor_masks,
    load_selected_base_brain_masks,
    resolve_device,
)


FINAL_CASE_COUNT = 250
FINAL_BASES_PER_SUBJECT = 2
EXPECTED_SUBJECT_COUNT = 125

OUTPUT_MANIFEST = (
    "external_evaluation_manifest_250.csv"
)

OUTPUT_ASSIGNMENTS = (
    "external_evaluation_manifest_250_assignments.csv"
)

OUTPUT_SUMMARY = (
    "external_evaluation_manifest_250_summary.json"
)


class FinalManifestError(
    RuntimeError
):
    """Raised when definitive manifest construction fails."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Finalize the definitive 250-case external "
            "BR-LoRA evaluation manifest."
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
            "candidate_assignments_all.csv from the completed nested "
            "external-manifest design audit. If omitted, uses "
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
            "Directory for the definitive 250-case manifest. "
            "If omitted, uses "
            "<nnunet_run_root>/definitive_external_manifest_250."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help=(
            "Established manifest-design seed. "
            "Donor matching uses seed + 1."
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
            f"{name} does not exist:\n"
            f"{resolved}"
        )

    return resolved


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
    missing = sorted(
        columns
        - set(
            table.columns
        )
    )

    if missing:
        raise FinalManifestError(
            f"{name} is missing required column(s): "
            + ", ".join(
                missing
            )
        )


def select_final_bases(
    candidate_assignments: pd.DataFrame,
) -> pd.DataFrame:
    """
    Retain exactly the rank-1 and rank-2 external bases.

    Existing donor-assignment columns remain available only for the
    post-matching comparison audit. They are never used to construct
    the new matching.
    """

    selected = (
        candidate_assignments.loc[
            candidate_assignments[
                "base_rank"
            ]
            <= FINAL_BASES_PER_SUBJECT
        ]
        .copy()
    )

    selected = (
        selected.sort_values(
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

    if len(
        selected
    ) != FINAL_CASE_COUNT:
        raise FinalManifestError(
            "Final base selection does not contain exactly "
            f"{FINAL_CASE_COUNT} cases.\n"
            f"Observed: {len(selected)}"
        )

    if selected[
        "external_subject_numeric_id"
    ].nunique() != EXPECTED_SUBJECT_COUNT:
        raise FinalManifestError(
            "Final base selection does not contain all "
            f"{EXPECTED_SUBJECT_COUNT} validation subjects."
        )

    counts = (
        selected.groupby(
            "external_subject_numeric_id"
        )
        .size()
    )

    if not (
        counts
        == FINAL_BASES_PER_SUBJECT
    ).all():
        raise FinalManifestError(
            "Every validation subject must contribute exactly "
            f"{FINAL_BASES_PER_SUBJECT} external bases."
        )

    if selected[
        [
            "external_subject_numeric_id",
            "external_slice_index",
        ]
    ].duplicated().any():
        raise FinalManifestError(
            "Final base selection contains duplicate external slices."
        )

    return selected


def reconstruct_compatibility_graph(
    *,
    selected: pd.DataFrame,
    donors: pd.DataFrame,
    validation_dataset: Any,
    packed_donor_masks: np.ndarray,
    device: torch.device,
    brain_threshold: float,
    min_overlap: float,
    external_modality: str,
    base_batch_size: int,
    donor_batch_size: int,
) -> list[np.ndarray]:
    """Reconstruct exact compatibility sets for the final 250 bases."""

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
        "Reconstructing exact 250-base compatibility graph..."
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

        lists = (
            compute_compatible_donor_indices_for_batch(
                base_masks=brain_masks[
                    base_start:base_end
                ],
                packed_donor_masks=packed_donor_masks,
                donor_areas=donor_areas,
                device=device,
                min_overlap=min_overlap,
                donor_batch_size=donor_batch_size,
            )
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
                raise FinalManifestError(
                    "Reconstructed compatibility count disagrees "
                    "with the completed pair-space audit.\n"
                    f"Case: "
                    f"{base_table.iloc[global_index]['case_id']}\n"
                    f"Expected: {expected:,}\n"
                    f"Observed: {observed:,}"
                )

            if observed <= 0:
                raise FinalManifestError(
                    "A selected final external base has no "
                    "compatible donor."
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
    ) != FINAL_CASE_COUNT:
        raise FinalManifestError(
            "Final compatibility graph has an unexpected size."
        )

    return compatibility_lists


def solve_final_matching(
    *,
    compatibility_lists: list[np.ndarray],
    donors: pd.DataFrame,
    donor_assignment_seed: int,
) -> tuple[
    np.ndarray,
    pd.DataFrame,
]:
    """
    Solve deterministic one-to-one matching on the final 250-base graph.

    The algorithm matches the established global-matching logic:
    - reproducible donor traversal ordering,
    - bases processed from smallest compatibility set upward,
    - augmenting paths used when a preferred donor is already occupied.
    """

    base_count = len(
        compatibility_lists
    )

    donor_count = len(
        donors
    )

    rng = np.random.default_rng(
        donor_assignment_seed
    )

    ordered_candidates: list[
        np.ndarray
    ] = []

    preference_maps: list[
        dict[int, int]
    ] = []

    degrees = np.zeros(
        base_count,
        dtype=np.int64,
    )

    for base_index, compatible in enumerate(
        compatibility_lists
    ):
        ordered = build_matching_candidate_order(
            compatible_indices=compatible,
            donors=donors,
            rng=rng,
        )

        if ordered.size != compatible.size:
            raise FinalManifestError(
                "Donor preference ordering changed compatibility-set size."
            )

        ordered_candidates.append(
            ordered
        )

        preference_maps.append(
            {
                int(donor_index):
                    rank
                for rank, donor_index in enumerate(
                    ordered,
                    start=1,
                )
            }
        )

        degrees[
            base_index
        ] = int(
            ordered.size
        )

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

    augmenting_path_length = np.zeros(
        base_count,
        dtype=np.int64,
    )

    def try_assign(
        base_index: int,
        seen_donors: np.ndarray,
        *,
        depth: int,
    ) -> tuple[
        bool,
        int,
    ]:
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

            occupied_by = int(
                donor_to_base[
                    donor_index
                ]
            )

            if occupied_by == -1:
                base_to_donor[
                    base_index
                ] = donor_index

                donor_to_base[
                    donor_index
                ] = base_index

                return (
                    True,
                    depth,
                )

            success, path_length = try_assign(
                occupied_by,
                seen_donors,
                depth=depth + 1,
            )

            if success:
                base_to_donor[
                    base_index
                ] = donor_index

                donor_to_base[
                    donor_index
                ] = base_index

                return (
                    True,
                    path_length,
                )

        return (
            False,
            depth,
        )

    print()
    print(
        "Solving final 250-base one-to-one matching..."
    )

    for order_position, base_index in enumerate(
        base_order,
        start=1,
    ):
        seen_donors = np.zeros(
            donor_count,
            dtype=bool,
        )

        success, path_length = try_assign(
            base_index,
            seen_donors,
            depth=1,
        )

        if not success:
            raise FinalManifestError(
                "No complete one-to-one donor matching exists "
                "for the final 250-base cohort."
            )

        augmenting_path_length[
            base_index
        ] = int(
            path_length
        )

        if (
            order_position % 50 == 0
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
        raise FinalManifestError(
            "Final matching left at least one base unmatched."
        )

    if np.unique(
        base_to_donor
    ).size != base_count:
        raise FinalManifestError(
            "Final matching reused donor slices."
        )

    final_preference_rank = np.asarray(
        [
            preference_maps[
                base_index
            ][
                int(
                    donor_index
                )
            ]
            for base_index, donor_index in enumerate(
                base_to_donor
            )
        ],
        dtype=np.int64,
    )

    diagnostics = pd.DataFrame(
        {
            "compatibility_degree":
                degrees,

            "final_donor_index":
                base_to_donor,

            "final_preference_rank":
                final_preference_rank,

            "augmenting_path_length":
                augmenting_path_length,

            "required_rerouting":
                augmenting_path_length
                > 1,
        }
    )

    return (
        base_to_donor,
        diagnostics,
    )


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
        candidate_path = resolve_file(
            args.candidate_assignments,
            name="Candidate assignments",
        )
    else:
        candidate_path = resolve_file(
            nnunet_run_root
            / "external_manifest_design_audit"
            / "candidate_assignments_all.csv",
            name="Candidate assignments",
        )

    if args.donor_morphology_csv is not None:
        donor_path = resolve_file(
            args.donor_morphology_csv,
            name="Donor morphology CSV",
        )
    else:
        donor_path = resolve_file(
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
        )

    save_folders_config(
        args.folders_file,
        folders_config,
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest_path = (
        output_dir
        / OUTPUT_MANIFEST
    )

    assignments_path = (
        output_dir
        / OUTPUT_ASSIGNMENTS
    )

    summary_path = (
        output_dir
        / OUTPUT_SUMMARY
    )

    existing = [
        path
        for path in (
            manifest_path,
            assignments_path,
            summary_path,
        )
        if path.exists()
    ]

    if existing:
        raise FinalManifestError(
            "Refusing to overwrite existing definitive "
            "manifest artifact(s):\n"
            + "\n".join(
                str(
                    path
                )
                for path in existing
            )
        )

    candidate_assignments = pd.read_csv(
        candidate_path
    )

    donors = pd.read_csv(
        donor_path
    )

    require_columns(
        candidate_assignments,
        {
            "case_id",
            "base_rank",
            "external_subject_numeric_id",
            "external_slice_index",
            "external_modality",
            "external_brain_pixels",
            "external_compatible_donor_count",
            "donor_h5_path",
        },
        name="Candidate assignments",
    )

    require_columns(
        donors,
        {
            "donor_h5_path",
            "volume",
            "slice",
            "tumor_area_pixels",
            "bbox_area",
            "connected_component_count",
            "largest_component_fraction",
            "centroid_x_normalized",
            "centroid_y_normalized",
            "centroid_laterality",
            "compatible_external_base_count",
        },
        name="Donor morphology CSV",
    )

    selected = select_final_bases(
        candidate_assignments
    )

    # Save the inherited 625-solution donor paths only for an explicit
    # old-versus-final comparison. They play no role in final rematching.
    inherited_donor_paths = (
        selected[
            "donor_h5_path"
        ]
        .astype(
            str
        )
        .to_numpy()
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
        raise FinalManifestError(
            "Brain threshold disagrees with the completed "
            "pair-space audit."
        )

    if not np.isclose(
        args.min_overlap,
        expected_overlap,
        rtol=0.0,
        atol=1e-12,
    ):
        raise FinalManifestError(
            "Minimum overlap disagrees with the completed "
            "pair-space audit."
        )

    external_modality = str(
        args.external_modality
    ).strip().lower()

    if not (
        selected[
            "external_modality"
        ]
        .astype(
            str
        )
        .str.lower()
        == external_modality
    ).all():
        raise FinalManifestError(
            "Selected candidate bases do not all use the "
            "requested external modality."
        )

    device = resolve_device(
        args.device
    )

    validation_dataset = (
        load_validation_dataset_specification(
            validation_path
        )
    )

    donor_assignment_seed = (
        int(
            args.seed
        )
        + 1
    )

    print()
    print(
        "=" * 78
    )
    print(
        "FINALIZING 250-CASE EXTERNAL BR-LoRA MANIFEST"
    )
    print(
        "=" * 78
    )
    print(
        "Validation subjects      : 125"
    )
    print(
        "Bases per subject        : 2"
    )
    print(
        "Final external cases     : 250"
    )
    print(
        "Base-selection source    : nested candidate design, ranks <= 2"
    )
    print(
        "Inherited donor use      : NONE"
    )
    print(
        "Final donor matching     : fresh global 250-base matching"
    )
    print(
        f"Base-selection seed      : "
        f"{args.seed}"
    )
    print(
        f"Donor-assignment seed    : "
        f"{donor_assignment_seed}"
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

    compatibility_lists = (
        reconstruct_compatibility_graph(
            selected=selected,
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
    )

    (
        final_donor_indices,
        matching_diagnostics,
    ) = solve_final_matching(
        compatibility_lists=compatibility_lists,
        donors=donors,
        donor_assignment_seed=donor_assignment_seed,
    )

    final_donors = (
        donors.iloc[
            final_donor_indices
        ]
        .reset_index(
            drop=True
        )
    )

    final_donor_paths = (
        final_donors[
            "donor_h5_path"
        ]
        .astype(
            str
        )
        .to_numpy()
    )

    changed_from_inherited = (
        final_donor_paths
        != inherited_donor_paths
    )

    assignment_table = pd.DataFrame(
        {
            "case_id":
                selected[
                    "case_id"
                ].to_numpy(),

            "external_subject_numeric_id":
                selected[
                    "external_subject_numeric_id"
                ].to_numpy(
                    dtype=np.int64
                ),

            "external_slice_index":
                selected[
                    "external_slice_index"
                ].to_numpy(
                    dtype=np.int64
                ),

            "external_modality":
                np.full(
                    FINAL_CASE_COUNT,
                    external_modality,
                    dtype=object,
                ),

            "base_rank":
                selected[
                    "base_rank"
                ].to_numpy(
                    dtype=np.int64
                ),

            "external_brain_pixels":
                selected[
                    "external_brain_pixels"
                ].to_numpy(
                    dtype=np.int64
                ),

            "compatible_donor_count":
                selected[
                    "external_compatible_donor_count"
                ].to_numpy(
                    dtype=np.int64
                ),

            "inherited_625_solution_donor_h5_path":
                inherited_donor_paths,

            "final_donor_index":
                final_donor_indices,

            "final_donor_h5_path":
                final_donor_paths,

            "final_donor_volume":
                final_donors[
                    "volume"
                ].to_numpy(
                    dtype=np.int64
                ),

            "final_donor_slice_index":
                final_donors[
                    "slice"
                ].to_numpy(
                    dtype=np.int64
                ),

            "final_donor_tumor_area_pixels":
                final_donors[
                    "tumor_area_pixels"
                ].to_numpy(
                    dtype=np.int64
                ),

            "final_donor_bbox_area":
                final_donors[
                    "bbox_area"
                ].to_numpy(),

            "final_donor_component_count":
                final_donors[
                    "connected_component_count"
                ].to_numpy(
                    dtype=np.int64
                ),

            "final_donor_largest_component_fraction":
                final_donors[
                    "largest_component_fraction"
                ].to_numpy(),

            "final_donor_centroid_x_normalized":
                final_donors[
                    "centroid_x_normalized"
                ].to_numpy(),

            "final_donor_centroid_y_normalized":
                final_donors[
                    "centroid_y_normalized"
                ].to_numpy(),

            "final_donor_centroid_laterality":
                final_donors[
                    "centroid_laterality"
                ].to_numpy(),

            "changed_from_inherited_625_solution":
                changed_from_inherited,
        }
    )

    assignment_table = pd.concat(
        [
            assignment_table,
            matching_diagnostics[
                [
                    "final_preference_rank",
                    "augmenting_path_length",
                    "required_rerouting",
                ]
            ].reset_index(
                drop=True
            ),
        ],
        axis=1,
    )

    if assignment_table[
        "final_donor_h5_path"
    ].duplicated().any():
        raise FinalManifestError(
            "Final assignment table contains donor reuse."
        )

    # Explicitly verify that each selected final donor is an exact edge
    # of its corresponding compatibility graph.
    for base_index, donor_index_value in enumerate(
        final_donor_indices
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
            raise FinalManifestError(
                "A final donor assignment is outside the exact "
                "compatibility graph."
            )

    final_manifest = pd.DataFrame(
        {
            "case_id":
                assignment_table[
                    "case_id"
                ],

            "external_subject_numeric_id":
                assignment_table[
                    "external_subject_numeric_id"
                ],

            "external_slice_index":
                assignment_table[
                    "external_slice_index"
                ],

            "external_modality":
                assignment_table[
                    "external_modality"
                ],

            "donor_h5_path":
                assignment_table[
                    "final_donor_h5_path"
                ],
        }
    )

    final_manifest.to_csv(
        manifest_path,
        index=False,
    )

    assignment_table.to_csv(
        assignments_path,
        index=False,
    )

    # Validate against the exact evaluator manifest contract.
    parsed_cases = (
        load_external_evaluation_manifest(
            manifest_path
        )
    )

    if len(
        parsed_cases
    ) != FINAL_CASE_COUNT:
        raise FinalManifestError(
            "Evaluator manifest parser did not return exactly "
            f"{FINAL_CASE_COUNT} cases."
        )

    total_edges = int(
        sum(
            compatible.size
            for compatible in compatibility_lists
        )
    )

    changed_count = int(
        changed_from_inherited.sum()
    )

    unique_training_volumes = int(
        assignment_table[
            "final_donor_volume"
        ].nunique()
    )

    volume_counts = (
        assignment_table[
            "final_donor_volume"
        ]
        .value_counts()
    )

    summary = {
        "manifest_role":
            (
                "Definitive 250-case external BR-LoRA evaluation manifest."
            ),

        "design_decision": {
            "selected_case_count":
                FINAL_CASE_COUNT,

            "validation_subject_count":
                EXPECTED_SUBJECT_COUNT,

            "bases_per_validation_subject":
                FINAL_BASES_PER_SUBJECT,

            "base_selection":
                (
                    "Exactly the rank-1 and rank-2 external bases from "
                    "the completed nested candidate-design audit."
                ),

            "donor_assignment":
                (
                    "Fresh deterministic global one-to-one matching "
                    "constructed using only the selected 250 external bases."
                ),

            "inherited_625_case_donor_assignments_used":
                False,

            "reason_for_rematching":
                (
                    "Donor assignments inherited from the 625-case design "
                    "could have been rerouted to accommodate rank-3 through "
                    "rank-5 bases that are absent from the definitive "
                    "250-case experiment."
                ),
        },

        "matching": {
            "compatibility_edge_count":
                total_edges,

            "unique_donor_slice_count":
                int(
                    assignment_table[
                        "final_donor_h5_path"
                    ].nunique()
                ),

            "unique_training_volume_count":
                unique_training_volumes,

            "maximum_cases_from_one_training_volume":
                int(
                    volume_counts.max()
                ),

            "median_cases_per_used_training_volume":
                float(
                    volume_counts.median()
                ),

            "bases_requiring_rerouting":
                int(
                    assignment_table[
                        "required_rerouting"
                    ].sum()
                ),

            "bases_requiring_rerouting_fraction":
                float(
                    assignment_table[
                        "required_rerouting"
                    ].mean()
                ),

            "first_preference_count":
                int(
                    (
                        assignment_table[
                            "final_preference_rank"
                        ]
                        == 1
                    ).sum()
                ),

            "first_preference_fraction":
                float(
                    (
                        assignment_table[
                            "final_preference_rank"
                        ]
                        == 1
                    ).mean()
                ),

            "maximum_final_preference_rank":
                int(
                    assignment_table[
                        "final_preference_rank"
                    ].max()
                ),

            "maximum_augmenting_path_length":
                int(
                    assignment_table[
                        "augmenting_path_length"
                    ].max()
                ),
        },

        "comparison_with_inherited_625_solution": {
            "assignments_changed_after_250_only_rematching":
                changed_count,

            "assignments_unchanged_after_250_only_rematching":
                int(
                    FINAL_CASE_COUNT
                    - changed_count
                ),

            "changed_fraction":
                float(
                    changed_count
                    / FINAL_CASE_COUNT
                ),
        },

        "established_eligibility_rules": {
            "external_base":
                (
                    "tumor_free_candidate == True, "
                    "predicted_tumor_pixels == 0, "
                    "compatible_donor_count > 0"
                ),

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

            "donor":
                (
                    "whole_tumor_pixels >= 300 and "
                    "established mask-margin rule already passed"
                ),
        },

        "output_artifacts": {
            "definitive_evaluation_manifest":
                str(
                    manifest_path
                ),

            "final_assignment_audit":
                str(
                    assignments_path
                ),
        },

        "source_artifacts": {
            "validation_dataset":
                str(
                    validation_path
                ),

            "candidate_assignments":
                str(
                    candidate_path
                ),

            "donor_morphology_csv":
                str(
                    donor_path
                ),

            "pair_space_summary":
                str(
                    pair_summary_path
                ),
        },

        "provenance": {
            "finalized_at_utc":
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

            "candidate_assignments_sha256":
                sha256_file(
                    candidate_path
                ),

            "donor_morphology_csv_sha256":
                sha256_file(
                    donor_path
                ),

            "pair_space_summary_sha256":
                sha256_file(
                    pair_summary_path
                ),

            "base_selection_seed":
                int(
                    args.seed
                ),

            "donor_assignment_seed":
                donor_assignment_seed,

            "device":
                str(
                    device
                ),
        },

        "status": {
            "definitive_manifest_selected":
                True,

            "definitive_manifest_frozen":
                True,

            "evaluator_contract_validation_passed":
                True,

            "br_lora_external_evaluation_run":
                False,
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
        "DEFINITIVE 250-CASE EXTERNAL MANIFEST: PASS"
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
        f"{EXPECTED_SUBJECT_COUNT}"
    )
    print(
        f"Bases per subject        : "
        f"{FINAL_BASES_PER_SUBJECT}"
    )
    print(
        f"Unique donor slices      : "
        f"{FINAL_CASE_COUNT}"
    )
    print(
        f"Training volumes used    : "
        f"{unique_training_volumes}"
    )
    print(
        f"250-only assignments changed vs inherited 625 solution: "
        f"{changed_count}"
    )
    print(
        f"Evaluator contract       : PASS"
    )
    print()
    print(
        f"Definitive manifest      : "
        f"{manifest_path}"
    )
    print(
        f"Assignment audit         : "
        f"{assignments_path}"
    )
    print(
        f"Summary                  : "
        f"{summary_path}"
    )
    print(
        "=" * 78
    )


if __name__ == "__main__":
    main()
