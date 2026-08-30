#!/usr/bin/env python3
"""
Audit internal behavior of the global external base-donor matching algorithm.

This script reconstructs the same 625-base compatibility graph used by the
candidate manifest design audit and reruns the same deterministic one-to-one
matching procedure while recording diagnostics.

It does not modify candidate assignments or freeze a definitive manifest.

Primary diagnostics
-------------------
- number of selected bases and graph edges,
- compatibility-degree distribution,
- number of augmenting-path attempts,
- number of assignments requiring rerouting,
- augmenting-path length distribution,
- maximum augmenting-path length,
- donor preference rank of final assignments,
- proportion assigned to first preference,
- proportion assigned within top 5 / top 10 preferences,
- exact donor uniqueness and compatibility verification.
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

from screening.brats_nnunet.scripts.audit_external_manifest_design import (
    build_matching_candidate_order,
    compute_compatible_donor_indices_for_batch,
    load_and_pack_donor_masks,
    load_selected_base_brain_masks,
    resolve_device,
)


OUTPUT_BASE_DIAGNOSTICS = (
    "matching_base_diagnostics.csv"
)

OUTPUT_SUMMARY = (
    "matching_diagnostics_summary.json"
)


class MatchingDiagnosticsError(
    RuntimeError
):
    """Raised when matching diagnostics fail."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit augmenting-path behavior of the external "
            "base-donor matching algorithm."
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
            "Candidate assignment table. If omitted, uses "
            "<nnunet_run_root>/external_manifest_design_audit/"
            "candidate_assignments_all.csv."
        ),
    )

    parser.add_argument(
        "--donor-morphology-csv",
        type=Path,
        default=None,
        help=(
            "Donor morphology table. If omitted, uses "
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
            "Directory for matching diagnostics. If omitted, uses "
            "<nnunet_run_root>/external_matching_diagnostics."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
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

    with path.open("rb") as file:
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

    return result.stdout.strip() == ""


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


def summarize_numeric(
    values: np.ndarray,
) -> dict[str, Any]:
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    return {
        "count":
            int(values.size),

        "minimum":
            float(values.min()),

        "q1":
            float(np.quantile(
                values,
                0.25,
            )),

        "median":
            float(np.median(
                values
            )),

        "mean":
            float(values.mean()),

        "q3":
            float(np.quantile(
                values,
                0.75,
            )),

        "maximum":
            float(values.max()),
    }


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
    list[np.ndarray],
]:
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
                selected["case_id"],

            "base_rank":
                selected["base_rank"],

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
        "Reconstructing complete 625-base compatibility graph..."
    )

    for base_start in range(
        0,
        len(base_table),
        base_batch_size,
    ):
        base_end = min(
            base_start
            + base_batch_size,
            len(base_table),
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
                raise MatchingDiagnosticsError(
                    "Recomputed compatibility count disagrees "
                    "with candidate assignment provenance.\n"
                    f"Case: "
                    f"{base_table.iloc[global_index]['case_id']}\n"
                    f"Expected: {expected:,}\n"
                    f"Observed: {observed:,}"
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

    return (
        selected,
        compatibility_lists,
    )


def run_matching_with_diagnostics(
    *,
    compatibility_lists: list[np.ndarray],
    donors: pd.DataFrame,
    seed: int,
) -> tuple[
    np.ndarray,
    pd.DataFrame,
    dict[str, Any],
]:
    """
    Reproduce the global matching while recording augmenting-path behavior.
    """

    base_count = len(
        compatibility_lists
    )

    donor_count = len(
        donors
    )

    rng = np.random.default_rng(
        seed
        + 1
    )

    ordered_candidates: list[
        np.ndarray
    ] = []

    preference_rank_maps: list[
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

        ordered_candidates.append(
            ordered
        )

        preference_rank_maps.append(
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
        ] = ordered.size

    tie_break = rng.random(
        base_count
    )

    base_order = sorted(
        range(base_count),
        key=lambda index: (
            int(degrees[index]),
            float(tie_break[index]),
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

    reroute_count_per_base = np.zeros(
        base_count,
        dtype=np.int64,
    )

    max_path_length_per_base = np.zeros(
        base_count,
        dtype=np.int64,
    )

    assignment_attempt_count = np.zeros(
        base_count,
        dtype=np.int64,
    )

    global_augmenting_path_lengths: list[int] = []

    global_recursive_reassignment_count = 0

    def try_assign(
        base_index: int,
        seen_donors: np.ndarray,
        *,
        depth: int,
        root_base: int,
    ) -> tuple[
        bool,
        int,
    ]:
        nonlocal global_recursive_reassignment_count

        assignment_attempt_count[
            root_base
        ] += 1

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

            current_base = int(
                donor_to_base[
                    donor_index
                ]
            )

            if current_base == -1:
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

            global_recursive_reassignment_count += 1

            success, path_length = try_assign(
                current_base,
                seen_donors,
                depth=depth + 1,
                root_base=root_base,
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
        "Running instrumented global matching..."
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
            root_base=base_index,
        )

        if not success:
            raise MatchingDiagnosticsError(
                "Instrumented matching failed to produce "
                "a complete one-to-one assignment."
            )

        global_augmenting_path_lengths.append(
            int(path_length)
        )

        max_path_length_per_base[
            base_index
        ] = int(
            path_length
        )

        if path_length > 1:
            reroute_count_per_base[
                base_index
            ] = 1

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
        raise MatchingDiagnosticsError(
            "Instrumented matching left an unmatched base."
        )

    if np.unique(
        base_to_donor
    ).size != base_count:
        raise MatchingDiagnosticsError(
            "Instrumented matching reused donor slices."
        )

    final_preference_rank = np.zeros(
        base_count,
        dtype=np.int64,
    )

    for base_index, donor_index_value in enumerate(
        base_to_donor
    ):
        donor_index = int(
            donor_index_value
        )

        if donor_index not in preference_rank_maps[
            base_index
        ]:
            raise MatchingDiagnosticsError(
                "Final matched donor is outside the "
                "base preference ordering."
            )

        final_preference_rank[
            base_index
        ] = preference_rank_maps[
            base_index
        ][
            donor_index
        ]

    base_diagnostics = pd.DataFrame(
        {
            "base_index":
                np.arange(
                    base_count,
                    dtype=np.int64,
                ),

            "compatibility_degree":
                degrees,

            "final_donor_index":
                base_to_donor,

            "final_preference_rank":
                final_preference_rank,

            "root_assignment_required_reroute":
                reroute_count_per_base.astype(
                    bool
                ),

            "augmenting_path_length":
                max_path_length_per_base,

            "recursive_assignment_calls":
                assignment_attempt_count,
        }
    )

    path_lengths = np.asarray(
        global_augmenting_path_lengths,
        dtype=np.int64,
    )

    summary = {
        "selected_base_count":
            base_count,

        "donor_count":
            donor_count,

        "compatibility_edge_count":
            int(
                sum(
                    compatible.size
                    for compatible in compatibility_lists
                )
            ),

        "compatibility_degree":
            summarize_numeric(
                degrees
            ),

        "augmenting_path_length":
            summarize_numeric(
                path_lengths
            ),

        "bases_requiring_rerouting":
            int(
                (
                    path_lengths
                    > 1
                ).sum()
            ),

        "bases_requiring_rerouting_fraction":
            float(
                (
                    path_lengths
                    > 1
                ).mean()
            ),

        "bases_assigned_first_preference":
            int(
                (
                    final_preference_rank
                    == 1
                ).sum()
            ),

        "bases_assigned_first_preference_fraction":
            float(
                (
                    final_preference_rank
                    == 1
                ).mean()
            ),

        "bases_assigned_top_5_preference":
            int(
                (
                    final_preference_rank
                    <= 5
                ).sum()
            ),

        "bases_assigned_top_5_preference_fraction":
            float(
                (
                    final_preference_rank
                    <= 5
                ).mean()
            ),

        "bases_assigned_top_10_preference":
            int(
                (
                    final_preference_rank
                    <= 10
                ).sum()
            ),

        "bases_assigned_top_10_preference_fraction":
            float(
                (
                    final_preference_rank
                    <= 10
                ).mean()
            ),

        "final_preference_rank":
            summarize_numeric(
                final_preference_rank
            ),

        "recursive_reassignment_events":
            int(
                global_recursive_reassignment_count
            ),

        "unique_donor_count":
            int(
                np.unique(
                    base_to_donor
                ).size
            ),
    }

    return (
        base_to_donor,
        base_diagnostics,
        summary,
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
            / "external_matching_diagnostics"
        )

    save_folders_config(
        args.folders_file,
        folders_config,
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    diagnostics_path = (
        output_dir
        / OUTPUT_BASE_DIAGNOSTICS
    )

    summary_path = (
        output_dir
        / OUTPUT_SUMMARY
    )

    existing = [
        path
        for path in (
            diagnostics_path,
            summary_path,
        )
        if path.exists()
    ]

    if existing:
        raise MatchingDiagnosticsError(
            "Refusing to overwrite existing output(s):\n"
            + "\n".join(
                str(path)
                for path in existing
            )
        )

    assignments = pd.read_csv(
        assignments_path
    )

    donors = pd.read_csv(
        donors_path
    )

    if len(assignments) != 625:
        raise MatchingDiagnosticsError(
            "Expected exactly 625 candidate assignments."
        )

    if assignments[
        "donor_h5_path"
    ].duplicated().any():
        raise MatchingDiagnosticsError(
            "Candidate assignments already contain donor reuse."
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
        raise MatchingDiagnosticsError(
            "Brain threshold disagrees with pair-space audit."
        )

    if not np.isclose(
        args.min_overlap,
        expected_overlap,
        rtol=0.0,
        atol=1e-12,
    ):
        raise MatchingDiagnosticsError(
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
    print("=" * 78)
    print("EXTERNAL MATCHING DIAGNOSTICS")
    print("=" * 78)
    print(
        f"Selected bases           : "
        f"{len(assignments):,}"
    )
    print(
        f"Eligible donor slices    : "
        f"{len(donors):,}"
    )
    print(
        f"Seed                     : "
        f"{args.seed}"
    )
    print(
        f"Device                   : "
        f"{device}"
    )
    print("=" * 78)

    packed_donor_masks = (
        load_and_pack_donor_masks(
            donors
        )
    )

    (
        ordered_assignments,
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

    (
        matched_donors,
        diagnostics,
        matching_summary,
    ) = run_matching_with_diagnostics(
        compatibility_lists=compatibility_lists,
        donors=donors,
        seed=args.seed,
    )

    diagnostics.insert(
        1,
        "case_id",
        ordered_assignments[
            "case_id"
        ].to_numpy(),
    )

    diagnostics.insert(
        2,
        "base_rank",
        ordered_assignments[
            "base_rank"
        ].to_numpy(
            dtype=np.int64
        ),
    )

    diagnostics.insert(
        3,
        "external_subject_numeric_id",
        ordered_assignments[
            "external_subject_numeric_id"
        ].to_numpy(
            dtype=np.int64
        ),
    )

    diagnostics.insert(
        4,
        "external_slice_index",
        ordered_assignments[
            "external_slice_index"
        ].to_numpy(
            dtype=np.int64
        ),
    )

    diagnostics[
        "instrumented_donor_h5_path"
    ] = donors.iloc[
        matched_donors
    ][
        "donor_h5_path"
    ].to_numpy()

    diagnostics[
        "recorded_candidate_donor_h5_path"
    ] = ordered_assignments[
        "donor_h5_path"
    ].to_numpy()

    exact_reproduction = bool(
        np.array_equal(
            diagnostics[
                "instrumented_donor_h5_path"
            ].to_numpy(),
            diagnostics[
                "recorded_candidate_donor_h5_path"
            ].to_numpy(),
        )
    )

    if not exact_reproduction:
        raise MatchingDiagnosticsError(
            "Instrumented matching did not reproduce the "
            "recorded candidate assignments exactly."
        )

    diagnostics.to_csv(
        diagnostics_path,
        index=False,
    )

    summary = {
        "audit_role":
            (
                "Diagnostic replay of the deterministic global "
                "one-to-one external base-donor matching algorithm."
            ),

        "matching_reproduced_recorded_assignments_exactly":
            exact_reproduction,

        "matching":
            matching_summary,

        "source_artifacts": {
            "validation_dataset":
                str(validation_path),

            "candidate_assignments":
                str(assignments_path),

            "donor_morphology_csv":
                str(donors_path),

            "pair_space_summary":
                str(pair_summary_path),
        },

        "output_artifacts": {
            "base_diagnostics":
                str(diagnostics_path),
        },

        "selection_status": {
            "assignments_modified":
                False,

            "definitive_manifest_frozen":
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
                resolve_git_commit(),

            "git_worktree_clean":
                resolve_git_worktree_clean(),

            "script_path":
                str(
                    Path(__file__).resolve()
                ),

            "script_sha256":
                sha256_file(
                    Path(__file__).resolve()
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

            "seed":
                args.seed,

            "device":
                str(device),
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

        file.write("\n")

    print()
    print("=" * 78)
    print("EXTERNAL MATCHING DIAGNOSTICS: PASS")
    print("=" * 78)

    print(
        "Recorded assignments exact:",
        exact_reproduction,
    )

    print(
        "Compatibility edges       :",
        f"{matching_summary['compatibility_edge_count']:,}",
    )

    print(
        "Bases requiring rerouting :",
        f"{matching_summary['bases_requiring_rerouting']:,}",
    )

    print(
        "Rerouting fraction        :",
        f"{matching_summary['bases_requiring_rerouting_fraction']:.4f}",
    )

    print(
        "First-preference donors   :",
        f"{matching_summary['bases_assigned_first_preference']:,}",
    )

    print(
        "First-preference fraction :",
        f"{matching_summary['bases_assigned_first_preference_fraction']:.4f}",
    )

    print(
        "Top-5 preference fraction :",
        f"{matching_summary['bases_assigned_top_5_preference_fraction']:.4f}",
    )

    print(
        "Top-10 preference fraction:",
        f"{matching_summary['bases_assigned_top_10_preference_fraction']:.4f}",
    )

    print(
        "Maximum augmenting path   :",
        int(
            matching_summary[
                "augmenting_path_length"
            ][
                "maximum"
            ]
        ),
    )

    print(
        "Unique donor slices       :",
        f"{matching_summary['unique_donor_count']:,}",
    )

    print()
    print(
        f"Base diagnostics          : "
        f"{diagnostics_path}"
    )

    print(
        f"Summary                   : "
        f"{summary_path}"
    )

    print()
    print(
        "No assignments were modified."
    )

    print("=" * 78)


if __name__ == "__main__":
    main()
