#!/usr/bin/env python3
"""
Design the frozen 10,000-case BR-LoRA synthetic library.

This script performs DESIGN ONLY. It does not synthesize images.

The design extends the already-generated 250-case BR-LoRA cohort to a
10,000-case synthetic library while preserving the exact regional-composition
compatibility definition used by audit_external_pair_space.py.

Frozen design rules
-------------------
1. Final library size: 10,000 cases.
2. Exactly 80 cases per each of 125 external BraTS validation subjects.
3. Existing Batch 0001 cases (synthetic_000001--synthetic_000250) are locked.
4. Future donors may come only from the frozen downstream-training split.
5. Every donor slice is used at most once across the complete 10,000 cases.
6. No donor training volume contributes more than 31 total cases.
7. Base slices are reused only as needed and are balanced within subject.
8. Every selected pair must satisfy the original compatibility rule:
       overlap_pixels >= 0.80 * donor_pixels
   using get_brain_mask on the external base and the whole-tumor donor mask.
9. Selection is deterministic given the configured random seed.

Important
---------
The historical pair-space audit included all 369 BraTS training subjects.
Because 37 subjects are now reserved for downstream segmentation validation,
this script recomputes compatibility using only the 332 downstream-training
subjects before selecting new cases.

No Cartesian 69-million-row edge table is materialized.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch

import scipy
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import maximum_flow

from src.config import (
    load_folders_config,
    resolve_path,
    save_folders_config,
)
from src.data import (
    get_brain_mask,
    load_h5_full,
    load_validation_dataset_specification,
    load_validation_slice,
)


TARGET_LIBRARY_SIZE = 10_000
TARGET_PER_EXTERNAL_SUBJECT = 80
EXISTING_CASE_COUNT = 250
NEW_CASE_COUNT = TARGET_LIBRARY_SIZE - EXISTING_CASE_COUNT

MIN_OVERLAP = 0.80
BRAIN_THRESHOLD = 0.05
DONOR_SUBJECT_CAP = 31

# A base enters the frozen 10,000-case design only when at least
# this many still-unused compatible donor slices are available.
# The threshold equals the maximum subject-level base-reuse
# ceiling required by the balanced 80-cases-per-subject design.
MIN_ACTIVE_BASE_DONOR_SUPPORT = 10

# Deterministic candidate thinning used by the validated exact
# sparse max-flow assignment. The 32-candidate graph achieved
# complete flow (9,750 / 9,750) under all frozen constraints.
EXACT_FLOW_CANDIDATE_LIMIT = 32

DEFAULT_SEED = 2026
DEFAULT_BASE_BATCH_SIZE = 128
DEFAULT_DONOR_BATCH_SIZE = 128

DESIGN_NAME = "br_lora_library_design_10000.csv"
AUDIT_NAME = "br_lora_library_design_10000_audit.csv"
SUMMARY_NAME = "br_lora_library_design_10000_summary.json"

DEFAULT_TRAINING_DONOR_POOL = (
    PROJECT_ROOT
    / "downstream_evaluation"
    / "manifests"
    / "brats_downstream_training_donor_pool.csv"
)

DEFAULT_BATCH0001_MANIFEST = (
    PROJECT_ROOT
    / "downstream_evaluation"
    / "manifests"
    / "br_lora_synthetic_250.csv"
)


class LibraryDesignError(RuntimeError):
    """Raised when frozen BR-LoRA library construction fails."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Construct the frozen 10,000-case BR-LoRA library design."
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
            "Registered BraTS 2020 validation_dataset.yaml. Overrides "
            "yaml_validation_dataset_path in --folders-file."
        ),
    )

    parser.add_argument(
        "--base-counts-csv",
        type=Path,
        default=None,
        help=(
            "external_base_compatibility_counts.csv. If omitted, uses "
            "<nnunet_run_root>/external_pair_space_audit/"
            "external_base_compatibility_counts.csv."
        ),
    )

    parser.add_argument(
        "--training-donor-pool",
        type=Path,
        default=None,
        help=(
            "Training-only donor-pool CSV after downstream split. "
            "If omitted, uses the tracked repository donor pool."
        ),
    )

    parser.add_argument(
        "--h5-root",
        type=Path,
        default=None,
        help=(
            "BraTS reconstructed H5 directory. Overrides h5_root "
            "in --folders-file."
        ),
    )

    parser.add_argument(
        "--batch0001-manifest",
        type=Path,
        default=None,
        help=(
            "Portable frozen 250-case synthetic manifest. If omitted, "
            "uses the tracked repository manifest."
        ),
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory for frozen design artifacts.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
    )

    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "mps", "cuda"],
        default="auto",
    )

    parser.add_argument(
        "--base-batch-size",
        type=int,
        default=DEFAULT_BASE_BATCH_SIZE,
    )

    parser.add_argument(
        "--donor-batch-size",
        type=int,
        default=DEFAULT_DONOR_BATCH_SIZE,
    )

    return parser.parse_args()


def resolve_file(path: Path, name: str) -> Path:
    path = path.expanduser().resolve()

    if not path.is_file():
        raise FileNotFoundError(
            f"{name} does not exist:\n{path}"
        )

    return path


def resolve_directory(path: Path, name: str) -> Path:
    path = path.expanduser().resolve()

    if not path.is_dir():
        raise NotADirectoryError(
            f"{name} does not exist or is not a directory:\n{path}"
        )

    return path


def resolve_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")

    if requested == "cuda":
        if not torch.cuda.is_available():
            raise LibraryDesignError(
                "CUDA was requested but is unavailable."
            )
        return torch.device("cuda")

    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise LibraryDesignError(
                "MPS was requested but is unavailable."
            )
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None

    value = result.stdout.strip()
    return value or None


def git_clean() -> bool | None:
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None

    return result.stdout.strip() == ""


def require_columns(
    table: pd.DataFrame,
    required: set[str],
    name: str,
) -> None:
    missing = sorted(required - set(table.columns))

    if missing:
        raise LibraryDesignError(
            f"{name} is missing required columns:\n"
            + "\n".join(missing)
        )


def load_existing_batch(
    path: Path,
) -> pd.DataFrame:
    table = pd.read_csv(path)

    require_columns(
        table,
        {
            "case_id",
            "external_subject_name",
            "external_slice_index",
            "donor_h5_file",
            "mask_pixels",
        },
        "Batch 0001 manifest",
    )

    if len(table) != EXISTING_CASE_COUNT:
        raise LibraryDesignError(
            "Batch 0001 must contain exactly "
            f"{EXISTING_CASE_COUNT} cases; observed {len(table)}."
        )

    subject_match = table[
        "external_subject_name"
    ].str.extract(
        r"BraTS20_Validation_(\d+)",
        expand=False,
    )

    if subject_match.isna().any():
        raise LibraryDesignError(
            "Batch 0001 contains an unexpected external subject name."
        )

    table["external_subject_numeric_id"] = (
        subject_match.astype(int)
    )

    donor_parse = table[
        "donor_h5_file"
    ].str.extract(
        r"volume_(\d+)_slice_(\d+)\.h5",
        expand=True,
    )

    if donor_parse.isna().any().any():
        raise LibraryDesignError(
            "Batch 0001 contains an unexpected donor H5 filename."
        )

    table["donor_volume"] = donor_parse[0].astype(int)
    table["donor_slice_index"] = donor_parse[1].astype(int)

    per_subject = (
        table.groupby("external_subject_numeric_id")
        .size()
    )

    if len(per_subject) != 125:
        raise LibraryDesignError(
            "Batch 0001 does not contain all 125 external subjects."
        )

    if not (per_subject == 2).all():
        raise LibraryDesignError(
            "Batch 0001 must contain exactly two cases per external subject."
        )

    if table["donor_h5_file"].duplicated().any():
        raise LibraryDesignError(
            "Batch 0001 unexpectedly reuses a donor slice."
        )

    return table.reset_index(drop=True)


def load_candidate_bases(
    path: Path,
) -> pd.DataFrame:
    bases = pd.read_csv(path)

    require_columns(
        bases,
        {
            "subject",
            "subject_numeric_id",
            "slice_index",
            "predicted_tumor_pixels",
            "tumor_free_candidate",
            "reference_modality",
            "brain_pixels",
            "compatible_donor_count",
        },
        "External-base compatibility table",
    )

    # Reproduce the previously eligible base population.
    bases = bases[
        bases["tumor_free_candidate"].astype(bool)
        & (bases["predicted_tumor_pixels"] == 0)
        & (bases["compatible_donor_count"] > 0)
    ].copy()

    bases = bases.sort_values(
        ["subject_numeric_id", "slice_index"]
    ).reset_index(drop=True)

    if len(bases) != 8632:
        raise LibraryDesignError(
            "Expected 8,632 previously compatible external bases; "
            f"observed {len(bases):,}."
        )

    if bases["subject_numeric_id"].nunique() != 125:
        raise LibraryDesignError(
            "Previously eligible base pool does not contain 125 subjects."
        )

    return bases


def load_training_donors(
    path: Path,
    h5_root: Path,
) -> pd.DataFrame:
    donors = pd.read_csv(path)

    require_columns(
        donors,
        {
            "volume",
            "slice",
            "donor_h5_path",
            "loaded_mask_pixels",
            "tumor_area_pixels",
            "bbox_area",
            "connected_component_count",
            "largest_component_fraction",
            "centroid_x_normalized",
            "centroid_y_normalized",
            "centroid_laterality",
        },
        "Training-only donor pool",
    )

    if len(donors) != 17935:
        raise LibraryDesignError(
            "Expected 17,935 downstream-training donor slices; "
            f"observed {len(donors):,}."
        )

    if donors["volume"].nunique() != 332:
        raise LibraryDesignError(
            "Expected 332 downstream-training donor subjects."
        )

    donors = donors.copy()

    donors["volume"] = donors["volume"].astype(int)
    donors["slice"] = donors["slice"].astype(int)

    # Prefer portable basename regardless of historical absolute path.
    donors["donor_h5_file"] = donors[
        "donor_h5_path"
    ].map(
        lambda x: Path(str(x)).name
    )

    donors["resolved_h5_path"] = donors[
        "donor_h5_file"
    ].map(
        lambda x: str(h5_root / x)
    )

    missing = [
        p
        for p in donors["resolved_h5_path"]
        if not Path(p).is_file()
    ]

    if missing:
        raise FileNotFoundError(
            "Training-only donor pool references missing H5 files. "
            f"First missing file:\n{missing[0]}"
        )

    if donors["donor_h5_file"].duplicated().any():
        raise LibraryDesignError(
            "Training donor pool contains duplicate donor slices."
        )

    return donors.reset_index(drop=True)


def load_base_masks(
    bases: pd.DataFrame,
    validation_dataset_path: Path,
) -> np.ndarray:
    dataset = load_validation_dataset_specification(
        validation_dataset_path
    )

    masks = np.empty(
        (len(bases), 240 * 240),
        dtype=np.uint8,
    )

    print()
    print("Loading external-base brain masks...")

    for row_number, row in enumerate(
        bases.itertuples(index=False),
        start=0,
    ):
        loaded = load_validation_slice(
            dataset,
            subject_numeric_id=int(row.subject_numeric_id),
            slice_index=int(row.slice_index),
            modality=str(row.reference_modality),
        )
        
        image = (
            loaded.image[
                0
            ]
            .detach()
            .cpu()
        )

        brain = get_brain_mask(
            image,
            threshold=BRAIN_THRESHOLD,
        )

        masks[row_number] = (
            brain
            .detach()
            .cpu()
            .numpy()
            .astype(
                np.uint8,
                copy=False,
            )
            .reshape(-1)
        )

        if (
            (row_number + 1) % 500 == 0
            or row_number + 1 == len(bases)
        ):
            print(
                f"  {row_number + 1:,} / {len(bases):,} bases",
                flush=True,
            )

    return masks


def load_donor_masks(
    donors: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    masks = np.empty(
        (len(donors), 240 * 240),
        dtype=np.uint8,
    )

    counts = np.empty(
        len(donors),
        dtype=np.int64,
    )

    print()
    print("Loading downstream-training donor masks...")

    for donor_index, row in enumerate(
        donors.itertuples(index=False),
        start=0,
    ):
        _, mask, _ = load_h5_full(
            row.resolved_h5_path,
            image_channel=0,
        )

        arr = (
            mask[
                0
            ]
            .detach()
            .cpu()
            .numpy()
            .astype(np.uint8)
        )

        loaded_count = int(arr.sum())

        expected_count = int(row.loaded_mask_pixels)

        if loaded_count != expected_count:
            raise LibraryDesignError(
                "Loaded donor-mask pixel count disagrees with frozen "
                "donor table.\n"
                f"Donor: {row.donor_h5_file}\n"
                f"Loaded: {loaded_count}\n"
                f"Expected: {expected_count}"
            )

        masks[donor_index] = arr.reshape(-1)
        counts[donor_index] = loaded_count

        if (
            (donor_index + 1) % 1000 == 0
            or donor_index + 1 == len(donors)
        ):
            print(
                f"  {donor_index + 1:,} / {len(donors):,} donors",
                flush=True,
            )

    return masks, counts


def recompute_training_only_compatibility(
    *,
    base_masks: np.ndarray,
    donor_masks: np.ndarray,
    donor_pixel_counts: np.ndarray,
    base_batch_size: int,
    donor_batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """
    Compute exact compatibility using the historical matrix definition.

    Returns
    -------
    counts
        Number of training-only compatible donors per base.
    donor_indices_by_base
        Integer donor indices compatible with each base.

    Only the restricted 8,632 x 17,935 space is considered.
    """
    n_base = len(base_masks)
    n_donor = len(donor_masks)

    counts = np.zeros(
        n_base,
        dtype=np.int64,
    )

    compatible_lists: list[list[np.ndarray]] = [
        [] for _ in range(n_base)
    ]

    total_base_batches = math.ceil(
        n_base / base_batch_size
    )

    print()
    print("Recomputing compatibility against training-only donors...")
    print("Device:", device)
    print("Bases :", f"{n_base:,}")
    print("Donors:", f"{n_donor:,}")

    with torch.no_grad():
        for base_batch_number, base_start in enumerate(
            range(0, n_base, base_batch_size),
            start=1,
        ):
            base_end = min(
                base_start + base_batch_size,
                n_base,
            )

            base_tensor = torch.from_numpy(
                base_masks[base_start:base_end]
            ).to(
                device=device,
                dtype=torch.float32,
            )

            for donor_start in range(
                0,
                n_donor,
                donor_batch_size,
            ):
                donor_end = min(
                    donor_start + donor_batch_size,
                    n_donor,
                )

                donor_tensor = torch.from_numpy(
                    donor_masks[donor_start:donor_end]
                ).to(
                    device=device,
                    dtype=torch.float32,
                )

                overlap_pixels = torch.matmul(
                    base_tensor,
                    donor_tensor.transpose(0, 1),
                )

                threshold = torch.from_numpy(
                    donor_pixel_counts[
                        donor_start:donor_end
                    ].astype(np.float32)
                    * MIN_OVERLAP
                ).to(device=device)

                compatible = (
                    overlap_pixels
                    >= threshold[None, :]
                )

                compatible_cpu = (
                    compatible
                    .detach()
                    .cpu()
                    .numpy()
                )

                for local_base in range(
                    compatible_cpu.shape[0]
                ):
                    hits = np.flatnonzero(
                        compatible_cpu[local_base]
                    )

                    if hits.size:
                        global_base = base_start + local_base

                        compatible_lists[
                            global_base
                        ].append(
                            hits.astype(np.int32)
                            + donor_start
                        )

                del donor_tensor
                del overlap_pixels
                del compatible

            del base_tensor

            print(
                f"  Base batch {base_batch_number:,} / "
                f"{total_base_batches:,} complete",
                flush=True,
            )

    donor_indices_by_base: list[np.ndarray] = []

    for base_index, chunks in enumerate(
        compatible_lists
    ):
        if chunks:
            merged = np.concatenate(chunks)
        else:
            merged = np.empty(
                0,
                dtype=np.int32,
            )

        donor_indices_by_base.append(merged)
        counts[base_index] = len(merged)

    return counts, donor_indices_by_base


def build_base_slots(
    *,
    bases: pd.DataFrame,
    training_counts: np.ndarray,
    donor_indices_by_base: list[np.ndarray],
    donors: pd.DataFrame,
    batch1: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    bases = bases.copy()

    bases[
        "training_compatible_donor_count"
    ] = training_counts

    existing_donor_files = set(
        batch1["donor_h5_file"].astype(str)
    )

    donor_files = (
        donors["donor_h5_file"]
        .astype(str)
        .to_numpy()
    )

    available_new_counts = np.zeros(
        len(bases),
        dtype=np.int64,
    )

    for base_index, candidate_indices in enumerate(
        donor_indices_by_base
    ):
        available_new_counts[base_index] = sum(
            donor_files[int(donor_index)]
            not in existing_donor_files
            for donor_index in candidate_indices
        )

    bases[
        "available_new_unique_donor_count"
    ] = available_new_counts

    usable = bases[
        bases["available_new_unique_donor_count"]
        >= MIN_ACTIVE_BASE_DONOR_SUPPORT
    ].copy()

    eligible_per_subject = (
        usable.groupby("subject_numeric_id")
        .size()
    )

    missing_subjects = sorted(
        set(
            bases["subject_numeric_id"]
            .astype(int)
            .unique()
        )
        - set(
            eligible_per_subject.index.astype(int)
        )
    )

    if missing_subjects:
        raise LibraryDesignError(
            "Minimum active-base donor support removed all "
            "bases for one or more external subjects:\n"
            + ", ".join(
                str(x)
                for x in missing_subjects
            )
        )

    print(
        "Minimum active-base donor support:",
        MIN_ACTIVE_BASE_DONOR_SUPPORT,
    )

    per_subject_available = (
        usable.groupby("subject_numeric_id")
        .size()
    )

    missing_subjects = sorted(
        set(range(1, 126))
        - set(per_subject_available.index)
    )

    if missing_subjects:
        raise LibraryDesignError(
            "At least one external subject has no training-only "
            "compatible base:\n"
            + str(missing_subjects)
        )

    records = []

    # Existing base usage contributes to final reuse balancing.
    existing_usage = Counter(
        (
            int(row.external_subject_numeric_id),
            int(row.external_slice_index),
        )
        for row in batch1.itertuples(index=False)
    )

    for subject_id in sorted(
        usable["subject_numeric_id"].unique()
    ):
        subject_bases = usable[
            usable["subject_numeric_id"] == subject_id
        ].copy()

        existing_subject = batch1[
            batch1["external_subject_numeric_id"] == subject_id
        ]

        needed = (
            TARGET_PER_EXTERNAL_SUBJECT
            - len(existing_subject)
        )

        # ------------------------------------------------------------
        # Determine the capacity-feasible active base set.
        #
        # A base's final capacity is its already-realized Batch-0001
        # use plus the number of still-unused compatible donor slices.
        #
        # We iteratively remove bases that cannot support the reuse
        # ceiling implied by the current number of active bases.
        # ------------------------------------------------------------

        subject_bases["existing_base_uses"] = [
            int(
                existing_usage[
                    (
                        int(subject_id),
                        int(slice_index),
                    )
                ]
            )
            for slice_index in subject_bases["slice_index"]
        ]

        subject_bases["final_base_capacity"] = (
            subject_bases["existing_base_uses"]
            + subject_bases[
                "available_new_unique_donor_count"
            ]
        )

        active = subject_bases.copy()

        while True:
            if len(active) == 0:
                raise LibraryDesignError(
                    "No capacity-feasible bases remain for "
                    f"external subject {subject_id}."
                )

            required_reuse_ceiling = math.ceil(
                TARGET_PER_EXTERNAL_SUBJECT
                / len(active)
            )

            retained = active[
                active["final_base_capacity"]
                >= required_reuse_ceiling
            ].copy()

            if len(retained) == len(active):
                break

            active = retained

        subject_bases = active

        # The retained bases must collectively have enough capacity
        # for all 80 final cases.
        if (
            subject_bases[
                "final_base_capacity"
            ].sum()
            < TARGET_PER_EXTERNAL_SUBJECT
        ):
            raise LibraryDesignError(
                "Capacity-feasible bases do not provide enough "
                "total final-case capacity.\n"
                f"Subject: {subject_id}\n"
                f"Required: {TARGET_PER_EXTERNAL_SUBJECT}\n"
                f"Available: "
                f"{int(subject_bases['final_base_capacity'].sum())}"
            )

        if needed != 78:
            raise LibraryDesignError(
                f"Expected 78 new cases for subject {subject_id}; "
                f"observed target {needed}."
            )

        # Use a deterministic random tie-break for otherwise identical
        # reuse levels.
        subject_bases["tie_break"] = rng.random(
            len(subject_bases)
        )

        usage = {
            int(row.slice_index):
                int(
                    existing_usage[
                        (
                            int(subject_id),
                            int(row.slice_index),
                        )
                    ]
                )
            for row in subject_bases.itertuples(index=False)
        }

        base_lookup = {
            int(row.slice_index): row
            for row in subject_bases.itertuples(index=False)
        }

        tie_lookup = {
            int(row.slice_index): float(row.tie_break)
            for row in subject_bases.itertuples(index=False)
        }

        for _ in range(needed):
            capacity = {
                int(row.slice_index):
                    int(
                        row.available_new_unique_donor_count
                    )
                for row in subject_bases.itertuples(index=False)
            }
            # Minimize base reuse first.
            # Among equally used bases, prefer more highly connected bases.
            # Fixed random tie-break avoids slice-number artifacts.
            feasible_slices = [
                slice_index
                for slice_index in usage
                if (
                    usage[slice_index]
                    - existing_usage[
                        (
                            int(subject_id),
                            int(slice_index),
                        )
                    ]
                )
                < capacity[slice_index]
            ]

            if not feasible_slices:
                raise LibraryDesignError(
                    "External subject does not have enough "
                    "distinct compatible donor capacity for "
                    f"{TARGET_PER_EXTERNAL_SUBJECT} cases.\n"
                    f"Subject: {subject_id}"
                )

            chosen_slice = min(
                feasible_slices,
                key=lambda slice_index: (
                    usage[slice_index],
                    -capacity[slice_index],
                    -int(
                        base_lookup[
                            slice_index
                        ].training_compatible_donor_count
                    ),
                    tie_lookup[slice_index],
                    slice_index,
                ),
            )

            row = base_lookup[chosen_slice]

            usage[chosen_slice] += 1

            records.append(
                {
                    "external_subject_name":
                        str(row.subject),
                    "external_subject_numeric_id":
                        int(subject_id),
                    "external_slice_index":
                        int(chosen_slice),
                    "external_modality":
                        str(row.reference_modality),
                    "external_brain_pixels":
                        int(row.brain_pixels),
                    "training_compatible_donor_count":
                        int(
                            row.training_compatible_donor_count
                        ),
                    "final_base_use_index":
                        int(usage[chosen_slice]),
                    "subject_active_base_count":
                        int(len(subject_bases)),

                    "subject_required_reuse_ceiling":
                        int(required_reuse_ceiling),
                }
            )

    slots = pd.DataFrame(records)

    if len(slots) != NEW_CASE_COUNT:
        raise LibraryDesignError(
            f"Expected {NEW_CASE_COUNT:,} new base slots; "
            f"observed {len(slots):,}."
        )

    per_subject = (
        slots.groupby("external_subject_numeric_id")
        .size()
    )

    if not (per_subject == 78).all():
        raise LibraryDesignError(
            "New base-slot table is not exactly 78 cases per subject."
        )

    return slots


def assign_unique_donors(
    *,
    slots: pd.DataFrame,
    bases: pd.DataFrame,
    donors: pd.DataFrame,
    donor_indices_by_base: list[np.ndarray],
    batch1: pd.DataFrame,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """
    Assign one globally unique compatible donor to every planned slot.

    Production assignment is solved exactly as a sparse integral max-flow
    problem after deterministic candidate thinning.

    Frozen constraints
    ------------------
    1. Every planned new case receives exactly one donor slice.
    2. Every donor slice is used at most once across the complete library.
    3. All Batch-0001 donor slices remain unavailable to new cases.
    4. Total use of any donor training volume is at most
       ``DONOR_SUBJECT_CAP`` across Batch 0001 and all new cases.
    5. Every slot-donor edge belongs to the previously audited compatibility
       graph.
    6. Each planned slot contributes at most
       ``EXACT_FLOW_CANDIDATE_LIMIT`` deterministic candidate edges.

    The candidate-thinning seed is derived from the global design seed and
    slot index. A failure to obtain complete flow is fatal: production does
    not silently change the candidate limit or relax a scientific constraint.
    """

    # ------------------------------------------------------------
    # Resolve each planned external base to its cached compatibility row.
    # ------------------------------------------------------------

    base_key_to_index = {
        (
            int(row.subject_numeric_id),
            int(row.slice_index),
        ): i
        for i, row in enumerate(
            bases.itertuples(index=False)
        )
    }

    donor_files = (
        donors["donor_h5_file"]
        .astype(str)
        .to_numpy()
    )

    donor_volumes = (
        donors["volume"]
        .astype(int)
        .to_numpy()
    )

    donor_file_to_index = {
        str(value): i
        for i, value in enumerate(
            donor_files
        )
    }

    # ------------------------------------------------------------
    # Freeze Batch 0001.
    # ------------------------------------------------------------

    existing_donor_files = set(
        batch1["donor_h5_file"].astype(str)
    )

    missing_existing = sorted(
        existing_donor_files
        - set(donor_file_to_index)
    )

    if missing_existing:
        raise LibraryDesignError(
            "Batch 0001 contains donor slices absent from the "
            "downstream-training donor pool.\n"
            + "\n".join(
                missing_existing[:10]
            )
        )

    frozen_donor_indices = {
        donor_file_to_index[value]
        for value in existing_donor_files
    }

    frozen_volume_usage = Counter(
        int(value)
        for value in batch1["donor_volume"]
    )

    if (
        max(
            frozen_volume_usage.values(),
            default=0,
        )
        > DONOR_SUBJECT_CAP
    ):
        raise LibraryDesignError(
            "Batch 0001 already violates the donor-subject cap."
        )

    # ------------------------------------------------------------
    # Construct complete available candidate lists.
    # ------------------------------------------------------------

    slot_candidates: list[np.ndarray] = []

    for row in slots.itertuples(index=False):

        key = (
            int(row.external_subject_numeric_id),
            int(row.external_slice_index),
        )

        if key not in base_key_to_index:
            raise LibraryDesignError(
                "A planned slot does not resolve to the frozen base pool.\n"
                f"Base key: {key}"
            )

        base_index = base_key_to_index[
            key
        ]

        retained = np.asarray(
            [
                int(i)
                for i in donor_indices_by_base[
                    base_index
                ]
                if int(i)
                not in frozen_donor_indices
            ],
            dtype=np.int32,
        )

        if (
            retained.size
            < MIN_ACTIVE_BASE_DONOR_SUPPORT
        ):
            raise LibraryDesignError(
                "A planned slot violates the frozen minimum active-base "
                "donor-support rule.\n"
                f"External subject: "
                f"{row.external_subject_numeric_id}\n"
                f"External slice: "
                f"{row.external_slice_index}\n"
                f"Available donors: {retained.size}\n"
                f"Required minimum: "
                f"{MIN_ACTIVE_BASE_DONOR_SUPPORT}"
            )

        slot_candidates.append(
            retained
        )

    candidate_counts = np.asarray(
        [
            len(values)
            for values in slot_candidates
        ],
        dtype=np.int64,
    )

    # ------------------------------------------------------------
    # Deterministic candidate thinning.
    #
    # If a slot has <=32 available candidates, retain all of them.
    # Otherwise select exactly 32 without replacement using a
    # deterministic slot-specific seed.
    # ------------------------------------------------------------

    selected_candidates: list[np.ndarray] = []

    for slot_index, values in enumerate(
        slot_candidates
    ):

        if (
            len(values)
            <= EXACT_FLOW_CANDIDATE_LIMIT
        ):
            chosen = np.asarray(
                values,
                dtype=np.int32,
            )

        else:
            slot_rng = np.random.default_rng(
                int(seed) * 1_000_003
                + int(slot_index)
            )

            chosen = np.asarray(
                slot_rng.choice(
                    values,
                    size=EXACT_FLOW_CANDIDATE_LIMIT,
                    replace=False,
                ),
                dtype=np.int32,
            )

        selected_candidates.append(
            chosen
        )

    # ------------------------------------------------------------
    # Residual donor-volume capacity after frozen Batch 0001.
    # ------------------------------------------------------------

    all_volumes = sorted(
        int(value)
        for value in donors[
            "volume"
        ].unique()
    )

    volume_to_local = {
        volume: i
        for i, volume in enumerate(
            all_volumes
        )
    }

    residual_capacity = {
        volume:
            DONOR_SUBJECT_CAP
            - int(
                frozen_volume_usage[
                    volume
                ]
            )
        for volume in all_volumes
    }

    if (
        min(
            residual_capacity.values()
        )
        < 0
    ):
        raise LibraryDesignError(
            "Batch 0001 produces negative residual donor-volume capacity."
        )

    if (
        sum(
            residual_capacity.values()
        )
        < len(slots)
    ):
        raise LibraryDesignError(
            "Aggregate residual donor-volume capacity is insufficient for "
            "the planned new cases."
        )

    # ------------------------------------------------------------
    # Sparse flow graph.
    #
    # SOURCE -> slot       capacity 1
    # slot   -> donor      capacity 1
    # donor  -> volume     capacity 1
    # volume -> SINK       residual donor-volume capacity
    # ------------------------------------------------------------

    n_slots = int(
        len(slots)
    )

    n_donors = int(
        len(donors)
    )

    n_volumes = int(
        len(all_volumes)
    )

    source = 0
    slot_offset = 1
    donor_offset = (
        slot_offset
        + n_slots
    )
    volume_offset = (
        donor_offset
        + n_donors
    )
    sink = (
        volume_offset
        + n_volumes
    )
    n_nodes = sink + 1

    rows: list[int] = []
    cols: list[int] = []
    capacities: list[int] = []

    # Source -> slot.
    for slot_index in range(
        n_slots
    ):
        rows.append(
            source
        )
        cols.append(
            slot_offset
            + slot_index
        )
        capacities.append(
            1
        )

    # Slot -> donor.
    donor_union: set[int] = set()
    slot_donor_edge_count = 0

    for slot_index, values in enumerate(
        selected_candidates
    ):

        slot_node = (
            slot_offset
            + slot_index
        )

        rows.extend(
            [slot_node]
            * len(values)
        )

        cols.extend(
            (
                donor_offset
                + values
            ).tolist()
        )

        capacities.extend(
            [1]
            * len(values)
        )

        donor_union.update(
            int(value)
            for value in values
        )

        slot_donor_edge_count += int(
            len(values)
        )

    # Donor -> donor volume.
    for donor_index in sorted(
        donor_union
    ):

        volume = int(
            donor_volumes[
                donor_index
            ]
        )

        rows.append(
            donor_offset
            + donor_index
        )

        cols.append(
            volume_offset
            + volume_to_local[
                volume
            ]
        )

        capacities.append(
            1
        )

    # Donor volume -> sink.
    for volume in all_volumes:

        capacity = int(
            residual_capacity[
                volume
            ]
        )

        if capacity <= 0:
            continue

        rows.append(
            volume_offset
            + volume_to_local[
                volume
            ]
        )

        cols.append(
            sink
        )

        capacities.append(
            capacity
        )

    graph = coo_matrix(
        (
            np.asarray(
                capacities,
                dtype=np.int32,
            ),
            (
                np.asarray(
                    rows,
                    dtype=np.int32,
                ),
                np.asarray(
                    cols,
                    dtype=np.int32,
                ),
            ),
        ),
        shape=(
            n_nodes,
            n_nodes,
        ),
        dtype=np.int32,
    ).tocsr()

    print()
    print(
        "Running exact sparse max-flow donor assignment..."
    )

    print(
        "Planned slots             :",
        f"{n_slots:,}",
    )

    print(
        "Candidate limit / slot    :",
        EXACT_FLOW_CANDIDATE_LIMIT,
    )

    print(
        "Slot-donor edges          :",
        f"{slot_donor_edge_count:,}",
    )

    print(
        "Distinct candidate donors :",
        f"{len(donor_union):,}",
    )

    print(
        "Sparse graph nodes        :",
        f"{n_nodes:,}",
    )

    print(
        "Sparse graph edges        :",
        f"{graph.nnz:,}",
    )

    flow_result = maximum_flow(
        graph,
        source,
        sink,
    )

    flow_value = int(
        flow_result.flow_value
    )

    print(
        "Maximum flow              :",
        f"{flow_value:,} / {n_slots:,}",
    )

    if flow_value != n_slots:
        raise LibraryDesignError(
            "The frozen exact-flow candidate graph does not admit a "
            "complete assignment.\n\n"
            f"Observed flow: {flow_value:,}\n"
            f"Required flow: {n_slots:,}\n"
            f"Candidate limit: {EXACT_FLOW_CANDIDATE_LIMIT}\n\n"
            "Production will not alter the candidate limit or relax a "
            "scientific constraint automatically."
        )

    # ------------------------------------------------------------
    # Extract integral slot -> donor assignments.
    # ------------------------------------------------------------

    flow = flow_result.flow.tocsr()

    assigned_indices = np.full(
        n_slots,
        -1,
        dtype=np.int32,
    )

    for slot_index in range(
        n_slots
    ):

        slot_node = (
            slot_offset
            + slot_index
        )

        start = flow.indptr[
            slot_node
        ]

        end = flow.indptr[
            slot_node + 1
        ]

        node_indices = flow.indices[
            start:end
        ]

        node_values = flow.data[
            start:end
        ]

        donor_mask = (
            (node_values > 0)
            & (
                node_indices
                >= donor_offset
            )
            & (
                node_indices
                < volume_offset
            )
        )

        matched_nodes = node_indices[
            donor_mask
        ]

        if len(matched_nodes) != 1:
            raise LibraryDesignError(
                "Exact flow did not produce exactly one donor for "
                f"planned slot {slot_index}."
            )

        assigned_indices[
            slot_index
        ] = (
            int(
                matched_nodes[0]
            )
            - donor_offset
        )

    # ------------------------------------------------------------
    # Independent hard validation.
    # ------------------------------------------------------------

    if (
        assigned_indices
        < 0
    ).any():
        raise LibraryDesignError(
            "Exact flow returned an unassigned planned slot."
        )

    if (
        len(
            np.unique(
                assigned_indices
            )
        )
        != n_slots
    ):
        raise LibraryDesignError(
            "Exact flow reused a donor slice."
        )

    final_volume_usage = Counter(
        frozen_volume_usage
    )

    for donor_index in assigned_indices:

        final_volume_usage[
            int(
                donor_volumes[
                    donor_index
                ]
            )
        ] += 1

    if (
        max(
            final_volume_usage.values(),
            default=0,
        )
        > DONOR_SUBJECT_CAP
    ):
        raise LibraryDesignError(
            "Exact flow exceeded the frozen donor-subject cap."
        )

    # Confirm selected assignment is in BOTH the thinned graph and
    # the complete frozen compatibility graph.
    for slot_index, donor_index in enumerate(
        assigned_indices
    ):

        donor_index = int(
            donor_index
        )

        if donor_index not in set(
            int(value)
            for value in selected_candidates[
                slot_index
            ]
        ):
            raise LibraryDesignError(
                "Exact-flow assignment is absent from the deterministic "
                "thinned candidate graph."
            )

        if donor_index not in set(
            int(value)
            for value in slot_candidates[
                slot_index
            ]
        ):
            raise LibraryDesignError(
                "Exact-flow assignment is absent from the complete frozen "
                "compatibility graph."
            )

    # ------------------------------------------------------------
    # Attach donor metadata.
    # ------------------------------------------------------------

    donor_rows = donors.iloc[
        assigned_indices
    ].reset_index(
        drop=True
    )

    result = slots.reset_index(
        drop=True
    ).copy()

    result[
        "candidate_donors_before_uniqueness"
    ] = candidate_counts

    result[
        "exact_flow_candidate_count"
    ] = [
        int(
            len(values)
        )
        for values in selected_candidates
    ]

    result["donor_volume"] = (
        donor_rows["volume"]
        .astype(int)
    )

    result["donor_slice_index"] = (
        donor_rows["slice"]
        .astype(int)
    )

    result["donor_h5_file"] = (
        donor_rows["donor_h5_file"]
        .astype(str)
    )

    result["donor_mask_pixels"] = (
        donor_rows["loaded_mask_pixels"]
        .astype(int)
    )

    result["donor_tumor_area_pixels"] = (
        donor_rows[
            "tumor_area_pixels"
        ]
    )

    result["donor_bbox_area"] = (
        donor_rows[
            "bbox_area"
        ]
    )

    result["donor_component_count"] = (
        donor_rows[
            "connected_component_count"
        ]
    )

    result[
        "donor_largest_component_fraction"
    ] = donor_rows[
        "largest_component_fraction"
    ]

    result[
        "donor_centroid_x_normalized"
    ] = donor_rows[
        "centroid_x_normalized"
    ]

    result[
        "donor_centroid_y_normalized"
    ] = donor_rows[
        "centroid_y_normalized"
    ]

    result[
        "donor_centroid_laterality"
    ] = donor_rows[
        "centroid_laterality"
    ].astype(str)

    donor_usage_values = list(
        final_volume_usage.values()
    )

    flow_metadata = {
        "algorithm":
            "scipy_sparse_maximum_flow",
        "scipy_version":
            str(
                scipy.__version__
            ),
        "candidate_limit_per_slot":
            int(
                EXACT_FLOW_CANDIDATE_LIMIT
            ),
        "maximum_flow":
            int(
                flow_value
            ),
        "required_flow":
            int(
                n_slots
            ),
        "complete_flow":
            bool(
                flow_value
                == n_slots
            ),
        "slot_donor_edges":
            int(
                slot_donor_edge_count
            ),
        "distinct_candidate_donors":
            int(
                len(
                    donor_union
                )
            ),
        "graph_nodes":
            int(
                n_nodes
            ),
        "graph_edges":
            int(
                graph.nnz
            ),
        "unique_assigned_donor_slices":
            int(
                len(
                    np.unique(
                        assigned_indices
                    )
                )
            ),
        "donor_subjects_used_final":
            int(
                len(
                    final_volume_usage
                )
            ),
        "maximum_final_donor_subject_use":
            int(
                max(
                    donor_usage_values,
                    default=0,
                )
            ),
        "donor_subjects_at_cap":
            int(
                sum(
                    value
                    == DONOR_SUBJECT_CAP
                    for value
                    in donor_usage_values
                )
            ),
    }

    print()
    print(
        "Exact donor assignment succeeded."
    )

    print(
        "Unique new donor slices   :",
        f"{len(np.unique(assigned_indices)):,}",
    )

    print(
        "Maximum donor-subject use :",
        max(
            donor_usage_values,
            default=0,
        ),
    )

    print(
        "Donor subjects at cap     :",
        sum(
            value
            == DONOR_SUBJECT_CAP
            for value
            in donor_usage_values
        ),
    )

    return (
        result,
        flow_metadata,
    )

def build_complete_design(
    *,
    batch1: pd.DataFrame,
    new_assignments: pd.DataFrame,
) -> pd.DataFrame:
    records = []

    # Preserve Batch 0001 in its existing order.
    for i, row in batch1.iterrows():
        library_index = i + 1

        records.append(
            {
                "library_index": library_index,
                "library_case_id":
                    f"synthetic_{library_index:06d}",
                "batch_id": "batch_0001",
                "case_status": "already_generated",
                "source_case_id": str(row["case_id"]),
                "external_subject_name":
                    str(row["external_subject_name"]),
                "external_subject_numeric_id":
                    int(row["external_subject_numeric_id"]),
                "external_slice_index":
                    int(row["external_slice_index"]),
                "external_modality": "flair",
                "external_brain_pixels": np.nan,
                "training_compatible_donor_count": np.nan,
                "final_base_use_index": np.nan,
                "donor_volume": int(row["donor_volume"]),
                "donor_slice_index":
                    int(row["donor_slice_index"]),
                "donor_h5_file":
                    str(row["donor_h5_file"]),
                "donor_mask_pixels":
                    int(row["mask_pixels"]),
                "donor_tumor_area_pixels": np.nan,
                "donor_bbox_area": np.nan,
                "donor_component_count": np.nan,
                "donor_largest_component_fraction": np.nan,
                "donor_centroid_x_normalized": np.nan,
                "donor_centroid_y_normalized": np.nan,
                "donor_centroid_laterality": np.nan,
                "candidate_donors_before_uniqueness": np.nan,
            }
        )

    # Sort new cases by subject and then stable base-slot order.
    new_assignments = (
        new_assignments
        .sort_values(
            [
                "external_subject_numeric_id",
                "external_slice_index",
                "donor_volume",
                "donor_slice_index",
            ],
            kind="stable",
        )
        .reset_index(drop=True)
    )

    for offset, row in new_assignments.iterrows():
        library_index = (
            EXISTING_CASE_COUNT
            + offset
            + 1
        )

        batch_number = (
            (library_index - 1) // 250
            + 1
        )

        records.append(
            {
                "library_index": library_index,
                "library_case_id":
                    f"synthetic_{library_index:06d}",
                "batch_id":
                    f"batch_{batch_number:04d}",
                "case_status": "planned",
                "source_case_id": "",
                "external_subject_name":
                    str(row["external_subject_name"]),
                "external_subject_numeric_id":
                    int(row["external_subject_numeric_id"]),
                "external_slice_index":
                    int(row["external_slice_index"]),
                "external_modality":
                    str(row["external_modality"]),
                "external_brain_pixels":
                    int(row["external_brain_pixels"]),
                "training_compatible_donor_count":
                    int(
                        row[
                            "training_compatible_donor_count"
                        ]
                    ),
                "final_base_use_index":
                    int(row["final_base_use_index"]),
                "donor_volume":
                    int(row["donor_volume"]),
                "donor_slice_index":
                    int(row["donor_slice_index"]),
                "donor_h5_file":
                    str(row["donor_h5_file"]),
                "donor_mask_pixels":
                    int(row["donor_mask_pixels"]),
                "donor_tumor_area_pixels":
                    row["donor_tumor_area_pixels"],
                "donor_bbox_area":
                    row["donor_bbox_area"],
                "donor_component_count":
                    row["donor_component_count"],
                "donor_largest_component_fraction":
                    row[
                        "donor_largest_component_fraction"
                    ],
                "donor_centroid_x_normalized":
                    row[
                        "donor_centroid_x_normalized"
                    ],
                "donor_centroid_y_normalized":
                    row[
                        "donor_centroid_y_normalized"
                    ],
                "donor_centroid_laterality":
                    row[
                        "donor_centroid_laterality"
                    ],
                "candidate_donors_before_uniqueness":
                    int(
                        row[
                            "candidate_donors_before_uniqueness"
                        ]
                    ),
            }
        )

    design = pd.DataFrame(records)

    design["pair_key"] = (
        design["external_subject_name"].astype(str)
        + "|slice_"
        + design["external_slice_index"].astype(str)
        + "|"
        + design["donor_h5_file"].astype(str)
    )

    return design


def audit_design(
    *,
    design: pd.DataFrame,
    batch1: pd.DataFrame,
) -> pd.DataFrame:
    checks = []

    def add(name: str, passed: bool, observed) -> None:
        checks.append(
            {
                "check": name,
                "passed": bool(passed),
                "observed": str(observed),
            }
        )

    add(
        "exact_library_size",
        len(design) == TARGET_LIBRARY_SIZE,
        len(design),
    )

    add(
        "unique_library_case_id",
        design["library_case_id"].nunique()
        == TARGET_LIBRARY_SIZE,
        design["library_case_id"].nunique(),
    )

    add(
        "unique_pair_key",
        design["pair_key"].nunique()
        == TARGET_LIBRARY_SIZE,
        design["pair_key"].nunique(),
    )

    add(
        "unique_donor_slices",
        design["donor_h5_file"].nunique()
        == TARGET_LIBRARY_SIZE,
        design["donor_h5_file"].nunique(),
    )

    per_external_subject = (
        design.groupby(
            "external_subject_numeric_id"
        ).size()
    )

    add(
        "125_external_subjects",
        len(per_external_subject) == 125,
        len(per_external_subject),
    )

    add(
        "80_cases_per_external_subject",
        bool(
            (per_external_subject == 80).all()
        ),
        (
            f"min={per_external_subject.min()}, "
            f"max={per_external_subject.max()}"
        ),
    )

    per_donor_volume = (
        design.groupby("donor_volume")
        .size()
    )

    add(
        "donor_subject_cap_31",
        int(per_donor_volume.max())
        <= DONOR_SUBJECT_CAP,
        int(per_donor_volume.max()),
    )

    add(
        "40_batches_of_250",
        (
            design["batch_id"].nunique() == 40
            and (
                design.groupby("batch_id")
                .size()
                == 250
            ).all()
        ),
        (
            f"batches={design['batch_id'].nunique()}, "
            f"min_batch={design.groupby('batch_id').size().min()}, "
            f"max_batch={design.groupby('batch_id').size().max()}"
        ),
    )

    # Exact preservation of Batch 0001 base/donor pair sequence.
    first = design.iloc[:250]

    batch1_pairs = list(
        zip(
            batch1["external_subject_numeric_id"],
            batch1["external_slice_index"],
            batch1["donor_h5_file"],
        )
    )

    design_pairs = list(
        zip(
            first["external_subject_numeric_id"],
            first["external_slice_index"],
            first["donor_h5_file"],
        )
    )

    add(
        "batch0001_exactly_preserved",
        batch1_pairs == design_pairs,
        batch1_pairs == design_pairs,
    )

    audit = pd.DataFrame(checks)

    failures = audit[
        ~audit["passed"]
    ]

    if len(failures):
        raise LibraryDesignError(
            "Frozen library design failed audit:\n\n"
            + failures.to_string(index=False)
        )

    return audit

def save_compatibility_cache(
    *,
    path: Path,
    counts: np.ndarray,
    donor_indices_by_base: list[np.ndarray],
) -> None:
    """Save variable-length per-base donor lists in CSR-style arrays."""

    offsets = np.zeros(
        len(donor_indices_by_base) + 1,
        dtype=np.int64,
    )

    for i, values in enumerate(
        donor_indices_by_base,
        start=1,
    ):
        offsets[i] = (
            offsets[i - 1]
            + len(values)
        )

    indices = np.concatenate(
        donor_indices_by_base
    ).astype(
        np.int32,
        copy=False,
    )

    np.savez_compressed(
        path,
        counts=counts.astype(
            np.int64,
            copy=False,
        ),
        offsets=offsets,
        indices=indices,
    )


def load_compatibility_cache(
    path: Path,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Load CSR-style training-only compatibility cache."""

    obj = np.load(
        path,
        allow_pickle=False,
    )

    counts = obj["counts"].astype(
        np.int64,
        copy=False,
    )

    offsets = obj["offsets"].astype(
        np.int64,
        copy=False,
    )

    indices = obj["indices"].astype(
        np.int32,
        copy=False,
    )

    lists = [
        indices[
            offsets[i]:offsets[i + 1]
        ]
        for i in range(len(counts))
    ]

    return counts, lists


def main() -> None:
    args = parse_args()

    folders_config = load_folders_config(
        args.folders_file
    )

    validation_dataset = resolve_file(
        resolve_path(
            key="yaml_validation_dataset_path",
            cli_value=args.validation_dataset,
            config=folders_config,
            selector=None,
        ),
        "Validation dataset specification",
    )

    h5_root = resolve_directory(
        resolve_path(
            key="h5_root",
            cli_value=args.h5_root,
            config=folders_config,
            selector=None,
        ),
        "BraTS H5 root",
    )

    if args.base_counts_csv is not None:
        base_counts_path = resolve_file(
            args.base_counts_csv,
            "External-base compatibility table",
        )
    else:
        nnunet_run_root = resolve_path(
            key="nnunet_run_root",
            cli_value=None,
            config=folders_config,
            selector=None,
        ).expanduser().resolve()

        base_counts_path = resolve_file(
            nnunet_run_root
            / "external_pair_space_audit"
            / "external_base_compatibility_counts.csv",
            "External-base compatibility table",
        )

    donor_pool_path = resolve_file(
        (
            args.training_donor_pool
            if args.training_donor_pool is not None
            else DEFAULT_TRAINING_DONOR_POOL
        ),
        "Training-only donor pool",
    )

    batch1_path = resolve_file(
        (
            args.batch0001_manifest
            if args.batch0001_manifest is not None
            else DEFAULT_BATCH0001_MANIFEST
        ),
        "Batch 0001 manifest",
    )

    save_folders_config(
        args.folders_file,
        folders_config,
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

    output_paths = {
        "design": output_dir / DESIGN_NAME,
        "audit": output_dir / AUDIT_NAME,
        "summary": output_dir / SUMMARY_NAME,
    }

    existing_outputs = [
        str(p)
        for p in output_paths.values()
        if p.exists()
    ]

    if existing_outputs:
        raise FileExistsError(
            "Refusing to overwrite existing frozen design artifacts:\n"
            + "\n".join(existing_outputs)
        )

    if args.base_batch_size <= 0:
        raise LibraryDesignError(
            "--base-batch-size must be positive."
        )

    if args.donor_batch_size <= 0:
        raise LibraryDesignError(
            "--donor-batch-size must be positive."
        )

    device = resolve_device(args.device)

    print("=" * 78)
    print("BR-LoRA 10,000-CASE LIBRARY DESIGN")
    print("=" * 78)
    print("Design only; no synthesis will be performed.")
    print()
    print("Target library size      :", f"{TARGET_LIBRARY_SIZE:,}")
    print("Existing frozen cases    :", f"{EXISTING_CASE_COUNT:,}")
    print("New cases to design      :", f"{NEW_CASE_COUNT:,}")
    print("Cases / external subject :", TARGET_PER_EXTERNAL_SUBJECT)
    print("Donor-subject cap        :", DONOR_SUBJECT_CAP)
    print("Donor-slice reuse        : prohibited")
    print("Minimum overlap          :", MIN_OVERLAP)
    print("Brain threshold          :", BRAIN_THRESHOLD)
    print("Random seed              :", args.seed)
    print("Device                   :", device)

    batch1 = load_existing_batch(
        batch1_path
    )

    bases = load_candidate_bases(
        base_counts_path
    )

    donors = load_training_donors(
        donor_pool_path,
        h5_root,
    )

    base_masks = load_base_masks(
        bases,
        validation_dataset,
    )

    donor_masks, donor_pixel_counts = (
        load_donor_masks(
            donors
        )
    )

    compatibility_cache = (
        output_dir
        / "training_only_compatibility_cache.npz"
    )

    if compatibility_cache.is_file():
        print()
        print(
            "Loading cached training-only compatibility:"
        )
        print(
            compatibility_cache
        )

        (
            training_compatibility_counts,
            donor_indices_by_base,
        ) = load_compatibility_cache(
            compatibility_cache
        )

        if len(training_compatibility_counts) != len(bases):
            raise LibraryDesignError(
                "Compatibility cache base count is incompatible "
                "with the current base table."
            )

    else:
        (
            training_compatibility_counts,
            donor_indices_by_base,
        ) = recompute_training_only_compatibility(
            base_masks=base_masks,
            donor_masks=donor_masks,
            donor_pixel_counts=donor_pixel_counts,
            base_batch_size=args.base_batch_size,
            donor_batch_size=args.donor_batch_size,
            device=device,
        )

        save_compatibility_cache(
            path=compatibility_cache,
            counts=training_compatibility_counts,
            donor_indices_by_base=donor_indices_by_base,
        )

        print()
        print(
            "Saved compatibility cache:",
            compatibility_cache,
        )

    bases[
        "training_compatible_donor_count"
    ] = training_compatibility_counts

    print()
    print("===== TRAINING-ONLY COMPATIBILITY =====")
    print(
        "Bases retained with >=1 donor :",
        int(
            (
                training_compatibility_counts
                > 0
            ).sum()
        ),
    )

    print(
        "Bases losing all compatibility:",
        int(
            (
                training_compatibility_counts
                == 0
            ).sum()
        ),
    )

    positive_counts = (
        training_compatibility_counts[
            training_compatibility_counts > 0
        ]
    )

    print(
        "Compatible donors / retained base:"
    )

    print(
        pd.Series(
            positive_counts
        ).describe().to_string()
    )

    per_subject_usable = (
        bases[
            bases[
                "training_compatible_donor_count"
            ] > 0
        ]
        .groupby("subject_numeric_id")
        .size()
    )

    print()
    print("Training-compatible bases / subject:")
    print(
        per_subject_usable
        .describe()
        .to_string()
    )

    if len(per_subject_usable) != 125:
        raise LibraryDesignError(
            "Training-only donor restriction removed all usable bases "
            "for at least one external subject."
        )

    rng = np.random.default_rng(
        args.seed
    )

    slots = build_base_slots(
        bases=bases,
        training_counts=training_compatibility_counts,
        donor_indices_by_base=donor_indices_by_base,
        donors=donors,
        batch1=batch1,
        rng=rng,
    )
    
    print()
    print("===== NEW BASE-SLOT DESIGN =====")
    print("New slots:", f"{len(slots):,}")

    # Final base usage across Batch 0001 + planned cases.
    existing_base_usage = (
        batch1[
            [
                "external_subject_numeric_id",
                "external_slice_index",
            ]
        ]
        .value_counts()
        .rename("existing")
    )

    new_base_usage = (
        slots[
            [
                "external_subject_numeric_id",
                "external_slice_index",
            ]
        ]
        .value_counts()
        .rename("new")
    )

    combined_base_usage = (
        pd.concat(
            [
                existing_base_usage,
                new_base_usage,
            ],
            axis=1,
        )
        .fillna(0)
        .sum(axis=1)
    )

    print(
        "Distinct bases used:",
        len(combined_base_usage),
    )

    print(
        "Maximum uses of one base:",
        int(combined_base_usage.max()),
    )

    print(
        "Base reuse distribution:"
    )

    print(
        combined_base_usage
        .value_counts()
        .sort_index()
        .to_string()
    )

    print()
    print("Assigning globally unique compatible donors...")

    (
        new_assignments,
        exact_flow_metadata,
    ) = assign_unique_donors(
        slots=slots,
        bases=bases,
        donors=donors,
        donor_indices_by_base=donor_indices_by_base,
        batch1=batch1,
        seed=args.seed,
    )

    design = build_complete_design(
        batch1=batch1,
        new_assignments=new_assignments,
    )

    audit = audit_design(
        design=design,
        batch1=batch1,
    )

    design.to_csv(
        output_paths["design"],
        index=False,
    )

    audit.to_csv(
        output_paths["audit"],
        index=False,
    )

    donor_usage = (
        design.groupby("donor_volume")
        .size()
    )

    base_usage = (
        design.groupby(
            [
                "external_subject_numeric_id",
                "external_slice_index",
            ]
        )
        .size()
    )

    summary = {
        "status": "frozen_design_complete",

        "design_role":
            (
                "Prespecified 10,000-case BR-LoRA synthetic-library "
                "conditioning design. No synthesis is performed by this "
                "script."
            ),

        "design_parameters": {
            "target_library_size":
                TARGET_LIBRARY_SIZE,
            "existing_batch0001_cases":
                EXISTING_CASE_COUNT,
            "new_cases":
                NEW_CASE_COUNT,
            "external_subject_count":
                125,
            "cases_per_external_subject":
                TARGET_PER_EXTERNAL_SUBJECT,
            "downstream_training_donor_subjects":
                int(donors["volume"].nunique()),
            "downstream_training_donor_slices":
                int(len(donors)),
            "donor_subject_cap":
                DONOR_SUBJECT_CAP,
            "donor_slice_reuse_allowed":
                False,
            "minimum_active_base_donor_support":
                MIN_ACTIVE_BASE_DONOR_SUPPORT,
            "exact_flow_candidate_limit_per_slot":
                EXACT_FLOW_CANDIDATE_LIMIT,
            "brain_threshold":
                BRAIN_THRESHOLD,
            "minimum_mask_inside_brain_fraction":
                MIN_OVERLAP,
            "random_seed":
                int(args.seed),
        },

        "training_only_compatibility": {
            "candidate_bases_before_training_donor_restriction":
                int(len(bases)),
            "bases_with_at_least_one_training_donor":
                int(
                    (
                        training_compatibility_counts
                        > 0
                    ).sum()
                ),
            "bases_losing_all_compatibility":
                int(
                    (
                        training_compatibility_counts
                        == 0
                    ).sum()
                ),
            "minimum_training_compatible_donors":
                int(
                    positive_counts.min()
                ),
            "median_training_compatible_donors":
                float(
                    np.median(
                        positive_counts
                    )
                ),
            "maximum_training_compatible_donors":
                int(
                    positive_counts.max()
                ),
        },

        "exact_donor_assignment": exact_flow_metadata,

        "final_library": {
            "cases":
                int(len(design)),
            "batches":
                int(
                    design["batch_id"]
                    .nunique()
                ),
            "unique_external_subjects":
                int(
                    design[
                        "external_subject_numeric_id"
                    ].nunique()
                ),
            "unique_external_base_slices":
                int(
                    design[
                        [
                            "external_subject_numeric_id",
                            "external_slice_index",
                        ]
                    ]
                    .drop_duplicates()
                    .shape[0]
                ),
            "maximum_base_reuse":
                int(base_usage.max()),
            "unique_donor_slices":
                int(
                    design[
                        "donor_h5_file"
                    ].nunique()
                ),
            "unique_donor_subjects_used":
                int(
                    design[
                        "donor_volume"
                    ].nunique()
                ),
            "maximum_cases_from_one_donor_subject":
                int(donor_usage.max()),
            "minimum_cases_from_used_donor_subject":
                int(donor_usage.min()),
            "median_cases_per_used_donor_subject":
                float(donor_usage.median()),
        },

        "provenance": {
            "created_at_utc":
                datetime.now(
                    timezone.utc
                ).isoformat(),
            "git_commit":
                git_commit(),
            "git_worktree_clean":
                git_clean(),
            "script_path":
                str(
                    Path(__file__)
                    .resolve()
                ),
            "script_sha256":
                sha256_file(
                    Path(__file__)
                    .resolve()
                ),
            "validation_dataset":
                str(validation_dataset),
            "validation_dataset_sha256":
                sha256_file(validation_dataset),
            "base_counts_csv":
                str(base_counts_path),
            "base_counts_sha256":
                sha256_file(base_counts_path),
            "training_donor_pool":
                str(donor_pool_path),
            "training_donor_pool_sha256":
                sha256_file(donor_pool_path),
            "batch0001_manifest":
                str(batch1_path),
            "batch0001_manifest_sha256":
                sha256_file(batch1_path),
        },

        "output_artifacts": {
            "design_csv":
                str(output_paths["design"]),
            "audit_csv":
                str(output_paths["audit"]),
        },
    }

    with output_paths["summary"].open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
            sort_keys=True,
        )

        file.write("\n")

    print()
    print("=" * 78)
    print("FROZEN LIBRARY DESIGN COMPLETE")
    print("=" * 78)

    print(
        "Cases                     :",
        f"{len(design):,}",
    )

    print(
        "External subjects         :",
        design[
            "external_subject_numeric_id"
        ].nunique(),
    )

    print(
        "Cases / external subject :",
        (
            design.groupby(
                "external_subject_numeric_id"
            )
            .size()
            .unique()
            .tolist()
        ),
    )

    print(
        "Unique base slices        :",
        design[
            [
                "external_subject_numeric_id",
                "external_slice_index",
            ]
        ]
        .drop_duplicates()
        .shape[0],
    )

    print(
        "Maximum base reuse        :",
        int(base_usage.max()),
    )

    print(
        "Unique donor slices       :",
        design[
            "donor_h5_file"
        ].nunique(),
    )

    print(
        "Donor subjects used       :",
        design[
            "donor_volume"
        ].nunique(),
    )

    print(
        "Maximum donor-subject use :",
        int(donor_usage.max()),
    )

    print(
        "Batches                   :",
        design[
            "batch_id"
        ].nunique(),
    )

    print()
    print("Audit:")
    print(
        audit.to_string(
            index=False
        )
    )

    print()
    print("Outputs:")
    for key, path in output_paths.items():
        print(
            f"  {key:8s}: {path}"
        )


if __name__ == "__main__":
    main()
