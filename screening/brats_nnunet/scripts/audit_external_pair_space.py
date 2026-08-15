#!/usr/bin/env python3
"""
Audit the complete eligible external-base / training-donor pair space.

This script does not construct the definitive external evaluation manifest.

It measures how many base-donor combinations satisfy the validated regional
composition rules after external tumor-free screening.

External-base eligibility
-------------------------
The input screening table must identify the base slice as

    tumor_free_candidate == True

where the current screening workflow defines this as

    predicted_tumor_pixels == 0

Training-donor eligibility
--------------------------
Training donors must satisfy the same whole-tumor slice-selection criterion
used by the training dataset:

    label0_pxl_cnt
  + label1_pxl_cnt
  + label2_pxl_cnt
    >= min_tumor_pixels

Pair eligibility
----------------
For each external base / donor pair:

1. The external base is loaded with ``load_validation_slice``.
2. The donor is loaded with ``load_h5_full``.
3. The donor mask must satisfy ``mask_has_margin``.
4. The external base brain mask is constructed with ``get_brain_mask``.
5. At least ``min_overlap`` of the donor-mask pixels must lie inside the
   external-base brain mask.

The full pair count is computed with batched matrix multiplication. No
Cartesian pair table is materialized.

Outputs
-------
external_pair_space_summary.json
external_base_compatibility_counts.csv
external_donor_compatibility_counts.csv
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )

import numpy as np
import pandas as pd
import torch

from src.data import (
    get_brain_mask,
    load_h5_full,
    load_validation_dataset_specification,
    load_validation_slice,
    mask_has_margin,
)


BASE_OUTPUT_NAME = (
    "external_base_compatibility_counts.csv"
)

DONOR_OUTPUT_NAME = (
    "external_donor_compatibility_counts.csv"
)

SUMMARY_OUTPUT_NAME = (
    "external_pair_space_summary.json"
)


class PairSpaceAuditError(
    RuntimeError
):
    """Raised when the external pair-space audit fails."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Count all eligible external-base / training-donor "
            "combinations under the validated composition rules."
        )
    )

    parser.add_argument(
        "--validation-dataset",
        required=True,
        type=Path,
        help="Registered BraTS validation_dataset.yaml.",
    )

    parser.add_argument(
        "--screening-csv",
        required=True,
        type=Path,
        help=(
            "Slice-level nnU-Net validation screening CSV."
        ),
    )

    parser.add_argument(
        "--training-manifest",
        required=True,
        type=Path,
        help="Training H5 manifest.csv.",
    )

    parser.add_argument(
        "--h5-root",
        required=True,
        type=Path,
        help=(
            "Directory containing reconstructed training H5 slices."
        ),
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory for audit outputs.",
    )

    parser.add_argument(
        "--external-modality",
        default="flair",
        help="External MRI modality. Default: flair.",
    )

    parser.add_argument(
        "--donor-image-channel",
        type=int,
        default=0,
        help=(
            "Training H5 image channel. Default: 0 (FLAIR)."
        ),
    )

    parser.add_argument(
        "--min-tumor-pixels",
        type=int,
        default=300,
        help=(
            "Minimum donor whole-tumor pixel count. Default: 300."
        ),
    )

    parser.add_argument(
        "--brain-threshold",
        type=float,
        default=0.05,
        help=(
            "Normalized external-base brain-mask threshold. "
            "Default: 0.05."
        ),
    )

    parser.add_argument(
        "--min-overlap",
        type=float,
        default=0.80,
        help=(
            "Minimum donor-mask fraction inside base brain. "
            "Default: 0.80."
        ),
    )

    parser.add_argument(
        "--mask-margin",
        type=int,
        default=10,
        help=(
            "Required donor-mask image-edge margin. Default: 10."
        ),
    )

    parser.add_argument(
        "--base-batch-size",
        type=int,
        default=128,
        help=(
            "Number of external brain masks processed per matrix batch. "
            "Default: 128."
        ),
    )

    parser.add_argument(
        "--donor-batch-size",
        type=int,
        default=128,
        help=(
            "Number of donor masks processed per matrix batch. "
            "Default: 128."
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
            "Device for pair-overlap matrix multiplication. "
            "Default: auto."
        ),
    )

    return parser.parse_args()


def resolve_device(
    requested: str,
) -> torch.device:
    if requested == "cpu":
        return torch.device(
            "cpu"
        )

    if requested == "cuda":
        if not torch.cuda.is_available():
            raise PairSpaceAuditError(
                "CUDA was requested but is unavailable."
            )

        return torch.device(
            "cuda"
        )

    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise PairSpaceAuditError(
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


def describe_integer_counts(
    values: np.ndarray,
) -> dict[str, float | int]:
    values = np.asarray(
        values,
        dtype=np.int64,
    )

    if values.size == 0:
        raise PairSpaceAuditError(
            "Cannot summarize an empty count array."
        )

    return {
        "minimum":
            int(
                values.min()
            ),

        "q1":
            float(
                np.quantile(
                    values,
                    0.25,
                )
            ),

        "median":
            float(
                np.median(
                    values
                )
            ),

        "mean":
            float(
                values.mean()
            ),

        "q3":
            float(
                np.quantile(
                    values,
                    0.75,
                )
            ),

        "maximum":
            int(
                values.max()
            ),
    }


def load_external_bases(
    *,
    screening_csv: Path,
    validation_dataset,
    modality: str,
    brain_threshold: float,
) -> tuple[
    pd.DataFrame,
    np.ndarray,
]:
    screening = pd.read_csv(
        screening_csv
    )

    required = {
        "subject",
        "subject_numeric_id",
        "slice_index",
        "predicted_tumor_pixels",
        "tumor_free_candidate",
    }

    missing = sorted(
        required
        - set(
            screening.columns
        )
    )

    if missing:
        raise PairSpaceAuditError(
            "Screening CSV is missing required column(s): "
            + ", ".join(
                missing
            )
        )

    candidates = screening.loc[
        screening[
            "tumor_free_candidate"
        ].astype(
            bool
        )
    ].copy()

    if candidates.empty:
        raise PairSpaceAuditError(
            "Screening CSV contains no tumor-free candidates."
        )

    if not (
        candidates[
            "predicted_tumor_pixels"
        ]
        == 0
    ).all():
        raise PairSpaceAuditError(
            "A retained tumor-free candidate has nonzero "
            "predicted_tumor_pixels."
        )

    candidates = candidates.sort_values(
        [
            "subject_numeric_id",
            "slice_index",
        ],
        kind="stable",
    ).reset_index(
        drop=True
    )

    brain_masks = np.empty(
        (
            len(
                candidates
            ),
            240 * 240,
        ),
        dtype=np.uint8,
    )

    print()
    print(
        "Loading external tumor-free bases..."
    )

    for index, row in enumerate(
        candidates.itertuples(
            index=False
        ),
        start=0,
    ):
        external = load_validation_slice(
            validation_dataset,
            subject_numeric_id=int(
                row.subject_numeric_id
            ),
            slice_index=int(
                row.slice_index
            ),
            modality=modality,
        )

        brain = get_brain_mask(
            external.image,
            threshold=brain_threshold,
        )

        if tuple(
            brain.shape
        ) != (
            1,
            240,
            240,
        ):
            raise PairSpaceAuditError(
                "Unexpected external brain-mask shape."
            )

        brain_masks[
            index,
            :,
        ] = (
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
            .reshape(
                -1
            )
        )

        count = index + 1

        if (
            count % 500 == 0
            or count == len(
                candidates
            )
        ):
            print(
                f"  External bases: "
                f"{count:,} / "
                f"{len(candidates):,}",
                flush=True,
            )

    candidates[
        "brain_pixels"
    ] = brain_masks.sum(
        axis=1,
        dtype=np.int64,
    )

    return (
        candidates,
        brain_masks,
    )


def load_eligible_donors(
    *,
    training_manifest: Path,
    h5_root: Path,
    donor_image_channel: int,
    min_tumor_pixels: int,
    mask_margin: int,
) -> tuple[
    pd.DataFrame,
    np.ndarray,
]:
    manifest = pd.read_csv(
        training_manifest
    )

    required = {
        "slice_path",
        "volume",
        "slice",
        "label0_pxl_cnt",
        "label1_pxl_cnt",
        "label2_pxl_cnt",
    }

    missing = sorted(
        required
        - set(
            manifest.columns
        )
    )

    if missing:
        raise PairSpaceAuditError(
            "Training manifest is missing required column(s): "
            + ", ".join(
                missing
            )
        )

    whole_tumor = (
        manifest[
            "label0_pxl_cnt"
        ]
        + manifest[
            "label1_pxl_cnt"
        ]
        + manifest[
            "label2_pxl_cnt"
        ]
    )

    donors = manifest.loc[
        whole_tumor
        >= min_tumor_pixels
    ].copy()

    donors[
        "whole_tumor_pixels"
    ] = whole_tumor.loc[
        donors.index
    ].astype(
        np.int64
    )

    donors = donors.reset_index(
        drop=True
    )

    if donors.empty:
        raise PairSpaceAuditError(
            "No training donors satisfy the tumor-pixel threshold."
        )

    kept_rows = []
    kept_masks = []

    print()
    print(
        "Loading training donors and applying mask-margin rule..."
    )

    for index, row in enumerate(
        donors.itertuples(
            index=False
        ),
        start=1,
    ):
        h5_path = (
            h5_root
            / Path(
                str(
                    row.slice_path
                )
            ).name
        )

        if not h5_path.is_file():
            raise FileNotFoundError(
                "Training donor H5 file not found:\n"
                f"{h5_path}"
            )

        (
            _donor_image,
            donor_mask,
            _donor_condition,
        ) = load_h5_full(
            h5_path,
            image_channel=donor_image_channel,
        )

        observed_pixels = int(
            donor_mask.sum().item()
        )

        expected_pixels = int(
            row.whole_tumor_pixels
        )

        if (
            observed_pixels
            != expected_pixels
        ):
            raise PairSpaceAuditError(
                "Training-manifest whole-tumor count does not "
                "match loaded whole-tumor mask.\n"
                f"H5: {h5_path}\n"
                f"Manifest: {expected_pixels}\n"
                f"Loaded:   {observed_pixels}"
            )

        has_margin = mask_has_margin(
            donor_mask,
            margin=mask_margin,
        )

        if has_margin:
            row_dict = row._asdict()

            row_dict[
                "donor_h5_path"
            ] = str(
                h5_path.resolve()
            )

            row_dict[
                "mask_has_margin"
            ] = True

            kept_rows.append(
                row_dict
            )

            kept_masks.append(
                donor_mask[
                    0
                ]
                .detach()
                .cpu()
                .numpy()
                .astype(
                    np.uint8,
                    copy=False,
                )
                .reshape(
                    -1
                )
            )

        if (
            index % 1000 == 0
            or index == len(
                donors
            )
        ):
            print(
                f"  Donors checked: "
                f"{index:,} / "
                f"{len(donors):,}",
                flush=True,
            )

    if not kept_rows:
        raise PairSpaceAuditError(
            "No donor masks survived the configured margin rule."
        )

    retained = pd.DataFrame(
        kept_rows
    )

    donor_masks = np.stack(
        kept_masks,
        axis=0,
    )

    retained[
        "loaded_mask_pixels"
    ] = donor_masks.sum(
        axis=1,
        dtype=np.int64,
    )

    return (
        retained,
        donor_masks,
    )


def compute_pair_counts(
    *,
    base_masks: np.ndarray,
    donor_masks: np.ndarray,
    donor_pixel_counts: np.ndarray,
    min_overlap: float,
    base_batch_size: int,
    donor_batch_size: int,
    device: torch.device,
) -> tuple[
    np.ndarray,
    np.ndarray,
    int,
]:
    base_count = int(
        base_masks.shape[
            0
        ]
    )

    donor_count = int(
        donor_masks.shape[
            0
        ]
    )

    if base_masks.shape[1] != donor_masks.shape[1]:
        raise PairSpaceAuditError(
            "Flattened base and donor masks have incompatible shapes."
        )

    compatible_per_base = np.zeros(
        base_count,
        dtype=np.int64,
    )

    compatible_per_donor = np.zeros(
        donor_count,
        dtype=np.int64,
    )

    total_pairs = 0

    total_base_batches = (
        base_count
        + base_batch_size
        - 1
    ) // base_batch_size

    total_donor_batches = (
        donor_count
        + donor_batch_size
        - 1
    ) // donor_batch_size

    print()
    print(
        "Computing complete base-donor compatibility space..."
    )

    print(
        f"Device                    : {device}"
    )

    print(
        f"Base batches              : {total_base_batches:,}"
    )

    print(
        f"Donor batches             : {total_donor_batches:,}"
    )

    print(
        f"Matrix block evaluations  : "
        f"{total_base_batches * total_donor_batches:,}"
    )

    with torch.no_grad():
        for base_batch_number, base_start in enumerate(
            range(
                0,
                base_count,
                base_batch_size,
            ),
            start=1,
        ):
            base_end = min(
                base_start
                + base_batch_size,
                base_count,
            )

            base_tensor = torch.from_numpy(
                base_masks[
                    base_start:base_end
                ]
            ).to(
                device=device,
                dtype=torch.float32,
            )

            base_batch_counts = np.zeros(
                base_end
                - base_start,
                dtype=np.int64,
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

                donor_tensor = torch.from_numpy(
                    donor_masks[
                        donor_start:donor_end
                    ]
                ).to(
                    device=device,
                    dtype=torch.float32,
                )

                overlap_pixels = torch.matmul(
                    base_tensor,
                    donor_tensor.transpose(
                        0,
                        1,
                    ),
                )

                threshold = torch.from_numpy(
                    donor_pixel_counts[
                        donor_start:donor_end
                    ].astype(
                        np.float32
                    )
                    * float(
                        min_overlap
                    )
                ).to(
                    device=device
                )

                compatible = (
                    overlap_pixels
                    >= threshold[
                        None,
                        :,
                    ]
                )

                block_base_counts = (
                    compatible.sum(
                        dim=1
                    )
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(
                        np.int64,
                        copy=False,
                    )
                )

                block_donor_counts = (
                    compatible.sum(
                        dim=0
                    )
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(
                        np.int64,
                        copy=False,
                    )
                )

                base_batch_counts += (
                    block_base_counts
                )

                compatible_per_donor[
                    donor_start:donor_end
                ] += block_donor_counts

                block_pairs = int(
                    block_base_counts.sum()
                )

                total_pairs += (
                    block_pairs
                )

                del donor_tensor
                del overlap_pixels
                del compatible

            compatible_per_base[
                base_start:base_end
            ] = base_batch_counts

            del base_tensor

            print(
                f"  Base batch "
                f"{base_batch_number:,} / "
                f"{total_base_batches:,} complete "
                f"({base_end:,} / {base_count:,} bases)",
                flush=True,
            )

    if int(
        compatible_per_base.sum()
    ) != total_pairs:
        raise PairSpaceAuditError(
            "Base-count total does not match accumulated pair total."
        )

    if int(
        compatible_per_donor.sum()
    ) != total_pairs:
        raise PairSpaceAuditError(
            "Donor-count total does not match accumulated pair total."
        )

    return (
        compatible_per_base,
        compatible_per_donor,
        total_pairs,
    )


def refuse_existing_outputs(
    output_dir: Path,
) -> None:
    existing = [
        output_dir
        / BASE_OUTPUT_NAME,
        output_dir
        / DONOR_OUTPUT_NAME,
        output_dir
        / SUMMARY_OUTPUT_NAME,
    ]

    existing = [
        path
        for path in existing
        if path.exists()
    ]

    if existing:
        raise PairSpaceAuditError(
            "Refusing to overwrite existing audit output(s):\n"
            + "\n".join(
                str(
                    path
                )
                for path in existing
            )
        )


def main() -> None:
    args = parse_args()

    validation_dataset_path = (
        args.validation_dataset
        .expanduser()
        .resolve()
    )

    screening_csv = (
        args.screening_csv
        .expanduser()
        .resolve()
    )

    training_manifest = (
        args.training_manifest
        .expanduser()
        .resolve()
    )

    h5_root = (
        args.h5_root
        .expanduser()
        .resolve()
    )

    output_dir = (
        args.output_dir
        .expanduser()
        .resolve()
    )

    if not screening_csv.is_file():
        raise FileNotFoundError(
            f"Screening CSV not found:\n{screening_csv}"
        )

    if not training_manifest.is_file():
        raise FileNotFoundError(
            f"Training manifest not found:\n{training_manifest}"
        )

    if not h5_root.is_dir():
        raise FileNotFoundError(
            f"H5 root not found:\n{h5_root}"
        )

    if args.min_tumor_pixels < 0:
        raise PairSpaceAuditError(
            "--min-tumor-pixels must be non-negative."
        )

    if not (
        0.0
        <= args.brain_threshold
        <= 1.0
    ):
        raise PairSpaceAuditError(
            "--brain-threshold must lie in [0, 1]."
        )

    if not (
        0.0
        <= args.min_overlap
        <= 1.0
    ):
        raise PairSpaceAuditError(
            "--min-overlap must lie in [0, 1]."
        )

    if args.mask_margin < 0:
        raise PairSpaceAuditError(
            "--mask-margin must be non-negative."
        )

    if args.base_batch_size <= 0:
        raise PairSpaceAuditError(
            "--base-batch-size must be positive."
        )

    if args.donor_batch_size <= 0:
        raise PairSpaceAuditError(
            "--donor-batch-size must be positive."
        )

    device = resolve_device(
        args.device
    )

    validation_dataset = (
        load_validation_dataset_specification(
            validation_dataset_path
        )
    )

    print()
    print(
        "=" * 78
    )

    print(
        "EXTERNAL BR-LoRA PAIR-SPACE AUDIT"
    )

    print(
        "=" * 78
    )

    print(
        "External tumor-free rule : "
        "predicted_tumor_pixels == 0"
    )

    print(
        f"Donor tumor threshold    : "
        f"{args.min_tumor_pixels}"
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
        f"Mask margin              : "
        f"{args.mask_margin}"
    )

    print(
        f"External modality        : "
        f"{args.external_modality}"
    )

    print(
        f"Donor image channel      : "
        f"{args.donor_image_channel}"
    )

    print(
        f"Computation device       : "
        f"{device}"
    )

    print(
        "=" * 78
    )

    bases, base_masks = (
        load_external_bases(
            screening_csv=screening_csv,
            validation_dataset=validation_dataset,
            modality=str(
                args.external_modality
            ).lower(),
            brain_threshold=float(
                args.brain_threshold
            ),
        )
    )

    donors, donor_masks = (
        load_eligible_donors(
            training_manifest=training_manifest,
            h5_root=h5_root,
            donor_image_channel=int(
                args.donor_image_channel
            ),
            min_tumor_pixels=int(
                args.min_tumor_pixels
            ),
            mask_margin=int(
                args.mask_margin
            ),
        )
    )

    (
        compatible_per_base,
        compatible_per_donor,
        total_pairs,
    ) = compute_pair_counts(
        base_masks=base_masks,
        donor_masks=donor_masks,
        donor_pixel_counts=donors[
            "loaded_mask_pixels"
        ].to_numpy(
            dtype=np.int64
        ),
        min_overlap=float(
            args.min_overlap
        ),
        base_batch_size=int(
            args.base_batch_size
        ),
        donor_batch_size=int(
            args.donor_batch_size
        ),
        device=device,
    )

    bases[
        "compatible_donor_count"
    ] = compatible_per_base

    donors[
        "compatible_external_base_count"
    ] = compatible_per_donor

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    refuse_existing_outputs(
        output_dir
    )

    base_output = (
        output_dir
        / BASE_OUTPUT_NAME
    )

    donor_output = (
        output_dir
        / DONOR_OUTPUT_NAME
    )

    summary_output = (
        output_dir
        / SUMMARY_OUTPUT_NAME
    )

    bases.to_csv(
        base_output,
        index=False,
    )

    donors.to_csv(
        donor_output,
        index=False,
    )

    bases_with_pair = int(
        (
            compatible_per_base
            > 0
        ).sum()
    )

    donors_with_pair = int(
        (
            compatible_per_donor
            > 0
        ).sum()
    )

    original_training_manifest = pd.read_csv(
        training_manifest
    )

    original_whole_tumor = (
        original_training_manifest[
            "label0_pxl_cnt"
        ]
        + original_training_manifest[
            "label1_pxl_cnt"
        ]
        + original_training_manifest[
            "label2_pxl_cnt"
        ]
    )

    donor_before_margin = int(
        (
            original_whole_tumor
            >= args.min_tumor_pixels
        ).sum()
    )

    summary = {
        "audit_definition": {
            "external_base":
                (
                    "tumor_free_candidate == True and "
                    "predicted_tumor_pixels == 0"
                ),

            "donor":
                (
                    "whole_tumor_pixels >= "
                    f"{args.min_tumor_pixels}"
                ),

            "brain_threshold":
                float(
                    args.brain_threshold
                ),

            "minimum_mask_inside_brain_fraction":
                float(
                    args.min_overlap
                ),

            "mask_margin_pixels":
                int(
                    args.mask_margin
                ),

            "external_modality":
                str(
                    args.external_modality
                ).lower(),

            "donor_image_channel":
                int(
                    args.donor_image_channel
                ),
        },

        "external_tumor_free_bases":
            int(
                len(
                    bases
                )
            ),

        "training_donors_before_margin":
            donor_before_margin,

        "training_donors_after_margin":
            int(
                len(
                    donors
                )
            ),

        "external_bases_with_at_least_one_compatible_donor":
            bases_with_pair,

        "external_bases_with_zero_compatible_donors":
            int(
                len(
                    bases
                )
                - bases_with_pair
            ),

        "donors_with_at_least_one_compatible_external_base":
            donors_with_pair,

        "donors_with_zero_compatible_external_bases":
            int(
                len(
                    donors
                )
                - donors_with_pair
            ),

        "total_compatible_base_donor_pairs":
            int(
                total_pairs
            ),

        "compatible_donors_per_external_base":
            describe_integer_counts(
                compatible_per_base
            ),

        "compatible_external_bases_per_donor":
            describe_integer_counts(
                compatible_per_donor
            ),

        "external_brain_pixels":
            describe_integer_counts(
                bases[
                    "brain_pixels"
                ].to_numpy(
                    dtype=np.int64
                )
            ),

        "paths": {
            "validation_dataset":
                str(
                    validation_dataset_path
                ),

            "screening_csv":
                str(
                    screening_csv
                ),

            "training_manifest":
                str(
                    training_manifest
                ),

            "h5_root":
                str(
                    h5_root
                ),
        },

        "provenance": {
            "git_commit":
                resolve_git_commit(),

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

            "training_manifest_sha256":
                sha256_file(
                    training_manifest
                ),

            "audited_at_utc":
                datetime.now(
                    timezone.utc
                ).isoformat(),

            "matrix_device":
                str(
                    device
                ),

            "base_batch_size":
                int(
                    args.base_batch_size
                ),

            "donor_batch_size":
                int(
                    args.donor_batch_size
                ),
        },
    }

    with summary_output.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
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
        "EXTERNAL PAIR-SPACE AUDIT: PASS"
    )

    print(
        "=" * 78
    )

    print(
        f"External bases           : "
        f"{len(bases):,}"
    )

    print(
        f"Donors before margin     : "
        f"{donor_before_margin:,}"
    )

    print(
        f"Donors after margin      : "
        f"{len(donors):,}"
    )

    print(
        f"Bases with >=1 donor     : "
        f"{bases_with_pair:,}"
    )

    print(
        f"Bases with zero donors   : "
        f"{len(bases) - bases_with_pair:,}"
    )

    print(
        f"Donors with >=1 base     : "
        f"{donors_with_pair:,}"
    )

    print(
        f"Compatible pairs         : "
        f"{total_pairs:,}"
    )

    print()
    print(
        f"Base counts              : "
        f"{base_output}"
    )

    print(
        f"Donor counts             : "
        f"{donor_output}"
    )

    print(
        f"Summary                  : "
        f"{summary_output}"
    )

    print(
        "=" * 78
    )


if __name__ == "__main__":
    main()
