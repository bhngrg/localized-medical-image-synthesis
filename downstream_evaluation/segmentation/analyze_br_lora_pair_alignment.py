#!/usr/bin/env python3

"""
Read-only diagnostic for BR-LoRA base/donor spatial compatibility.

For each frozen synthetic-library case, this script reconstructs the external
base brain mask, validates the stored donor/transferred mask, and summarizes:

1. base/donor spatial geometry;
2. BR-LoRA posterior variability inside the transferred lesion;
3. intensity continuity across the hard-composite lesion boundary.

The accepted BR-LoRA library is treated as read-only. No synthetic artifacts,
manifests, checkpoints, or downstream models are modified.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(REPO_ROOT),
    )

import numpy as np
import pandas as pd
import torch
import yaml

from scipy.ndimage import (
    binary_erosion,
    distance_transform_edt,
)

from src.data import (
    load_validation_dataset_specification,
    load_validation_slice,
)
from src.data.preprocessing import get_brain_mask


DEFAULT_MANIFEST = Path(
    "downstream_evaluation/manifests/"
    "br_lora_library_design_10000/"
    "br_lora_library_design_10000.csv"
)

DEFAULT_FOLDERS = Path("data/folders.yaml")

DEFAULT_OUTPUT_DIR = Path(
    "results/downstream_segmentation/"
    "synthetic_pair_diagnostics/br_lora"
)

POSTERIOR_SAMPLES = 100
IMAGE_SHAPE = (240, 240)
COMPATIBILITY_THRESHOLD = 0.80
FEATHER_WIDTHS = (1, 2, 3, 4, 5)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze spatial compatibility and posterior variability "
            "for the frozen BR-LoRA synthetic library."
        )
    )

    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
    )
    parser.add_argument(
        "--folders",
        type=Path,
        default=DEFAULT_FOLDERS,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional number of cases for a smoke test.",
    )
    parser.add_argument(
        "--inner-boundary-width",
        type=int,
        default=5,
        help=(
            "Width in pixels of the lesion-side boundary band used "
            "for posterior-variability summaries."
        ),
    )

    return parser.parse_args()


def load_folders(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        payload = yaml.safe_load(file)

    if not isinstance(payload, dict):
        raise ValueError("folders.yaml must contain a mapping.")

    return payload


def resolve_path(value: str | Path) -> Path:
    return Path(value).expanduser().resolve()


def centroid(mask: np.ndarray) -> tuple[float, float]:
    coordinates = np.argwhere(mask)

    if coordinates.size == 0:
        return float("nan"), float("nan")

    y, x = coordinates.mean(axis=0)

    return float(x), float(y)


def summarize_values(
    values: np.ndarray,
    prefix: str,
) -> dict[str, float]:
    if values.size == 0:
        return {
            f"{prefix}_mean": float("nan"),
            f"{prefix}_median": float("nan"),
            f"{prefix}_p95": float("nan"),
            f"{prefix}_max": float("nan"),
        }

    return {
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_median": float(np.median(values)),
        f"{prefix}_p95": float(np.quantile(values, 0.95)),
        f"{prefix}_max": float(np.max(values)),
    }


def boundary_crossing_differences(
    image: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """
    Return absolute intensity differences across 4-connected mask edges.

    Each horizontal or vertical adjacency for which exactly one endpoint is
    inside the transferred mask is counted once.
    """

    if image.shape != mask.shape:
        raise ValueError(
            "Boundary image and mask shapes do not match."
        )

    horizontal_crossing = (
        mask[:, :-1]
        != mask[:, 1:]
    )

    horizontal_difference = np.abs(
        image[:, :-1]
        - image[:, 1:]
    )[
        horizontal_crossing
    ]

    vertical_crossing = (
        mask[:-1, :]
        != mask[1:, :]
    )

    vertical_difference = np.abs(
        image[:-1, :]
        - image[1:, :]
    )[
        vertical_crossing
    ]

    return np.concatenate(
        [
            horizontal_difference,
            vertical_difference,
        ]
    )


def safe_ratio(
    numerator: float,
    denominator: float,
) -> float:
    """Return a finite descriptive ratio when its denominator is positive."""

    if (
        not np.isfinite(numerator)
        or not np.isfinite(denominator)
        or denominator <= 0
    ):
        return float("nan")

    return float(
        numerator
        / denominator
    )


def make_inner_feather_weights(
    mask: np.ndarray,
    width: int,
) -> np.ndarray:
    """
    Construct deterministic inner-only feathering weights.

    Outside-mask pixels receive weight zero exactly. Inside the mask,
    prediction weight increases with distance from the mask boundary
    until reaching one in the lesion interior.
    """

    if width <= 0:
        raise ValueError(
            "Feather width must be positive."
        )

    if mask.ndim != 2:
        raise ValueError(
            "Feather mask must be two-dimensional."
        )

    distance_inside = distance_transform_edt(
        mask
    )

    weights = np.zeros(
        mask.shape,
        dtype=np.float32,
    )

    weights[mask] = np.minimum(
        distance_inside[mask]
        / float(width),
        1.0,
    )

    return weights


def main() -> None:
    args = parse_args()

    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive.")

    if args.inner_boundary_width <= 0:
        raise ValueError(
            "--inner-boundary-width must be positive."
        )

    folders = load_folders(args.folders)

    library_root = resolve_path(
        folders["br_lora_library_root"]
    )

    validation_yaml = resolve_path(
        folders["yaml_validation_dataset_path"]
    )

    manifest_path = resolve_path(args.manifest)
    output_dir = resolve_path(args.output_dir)

    if not library_root.is_dir():
        raise FileNotFoundError(
            f"BR-LoRA library root not found: {library_root}"
        )

    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Manifest not found: {manifest_path}"
        )

    validation_dataset = load_validation_dataset_specification(
        validation_yaml
    )

    manifest = pd.read_csv(manifest_path)

    if args.limit is not None:
        manifest = manifest.iloc[: args.limit].copy()

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows: list[dict[str, object]] = []

    for position, row in enumerate(
        manifest.itertuples(index=False),
        start=1,
    ):
        case_status = str(
            row.case_status
        )

        if case_status not in (
            "already_generated",
            "planned",
        ):
            raise ValueError(
                f"{row.library_case_id}: unsupported case_status "
                f"{case_status!r}."
            )

        artifact_case_id = str(
            row.library_case_id
        )

        case_directory = (
            library_root
            / "batches"
            / str(row.batch_id)
            / artifact_case_id
        )

        posterior_path = (
            case_directory
            / "posterior_samples.pt"
        )

        if not posterior_path.is_file():
            raise FileNotFoundError(
                f"Posterior artifact not found: {posterior_path}"
            )

        payload = torch.load(
            posterior_path,
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )

        prediction_samples = payload[
            "prediction_samples"
        ]

        stored_base = payload["base_image"]
        transferred_mask = payload[
            "transferred_mask"
        ]
        stored_donor = payload[
            "donor_image"
        ]

        expected_shape = (
            POSTERIOR_SAMPLES,
            1,
            1,
            *IMAGE_SHAPE,
        )

        if tuple(prediction_samples.shape) != expected_shape:
            raise ValueError(
                f"{row.library_case_id}: unexpected posterior shape "
                f"{tuple(prediction_samples.shape)}"
            )

        if tuple(stored_base.shape) != (1, *IMAGE_SHAPE):
            raise ValueError(
                f"{row.library_case_id}: unexpected base-image shape."
            )

        if tuple(transferred_mask.shape) != (1, *IMAGE_SHAPE):
            raise ValueError(
                f"{row.library_case_id}: unexpected transferred-mask shape."
            )

        if tuple(stored_donor.shape) != (1, *IMAGE_SHAPE):
            raise ValueError(
                f"{row.library_case_id}: unexpected donor-image shape."
            )

        if not torch.isfinite(stored_donor).all():
            raise ValueError(
                f"{row.library_case_id}: donor image contains "
                "non-finite values."
            )

        external = load_validation_slice(
            validation_dataset,
            subject_numeric_id=int(
                row.external_subject_numeric_id
            ),
            slice_index=int(
                row.external_slice_index
            ),
            modality=str(
                row.external_modality
            ),
        )

        if not torch.equal(
            stored_base,
            external.image,
        ):
            raise ValueError(
                f"{row.library_case_id}: stored base image does not "
                "exactly match reconstructed external base."
            )

        base_image = (
            stored_base
            .squeeze(0)
            .to(dtype=torch.float32)
        )

        mask_tensor = (
            transferred_mask
            .squeeze(0)
            .to(dtype=torch.float32)
        )

        unique_mask = torch.unique(mask_tensor)

        if not torch.all(
            (unique_mask == 0)
            | (unique_mask == 1)
        ):
            raise ValueError(
                f"{row.library_case_id}: transferred mask is not binary."
            )

        mask = mask_tensor.numpy() > 0

        observed_mask_pixels = int(mask.sum())

        if observed_mask_pixels != int(
            row.donor_mask_pixels
        ):
            raise ValueError(
                f"{row.library_case_id}: donor-mask pixel count mismatch."
            )

        brain_tensor = get_brain_mask(
            base_image.unsqueeze(0),
            threshold=0.05,
        )

        brain = (
            brain_tensor
            .squeeze()
            .cpu()
            .numpy()
            > 0
        )

        brain_pixels = int(brain.sum())

        overlap_pixels = int(
            np.logical_and(
                mask,
                brain,
            ).sum()
        )

        containment = (
            overlap_pixels
            / observed_mask_pixels
        )

        if containment + 1e-12 < COMPATIBILITY_THRESHOLD:
            raise ValueError(
                f"{row.library_case_id}: recomputed containment "
                f"{containment:.6f} is below "
                f"{COMPATIBILITY_THRESHOLD:.2f}."
            )

        outside_brain_pixels = (
            observed_mask_pixels
            - overlap_pixels
        )

        lesion_x, lesion_y = centroid(mask)
        brain_x, brain_y = centroid(brain)

        brain_coordinates = np.argwhere(brain)

        brain_height = (
            int(
                brain_coordinates[:, 0].max()
                - brain_coordinates[:, 0].min()
                + 1
            )
        )

        brain_width = (
            int(
                brain_coordinates[:, 1].max()
                - brain_coordinates[:, 1].min()
                + 1
            )
        )

        brain_diagonal = float(
            np.hypot(
                brain_width,
                brain_height,
            )
        )

        centroid_distance = float(
            np.hypot(
                lesion_x - brain_x,
                lesion_y - brain_y,
            )
        )

        centroid_distance_normalized = (
            centroid_distance / brain_diagonal
        )

        # Distance from each in-brain lesion pixel to the nearest
        # non-brain pixel. Zero is retained for lesion pixels that
        # already fall outside the reconstructed base brain.
        distance_inside_brain = distance_transform_edt(
            brain
        )

        lesion_boundary_distances = np.where(
            brain[mask],
            distance_inside_brain[mask],
            0.0,
        )

        eroded_mask = binary_erosion(
            mask,
            iterations=args.inner_boundary_width,
            border_value=0,
        )

        inner_boundary = (
            mask
            & ~eroded_mask
        )

        prediction_samples = (
            prediction_samples
            .squeeze(1)
            .squeeze(1)
            .to(dtype=torch.float32)
        )

        posterior_variance = (
            prediction_samples.var(
                dim=0,
                correction=0,
            )
            .numpy()
        )

        inside_variance = posterior_variance[
            mask
        ]

        inner_boundary_variance = posterior_variance[
            inner_boundary
        ]

        posterior_mean = prediction_samples.mean(
            dim=0
        )

        absolute_draw_deviation = (
            prediction_samples
            - posterior_mean.unsqueeze(0)
        ).abs().mean(dim=0).numpy()

        posterior_mean_numpy = (
            posterior_mean
            .cpu()
            .numpy()
        )

        base_numpy = (
            base_image
            .cpu()
            .numpy()
        )

        donor_numpy = (
            stored_donor
            .squeeze(0)
            .to(dtype=torch.float32)
            .cpu()
            .numpy()
        )

        composite_mean = np.where(
            mask,
            posterior_mean_numpy,
            base_numpy,
        )

        composite_boundary_differences = (
            boundary_crossing_differences(
                composite_mean,
                mask,
            )
        )

        donor_boundary_differences = (
            boundary_crossing_differences(
                donor_numpy,
                mask,
            )
        )

        if composite_boundary_differences.size == 0:
            raise ValueError(
                f"{row.library_case_id}: transferred mask has no "
                "boundary-crossing edges."
            )

        if (
            donor_boundary_differences.size
            != composite_boundary_differences.size
        ):
            raise ValueError(
                f"{row.library_case_id}: donor and composite boundary "
                "edge counts disagree."
            )

        composite_boundary_summary = summarize_values(
            composite_boundary_differences,
            "composite_boundary_abs_difference",
        )

        donor_boundary_summary = summarize_values(
            donor_boundary_differences,
            "donor_boundary_abs_difference",
        )

        feather_summaries: dict[str, float | int] = {}

        for feather_width in FEATHER_WIDTHS:
            feather_weights = make_inner_feather_weights(
                mask,
                feather_width,
            )

            feathered_composite = (
                base_numpy
                + feather_weights
                * (
                    posterior_mean_numpy
                    - base_numpy
                )
            )

            # Inner-only feathering must retain the original
            # outside-mask preservation guarantee exactly.
            if not np.array_equal(
                feathered_composite[~mask],
                base_numpy[~mask],
            ):
                raise ValueError(
                    f"{row.library_case_id}: feather width "
                    f"{feather_width} changed pixels outside the mask."
                )

            feathered_boundary_differences = (
                boundary_crossing_differences(
                    feathered_composite,
                    mask,
                )
            )

            if (
                feathered_boundary_differences.size
                != composite_boundary_differences.size
            ):
                raise ValueError(
                    f"{row.library_case_id}: feather width "
                    f"{feather_width} changed the number of "
                    "boundary-crossing edges."
                )

            prefix = (
                f"feather_{feather_width}px_"
                "boundary_abs_difference"
            )

            feathered_boundary_summary = summarize_values(
                feathered_boundary_differences,
                prefix,
            )

            feather_summaries.update(
                feathered_boundary_summary
            )

            feather_mean = (
                feathered_boundary_summary[
                    f"{prefix}_mean"
                ]
            )

            feather_median = (
                feathered_boundary_summary[
                    f"{prefix}_median"
                ]
            )

            feather_summaries[
                f"feather_{feather_width}px_"
                "to_donor_boundary_mean_ratio"
            ] = safe_ratio(
                feather_mean,
                donor_boundary_summary[
                    "donor_boundary_abs_difference_mean"
                ],
            )

            feather_summaries[
                f"feather_{feather_width}px_"
                "to_donor_boundary_median_ratio"
            ] = safe_ratio(
                feather_median,
                donor_boundary_summary[
                    "donor_boundary_abs_difference_median"
                ],
            )

            correction = np.abs(
                feathered_composite
                - composite_mean
            )

            affected = (
                mask
                & (
                    feather_weights
                    < 1.0
                )
            )

            feather_summaries[
                f"feather_{feather_width}px_"
                "affected_pixels"
            ] = int(
                affected.sum()
            )

            feather_summaries.update(
                summarize_values(
                    correction[mask],
                    (
                        f"feather_{feather_width}px_"
                        "abs_correction_inside_mask"
                    ),
                )
            )

            feather_summaries.update(
                summarize_values(
                    correction[affected],
                    (
                        f"feather_{feather_width}px_"
                        "abs_correction_affected_band"
                    ),
                )
            )

        record: dict[str, object] = {
            "library_index": int(
                row.library_index
            ),
            "library_case_id": str(
                row.library_case_id
            ),
            "batch_id": str(
                row.batch_id
            ),
            "source_case_id": str(
                row.source_case_id
            ),
            "external_subject_name": str(
                row.external_subject_name
            ),
            "external_subject_numeric_id": int(
                row.external_subject_numeric_id
            ),
            "external_slice_index": int(
                row.external_slice_index
            ),
            "donor_volume": int(
                row.donor_volume
            ),
            "donor_slice_index": int(
                row.donor_slice_index
            ),
            "donor_h5_file": str(
                row.donor_h5_file
            ),
            "donor_mask_pixels": observed_mask_pixels,
            "base_brain_pixels": brain_pixels,
            "mask_inside_base_brain_pixels": overlap_pixels,
            "mask_outside_base_brain_pixels": outside_brain_pixels,
            "mask_inside_base_brain_fraction": float(
                containment
            ),
            "mask_outside_base_brain_fraction": float(
                1.0 - containment
            ),
            "lesion_centroid_x": lesion_x,
            "lesion_centroid_y": lesion_y,
            "base_brain_centroid_x": brain_x,
            "base_brain_centroid_y": brain_y,
            "lesion_to_brain_centroid_distance_pixels": (
                centroid_distance
            ),
            "lesion_to_brain_centroid_distance_normalized": (
                centroid_distance_normalized
            ),
            "lesion_brain_boundary_distance_mean": float(
                np.mean(
                    lesion_boundary_distances
                )
            ),
            "lesion_brain_boundary_distance_min": float(
                np.min(
                    lesion_boundary_distances
                )
            ),
            "lesion_brain_boundary_distance_p05": float(
                np.quantile(
                    lesion_boundary_distances,
                    0.05,
                )
            ),
            "posterior_samples": POSTERIOR_SAMPLES,
            "inner_boundary_width_pixels": int(
                args.inner_boundary_width
            ),
            "inner_boundary_pixels": int(
                inner_boundary.sum()
            ),
            "boundary_crossing_edges": int(
                composite_boundary_differences.size
            ),
        }

        record.update(
            summarize_values(
                inside_variance,
                "posterior_variance_inside_mask",
            )
        )

        record.update(
            summarize_values(
                inner_boundary_variance,
                "posterior_variance_inner_boundary",
            )
        )

        record.update(
            summarize_values(
                absolute_draw_deviation[mask],
                "posterior_abs_deviation_inside_mask",
            )
        )

        record.update(
            summarize_values(
                absolute_draw_deviation[inner_boundary],
                "posterior_abs_deviation_inner_boundary",
            )
        )

        record.update(
            composite_boundary_summary
        )

        record.update(
            donor_boundary_summary
        )

        record[
            "composite_to_donor_boundary_mean_ratio"
        ] = safe_ratio(
            composite_boundary_summary[
                "composite_boundary_abs_difference_mean"
            ],
            donor_boundary_summary[
                "donor_boundary_abs_difference_mean"
            ],
        )

        record[
            "composite_to_donor_boundary_median_ratio"
        ] = safe_ratio(
            composite_boundary_summary[
                "composite_boundary_abs_difference_median"
            ],
            donor_boundary_summary[
                "donor_boundary_abs_difference_median"
            ],
        )

        record.update(
            feather_summaries
        )

        rows.append(record)

        if (
            position == 1
            or position % 25 == 0
            or position == len(manifest)
        ):
            print(
                f"Processed {position:,}/{len(manifest):,} cases"
            )

    result = pd.DataFrame(rows)

    csv_path = (
        output_dir
        / "pair_alignment_metrics.csv"
    )

    result.to_csv(
        csv_path,
        index=False,
    )

    summary = {
        "cases": int(len(result)),
        "compatibility_threshold": (
            COMPATIBILITY_THRESHOLD
        ),
        "posterior_samples_per_case": (
            POSTERIOR_SAMPLES
        ),
        "inner_boundary_width_pixels": int(
            args.inner_boundary_width
        ),
        "mask_inside_base_brain_fraction": {
            "min": float(
                result[
                    "mask_inside_base_brain_fraction"
                ].min()
            ),
            "median": float(
                result[
                    "mask_inside_base_brain_fraction"
                ].median()
            ),
            "mean": float(
                result[
                    "mask_inside_base_brain_fraction"
                ].mean()
            ),
            "max": float(
                result[
                    "mask_inside_base_brain_fraction"
                ].max()
            ),
        },
        "cases_with_any_mask_outside_brain": int(
            (
                result[
                    "mask_outside_base_brain_pixels"
                ]
                > 0
            ).sum()
        ),
    }

    summary_path = (
        output_dir
        / "pair_alignment_summary.json"
    )

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
        )
        file.write("\n")

    print()
    print("Diagnostic complete.")
    print(f"Metrics : {csv_path}")
    print(f"Summary : {summary_path}")


if __name__ == "__main__":
    main()
