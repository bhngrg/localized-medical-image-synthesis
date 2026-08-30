#!/usr/bin/env python3

from __future__ import annotations

import csv
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from downstream_evaluation.segmentation.model import VanillaUNet
from src.data.preprocessing import (
    normalize_image_channel,
    standardize_nifti_slice,
)


REPO_ROOT = Path(__file__).resolve().parents[2]

MANIFEST_PATH = (
    REPO_ROOT
    / "downstream_evaluation"
    / "manifests"
    / "ucsf_pdgm_external_202_subjects.csv"
)

RAW_ROOT = Path(
    "/scratch/bhanug/archive/UCSF_PDGM/raw/"
    "PKG - UCSF-PDGM Version 5/UCSF-PDGM-v5"
)

OUTPUT_ROOT = (
    REPO_ROOT
    / "outputs"
    / "downstream_segmentation"
    / "ucsf_pdgm_external_202"
)

CHECKPOINTS = {
    "real_only": (
        REPO_ROOT
        / "outputs"
        / "downstream_segmentation"
        / "real_only_seed42_a30_normal_q"
        / "best_model.pt"
    ),
    "real_plus_br_lora_posterior_mean": (
        REPO_ROOT
        / "outputs"
        / "downstream_segmentation"
        / "real_plus_br_lora_seed42_a30_normal_q_rerun"
        / "best_model.pt"
    ),
    "real_plus_br_lora_posterior_sampling": (
        REPO_ROOT
        / "outputs"
        / "downstream_segmentation"
        / "real_plus_br_lora_posterior_seed42_a30_normal_q"
        / "best_model.pt"
    ),
}

BATCH_SIZE = 26
EXPECTED_SHAPE = (240, 240, 155)


def load_manifest() -> list[dict[str, str]]:
    with MANIFEST_PATH.open(newline="") as f:
        rows = list(csv.DictReader(f))

    if len(rows) != 202:
        raise RuntimeError(
            f"Expected 202 UCSF-PDGM subjects, found {len(rows)}."
        )

    return rows


def prepare_subject(
    row: dict[str, str],
) -> tuple[np.ndarray, np.ndarray]:
    flair_path = RAW_ROOT / row["flair_relative_path"]
    mask_path = RAW_ROOT / row["segmentation_relative_path"]

    flair = np.asarray(
        nib.load(flair_path).dataobj,
        dtype=np.float64,
    )

    segmentation = np.asarray(
        nib.load(mask_path).dataobj,
    )

    if flair.shape != EXPECTED_SHAPE:
        raise RuntimeError(
            f"{row['subject_id']} FLAIR shape {flair.shape} "
            f"does not match {EXPECTED_SHAPE}."
        )

    if segmentation.shape != EXPECTED_SHAPE:
        raise RuntimeError(
            f"{row['subject_id']} segmentation shape "
            f"{segmentation.shape} does not match {EXPECTED_SHAPE}."
        )

    images = np.empty(
        (EXPECTED_SHAPE[2], 1, EXPECTED_SHAPE[0], EXPECTED_SHAPE[1]),
        dtype=np.float32,
    )

    masks = np.empty_like(
        images,
        dtype=np.float32,
    )

    for z in range(EXPECTED_SHAPE[2]):
        normalized = normalize_image_channel(
            standardize_nifti_slice(
                flair[:, :, z]
            )
        )

        images[z, 0] = normalized
        masks[z, 0] = (
            segmentation[:, :, z] > 0
        ).astype(np.float32)

    if not np.isfinite(images).all():
        raise RuntimeError(
            f"{row['subject_id']} contains non-finite model inputs."
        )

    return images, masks


def per_slice_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    predictions = (
        torch.sigmoid(logits) >= threshold
    ).float()

    dims = (1, 2, 3)

    intersection = (
        predictions * targets
    ).sum(dim=dims)

    prediction_sum = predictions.sum(
        dim=dims
    )

    target_sum = targets.sum(
        dim=dims
    )

    denominator = prediction_sum + target_sum

    dice = torch.where(
        denominator > 0,
        2.0 * intersection / denominator,
        torch.ones_like(denominator),
    )

    union = (
        predictions
        + targets
        - predictions * targets
    ).sum(dim=dims)

    iou = torch.where(
        union > 0,
        intersection / union,
        torch.ones_like(union),
    )

    return dice, iou


def evaluate_checkpoint(
    experiment_name: str,
    checkpoint_path: Path,
    manifest_rows: list[dict[str, str]],
    device: torch.device,
) -> None:
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
    )

    threshold = float(
        checkpoint.get("threshold", 0.5)
    )

    model = VanillaUNet(
        in_channels=1,
        out_channels=1,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(device)
    model.eval()

    slice_rows = []
    subject_rows = []

    all_dice = []
    all_iou = []
    positive_dice = []
    positive_iou = []

    for subject_index, row in enumerate(
        manifest_rows,
        start=1,
    ):
        subject_id = row["subject_id"]

        images_np, masks_np = prepare_subject(
            row
        )

        subject_dice = []
        subject_iou = []
        subject_positive_dice = []
        subject_positive_iou = []

        subject_intersection = 0.0
        subject_prediction_sum = 0.0
        subject_target_sum = 0.0

        for start in range(
            0,
            EXPECTED_SHAPE[2],
            BATCH_SIZE,
        ):
            stop = min(
                start + BATCH_SIZE,
                EXPECTED_SHAPE[2],
            )

            images = torch.from_numpy(
                images_np[start:stop]
            ).to(
                device,
                non_blocking=True,
            )

            masks = torch.from_numpy(
                masks_np[start:stop]
            ).to(
                device,
                non_blocking=True,
            )

            with torch.no_grad():
                logits = model(images)

                dice, iou = per_slice_metrics(
                    logits,
                    masks,
                    threshold,
                )

                predictions = (
                    torch.sigmoid(logits) >= threshold
                ).float()

                subject_intersection += float(
                    (predictions * masks).sum().item()
                )

                subject_prediction_sum += float(
                    predictions.sum().item()
                )

                subject_target_sum += float(
                    masks.sum().item()
                )

            dice_cpu = dice.cpu().numpy()
            iou_cpu = iou.cpu().numpy()
            positive_cpu = (
                masks.sum(dim=(1, 2, 3)) > 0
            ).cpu().numpy()

            for local_index in range(stop - start):
                z = start + local_index

                d = float(dice_cpu[local_index])
                j = float(iou_cpu[local_index])
                is_positive = bool(
                    positive_cpu[local_index]
                )

                slice_rows.append(
                    {
                        "experiment": experiment_name,
                        "subject_id": subject_id,
                        "slice_index": z,
                        "tumor_positive": is_positive,
                        "dice": d,
                        "iou": j,
                    }
                )

                all_dice.append(d)
                all_iou.append(j)

                subject_dice.append(d)
                subject_iou.append(j)

                if is_positive:
                    positive_dice.append(d)
                    positive_iou.append(j)

                    subject_positive_dice.append(d)
                    subject_positive_iou.append(j)

        subject_denominator = (
            subject_prediction_sum
            + subject_target_sum
        )

        if subject_denominator > 0:
            subject_volume_dice = (
                2.0
                * subject_intersection
                / subject_denominator
            )
        else:
            subject_volume_dice = 1.0

        subject_union = (
            subject_prediction_sum
            + subject_target_sum
            - subject_intersection
        )

        if subject_union > 0:
            subject_volume_iou = (
                subject_intersection
                / subject_union
            )
        else:
            subject_volume_iou = 1.0

        subject_rows.append(
            {
                "experiment": experiment_name,
                "subject_id": subject_id,
                "n_slices": len(subject_dice),
                "n_positive_slices": len(
                    subject_positive_dice
                ),
                "volumetric_dice": float(
                    subject_volume_dice
                ),
                "volumetric_iou": float(
                    subject_volume_iou
                ),
                "mean_dice_all_slices": float(
                    np.mean(subject_dice)
                ),
                "mean_iou_all_slices": float(
                    np.mean(subject_iou)
                ),
                "mean_dice_positive_slices": float(
                    np.mean(subject_positive_dice)
                ),
                "mean_iou_positive_slices": float(
                    np.mean(subject_positive_iou)
                ),
            }
        )

        print(
            f"[{experiment_name}] "
            f"{subject_index:03d}/{len(manifest_rows)} "
            f"{subject_id} "
            f"positive_slices={len(subject_positive_dice)}"
        )

    output_dir = OUTPUT_ROOT / experiment_name
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    slice_path = output_dir / "slice_metrics.csv"
    subject_path = output_dir / "subject_metrics.csv"
    summary_path = output_dir / "summary.csv"

    with slice_path.open(
        "w",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(
                slice_rows[0].keys()
            ),
        )
        writer.writeheader()
        writer.writerows(slice_rows)

    with subject_path.open(
        "w",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(
                subject_rows[0].keys()
            ),
        )
        writer.writeheader()
        writer.writerows(subject_rows)

    summary_row = {
        "experiment": experiment_name,
        "checkpoint_path": str(
            checkpoint_path.relative_to(
                REPO_ROOT
            )
        ),
        "checkpoint_epoch": int(
            checkpoint["epoch"]
        ),
        "threshold": threshold,
        "n_subjects": len(manifest_rows),
        "n_slices": len(all_dice),
        "n_positive_slices": len(
            positive_dice
        ),
        "mean_dice_all_slices": float(
            np.mean(all_dice)
        ),
        "mean_iou_all_slices": float(
            np.mean(all_iou)
        ),
        "mean_dice_positive_slices": float(
            np.mean(positive_dice)
        ),
        "mean_iou_positive_slices": float(
            np.mean(positive_iou)
        ),
        "mean_subject_volumetric_dice": float(
            np.mean(
                [
                    r["volumetric_dice"]
                    for r in subject_rows
                ]
            )
        ),
        "mean_subject_volumetric_iou": float(
            np.mean(
                [
                    r["volumetric_iou"]
                    for r in subject_rows
                ]
            )
        ),
        "mean_subject_dice_all_slices": float(
            np.mean(
                [
                    r["mean_dice_all_slices"]
                    for r in subject_rows
                ]
            )
        ),
        "mean_subject_iou_all_slices": float(
            np.mean(
                [
                    r["mean_iou_all_slices"]
                    for r in subject_rows
                ]
            )
        ),
        "mean_subject_dice_positive_slices": float(
            np.mean(
                [
                    r["mean_dice_positive_slices"]
                    for r in subject_rows
                ]
            )
        ),
        "mean_subject_iou_positive_slices": float(
            np.mean(
                [
                    r["mean_iou_positive_slices"]
                    for r in subject_rows
                ]
            )
        ),
    }

    with summary_path.open(
        "w",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(
                summary_row.keys()
            ),
        )
        writer.writeheader()
        writer.writerow(summary_row)

    print()
    print(f"Completed: {experiment_name}")
    print(
        "Mean positive-slice Dice:",
        summary_row[
            "mean_dice_positive_slices"
        ],
    )
    print(
        "Mean subject positive-slice Dice:",
        summary_row[
            "mean_subject_dice_positive_slices"
        ],
    )
    print(
        "Mean subject volumetric Dice:",
        summary_row[
            "mean_subject_volumetric_dice"
        ],
    )
    print("Summary:", summary_path)
    print()


def main() -> None:
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print("Device:", device)
    print("External manifest:", MANIFEST_PATH)
    print("Raw root:", RAW_ROOT)
    print()

    manifest_rows = load_manifest()

    for experiment_name, checkpoint_path in CHECKPOINTS.items():
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_path}"
            )

        evaluate_checkpoint(
            experiment_name=experiment_name,
            checkpoint_path=checkpoint_path,
            manifest_rows=manifest_rows,
            device=device,
        )

    print("All UCSF-PDGM evaluations complete.")


if __name__ == "__main__":
    main()
