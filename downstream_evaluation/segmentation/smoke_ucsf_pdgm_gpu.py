#!/usr/bin/env python3

from __future__ import annotations

import numpy as np
import torch

from downstream_evaluation.segmentation.evaluate_ucsf_pdgm import (
    BATCH_SIZE,
    CHECKPOINTS,
    EXPECTED_SHAPE,
    load_manifest,
    per_slice_metrics,
    prepare_subject,
)
from downstream_evaluation.segmentation.model import VanillaUNet


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for the UCSF-PDGM GPU smoke test."
        )

    device = torch.device("cuda")

    manifest_rows = load_manifest()

    if len(manifest_rows) != 202:
        raise RuntimeError(
            f"Expected 202 external subjects, found {len(manifest_rows)}."
        )

    row = manifest_rows[0]
    subject_id = row["subject_id"]

    print("Smoke-test subject:", subject_id)

    images_np, masks_np = prepare_subject(
        row
    )

    print("Images shape:", images_np.shape)
    print("Images dtype:", images_np.dtype)
    print("Masks shape:", masks_np.shape)
    print("Masks dtype:", masks_np.dtype)
    print("Image minimum:", float(images_np.min()))
    print("Image maximum:", float(images_np.max()))
    print(
        "Tumor-positive slices:",
        int(
            (
                masks_np.sum(axis=(1, 2, 3))
                > 0
            ).sum()
        ),
    )

    expected_tensor_shape = (
        EXPECTED_SHAPE[2],
        1,
        EXPECTED_SHAPE[0],
        EXPECTED_SHAPE[1],
    )

    if images_np.shape != expected_tensor_shape:
        raise RuntimeError(
            f"Unexpected image tensor shape: {images_np.shape}"
        )

    if masks_np.shape != expected_tensor_shape:
        raise RuntimeError(
            f"Unexpected mask tensor shape: {masks_np.shape}"
        )

    if images_np.dtype != np.float32:
        raise RuntimeError(
            f"Unexpected image dtype: {images_np.dtype}"
        )

    if masks_np.dtype != np.float32:
        raise RuntimeError(
            f"Unexpected mask dtype: {masks_np.dtype}"
        )

    if not np.isfinite(images_np).all():
        raise RuntimeError(
            "Prepared UCSF image contains non-finite values."
        )

    checkpoint_path = CHECKPOINTS["real_only"]

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
    )

    threshold = float(
        checkpoint["threshold"]
    )

    print("Checkpoint:", checkpoint_path)
    print("Checkpoint epoch:", checkpoint["epoch"])
    print("Threshold:", threshold)

    model = VanillaUNet(
        in_channels=1,
        out_channels=1,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(device)
    model.eval()

    all_dice = []
    all_iou = []

    intersection = 0.0
    prediction_sum = 0.0
    target_sum = 0.0

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
        ).to(device)

        masks = torch.from_numpy(
            masks_np[start:stop]
        ).to(device)

        with torch.no_grad():
            logits = model(images)

            if logits.shape != masks.shape:
                raise RuntimeError(
                    "Model output shape does not match mask shape: "
                    f"{tuple(logits.shape)} vs {tuple(masks.shape)}"
                )

            if not torch.isfinite(logits).all():
                raise RuntimeError(
                    "Model produced non-finite logits."
                )

            dice, iou = per_slice_metrics(
                logits,
                masks,
                threshold,
            )

            predictions = (
                torch.sigmoid(logits)
                >= threshold
            ).float()

            intersection += float(
                (predictions * masks).sum().item()
            )

            prediction_sum += float(
                predictions.sum().item()
            )

            target_sum += float(
                masks.sum().item()
            )

        all_dice.extend(
            dice.cpu().tolist()
        )

        all_iou.extend(
            iou.cpu().tolist()
        )

    if len(all_dice) != EXPECTED_SHAPE[2]:
        raise RuntimeError(
            f"Expected {EXPECTED_SHAPE[2]} slice metrics, "
            f"found {len(all_dice)}."
        )

    denominator = (
        prediction_sum
        + target_sum
    )

    volumetric_dice = (
        2.0 * intersection / denominator
        if denominator > 0
        else 1.0
    )

    union = (
        prediction_sum
        + target_sum
        - intersection
    )

    volumetric_iou = (
        intersection / union
        if union > 0
        else 1.0
    )

    positive_mask = (
        masks_np.sum(axis=(1, 2, 3))
        > 0
    )

    positive_dice = np.asarray(
        all_dice,
        dtype=np.float64,
    )[positive_mask]

    positive_iou = np.asarray(
        all_iou,
        dtype=np.float64,
    )[positive_mask]

    if positive_dice.size == 0:
        raise RuntimeError(
            "Smoke-test subject has no tumor-positive slices."
        )

    print()
    print("Smoke-test metrics")
    print("Mean all-slice Dice:", float(np.mean(all_dice)))
    print("Mean all-slice IoU:", float(np.mean(all_iou)))
    print(
        "Mean positive-slice Dice:",
        float(positive_dice.mean()),
    )
    print(
        "Mean positive-slice IoU:",
        float(positive_iou.mean()),
    )
    print("Volumetric Dice:", volumetric_dice)
    print("Volumetric IoU:", volumetric_iou)

    print()
    print("UCSF-PDGM GPU smoke test PASSED.")


if __name__ == "__main__":
    main()
