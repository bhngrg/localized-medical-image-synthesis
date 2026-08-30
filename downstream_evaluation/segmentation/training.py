"""
Shared training utilities for downstream tumor segmentation.
"""

from __future__ import annotations

import torch

from downstream_evaluation.segmentation.metrics import (
    dice_coefficient,
    iou_score,
)


def segmentation_collate(
    batch: list[dict[str, object]],
) -> dict[str, torch.Tensor]:
    """
    Collate only image and mask tensors.

    Real and BR-LoRA synthetic datasets expose different provenance fields,
    so downstream training intentionally collates only the tensors required
    by the segmentation objective.
    """
    return {
        "image": torch.stack(
            [sample["image"] for sample in batch]
        ),
        "mask": torch.stack(
            [sample["mask"] for sample in batch]
        ),
    }


def run_train_epoch(
    model: torch.nn.Module,
    loader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    threshold: float = 0.5,
) -> tuple[float, float]:
    """Run one downstream segmentation training epoch."""
    model.train()

    total_loss = 0.0
    total_dice = 0.0
    total_samples = 0

    for batch in loader:
        images = batch["image"].to(
            device,
            non_blocking=True,
        )
        masks = batch["mask"].to(
            device,
            non_blocking=True,
        )

        optimizer.zero_grad(
            set_to_none=True,
        )

        logits = model(images)

        loss = criterion(
            logits,
            masks,
        )

        loss.backward()
        optimizer.step()

        batch_size = images.shape[0]

        with torch.no_grad():
            dice = dice_coefficient(
                logits,
                masks,
                threshold=threshold,
            )

        total_loss += float(loss.item()) * batch_size
        total_dice += float(dice.item()) * batch_size
        total_samples += batch_size

    return (
        total_loss / total_samples,
        total_dice / total_samples,
    )


@torch.no_grad()
def run_validation_epoch(
    model: torch.nn.Module,
    loader,
    criterion: torch.nn.Module,
    device: torch.device,
    *,
    threshold: float = 0.5,
) -> tuple[float, float, float, float, float]:
    """
    Run one validation epoch.

    Returns all-slice loss/Dice/IoU followed by tumor-positive-slice
    Dice/IoU, preserving the metrics used in the preliminary experiments.
    """
    model.eval()

    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    total_samples = 0

    positive_dice_sum = 0.0
    positive_iou_sum = 0.0
    positive_samples = 0

    for batch in loader:
        images = batch["image"].to(
            device,
            non_blocking=True,
        )
        masks = batch["mask"].to(
            device,
            non_blocking=True,
        )

        logits = model(images)

        loss = criterion(
            logits,
            masks,
        )

        dice = dice_coefficient(
            logits,
            masks,
            threshold=threshold,
        )

        iou = iou_score(
            logits,
            masks,
            threshold=threshold,
        )

        batch_size = images.shape[0]

        total_loss += float(loss.item()) * batch_size
        total_dice += float(dice.item()) * batch_size
        total_iou += float(iou.item()) * batch_size
        total_samples += batch_size

        positive_mask = (
            masks.sum(dim=(1, 2, 3)) > 0
        )

        n_positive = int(
            positive_mask.sum().item()
        )

        if n_positive > 0:
            positive_logits = logits[positive_mask]
            positive_targets = masks[positive_mask]

            positive_dice = dice_coefficient(
                positive_logits,
                positive_targets,
                threshold=threshold,
            )

            positive_iou = iou_score(
                positive_logits,
                positive_targets,
                threshold=threshold,
            )

            positive_dice_sum += (
                float(positive_dice.item())
                * n_positive
            )

            positive_iou_sum += (
                float(positive_iou.item())
                * n_positive
            )

            positive_samples += n_positive

    if positive_samples == 0:
        raise RuntimeError(
            "Validation set contains no tumor-positive slices."
        )

    return (
        total_loss / total_samples,
        total_dice / total_samples,
        total_iou / total_samples,
        positive_dice_sum / positive_samples,
        positive_iou_sum / positive_samples,
    )
