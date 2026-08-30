#!/usr/bin/env python3

from __future__ import annotations

import torch


def _threshold_predictions(
    logits: torch.Tensor,
    threshold: float,
) -> torch.Tensor:
    probabilities = torch.sigmoid(logits)

    return (
        probabilities >= threshold
    ).float()


def dice_coefficient(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
) -> torch.Tensor:
    """
    Mean binary Dice coefficient across samples in a batch.
    """
    predictions = _threshold_predictions(
        logits,
        threshold,
    )

    dims = tuple(
        range(
            1,
            predictions.ndim,
        )
    )

    intersection = (
        predictions * targets
    ).sum(dim=dims)

    prediction_sum = predictions.sum(
        dim=dims
    )

    target_sum = targets.sum(
        dim=dims
    )

    denominator = (
        prediction_sum
        + target_sum
    )

    dice = torch.where(
        denominator > 0,
        2.0 * intersection / denominator,
        torch.ones_like(denominator),
    )

    return dice.mean()


def iou_score(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
) -> torch.Tensor:
    """
    Mean binary intersection-over-union across samples in a batch.
    """
    predictions = _threshold_predictions(
        logits,
        threshold,
    )

    dims = tuple(
        range(
            1,
            predictions.ndim,
        )
    )

    intersection = (
        predictions * targets
    ).sum(dim=dims)

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

    return iou.mean()
