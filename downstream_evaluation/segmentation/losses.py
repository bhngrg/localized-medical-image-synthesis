#!/usr/bin/env python3

from __future__ import annotations

import torch
import torch.nn as nn


def dice_loss_from_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    smooth: float = 1.0,
) -> torch.Tensor:
    """
    Soft Dice loss computed from raw model logits.
    """
    probabilities = torch.sigmoid(logits)

    intersection = 2.0 * (
        probabilities * targets
    ).sum()

    denominator = (
        probabilities.sum()
        + targets.sum()
    )

    dice = (
        intersection + smooth
    ) / (
        denominator + smooth
    )

    return 1.0 - dice


class BCEDiceLoss(nn.Module):
    """
    Combined BCE-with-logits and soft Dice loss.
    """

    def __init__(
        self,
        smooth: float = 1.0,
    ) -> None:
        super().__init__()

        self.smooth = float(smooth)
        self.bce = nn.BCEWithLogitsLoss()

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        bce = self.bce(
            logits,
            targets,
        )

        dice = dice_loss_from_logits(
            logits,
            targets,
            smooth=self.smooth,
        )

        return bce + dice
