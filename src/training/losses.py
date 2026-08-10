"""
Loss functions for the patch-conditioned x0 diffusion baseline.

Extracted from the reference notebook training cell.
"""

from __future__ import annotations

import torch


DEFAULT_OUTSIDE_LOSS_WEIGHT = 0.05


def masked_x0_loss(
    pred_x0: torch.Tensor,
    x0: torch.Tensor,
    mask: torch.Tensor,
    outside_weight: float = DEFAULT_OUTSIDE_LOSS_WEIGHT,
) -> torch.Tensor:
    """
    Compute the reference notebook's masked x0 prediction loss.

    The notebook gives full weight to L1 error inside the tumor mask and
    weight 0.05 to L1 error outside the mask.

    Parameters
    ----------
    pred_x0
        Predicted clean image.
    x0
        Ground-truth clean image.
    mask
        Binary tumor mask.
    outside_weight
        Weight applied to the outside-mask L1 term.
        Notebook default: 0.05.
    """
    if pred_x0.shape != x0.shape:
        raise ValueError(
            "pred_x0 and x0 must have identical shapes."
        )

    if mask.shape != x0.shape:
        raise ValueError(
            "mask and x0 must have identical shapes."
        )

    if outside_weight < 0:
        raise ValueError(
            "outside_weight must be non-negative."
        )

    inside_l1 = (
        torch.abs(
            pred_x0 - x0
        )
        * mask
    ).sum() / (
        mask.sum() + 1e-8
    )

    outside_l1 = (
        torch.abs(
            pred_x0 - x0
        )
        * (
            1.0 - mask
        )
    ).sum() / (
        (
            1.0 - mask
        ).sum()
        + 1e-8
    )

    return (
        inside_l1
        + outside_weight
        * outside_l1
    )
