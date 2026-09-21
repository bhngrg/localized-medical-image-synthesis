#!/usr/bin/env python3

"""
Deterministic inner-only feathered regional composition.

The base image is preserved exactly outside the transferred lesion mask.
Inside the mask, prediction weight increases linearly with Euclidean
distance from the mask boundary until reaching one at the configured
feather width.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt


DEFAULT_INNER_FEATHER_WIDTH = 4


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

    if mask.dtype != np.bool_:
        raise TypeError(
            "Feather mask must be a boolean NumPy array."
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


def inner_feather_composite(
    *,
    prediction: torch.Tensor,
    base_image: torch.Tensor,
    transferred_mask: torch.Tensor,
    width: int,
) -> torch.Tensor:
    """
    Apply deterministic inner-only feathered regional composition.

    Inputs must have shape (1, H, W). Pixels outside the binary
    transferred mask are copied exactly from ``base_image``.
    """

    expected_ndim = 3

    for name, tensor in (
        ("prediction", prediction),
        ("base_image", base_image),
        ("transferred_mask", transferred_mask),
    ):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(
                f"{name} must be a torch.Tensor."
            )

        if tensor.ndim != expected_ndim:
            raise ValueError(
                f"{name} must have shape (1, H, W); "
                f"got {tuple(tensor.shape)}."
            )

        if tensor.shape[0] != 1:
            raise ValueError(
                f"{name} must have one channel; "
                f"got shape {tuple(tensor.shape)}."
            )

        if not torch.isfinite(tensor).all():
            raise ValueError(
                f"{name} contains non-finite values."
            )

    if (
        prediction.shape != base_image.shape
        or prediction.shape != transferred_mask.shape
    ):
        raise ValueError(
            "prediction, base_image, and transferred_mask "
            "must have identical shapes."
        )

    prediction = prediction.detach().to(
        dtype=torch.float32,
        device="cpu",
    )
    base_image = base_image.detach().to(
        dtype=torch.float32,
        device="cpu",
    )
    transferred_mask = transferred_mask.detach().to(
        dtype=torch.float32,
        device="cpu",
    )

    unique_mask = torch.unique(
        transferred_mask
    )

    if not torch.all(
        (unique_mask == 0)
        | (unique_mask == 1)
    ):
        raise ValueError(
            "transferred_mask must be binary."
        )

    mask_numpy = (
        transferred_mask
        .squeeze(0)
        .numpy()
        .astype(bool, copy=False)
    )

    weights_numpy = make_inner_feather_weights(
        mask_numpy,
        width,
    )

    weights = torch.from_numpy(
        weights_numpy
    ).unsqueeze(0)

    composite = base_image.clone()

    feather_band = (
        (weights > 0)
        & (weights < 1)
    )

    composite[feather_band] = (
        base_image[feather_band]
        + weights[feather_band]
        * (
            prediction[feather_band]
            - base_image[feather_band]
        )
    )

    full_prediction = weights == 1

    composite[full_prediction] = (
        prediction[full_prediction]
    )

    outside_mask = transferred_mask == 0

    if not torch.equal(
        composite[outside_mask],
        base_image[outside_mask],
    ):
        raise RuntimeError(
            "Inner feathering failed exact outside-mask preservation."
        )

    if not torch.isfinite(composite).all():
        raise RuntimeError(
            "Inner-feathered composite contains non-finite values."
        )

    return composite


__all__ = [
    "DEFAULT_INNER_FEATHER_WIDTH",
    "inner_feather_composite",
    "make_inner_feather_weights",
]
