"""
Posterior-product utilities for BR-LoRA inference.

This module operates on posterior prediction realizations produced by BR-LoRA
inference. It provides reusable construction of posterior mean, variance, and
standard-deviation maps together with deterministic hard regional composition.

No model loading, posterior sampling, dataset access, case selection, or output
serialization is performed here.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


class PosteriorProductsError(
    ValueError
):
    """Raised when BR-LoRA posterior products cannot be constructed."""


@dataclass(
    frozen=True,
    slots=True,
)
class PosteriorProducts:
    """Posterior summaries derived from one prediction realization stack."""

    sample_count: int

    prediction_samples: torch.Tensor

    prediction_mean: torch.Tensor
    prediction_variance: torch.Tensor
    prediction_std: torch.Tensor


def _validate_prediction_samples(
    prediction_samples: torch.Tensor,
) -> None:
    """Validate one BR-LoRA posterior prediction stack."""

    if not isinstance(
        prediction_samples,
        torch.Tensor,
    ):
        raise TypeError(
            "`prediction_samples` must be a torch.Tensor."
        )

    if prediction_samples.ndim != 5:
        raise PosteriorProductsError(
            "`prediction_samples` must have shape "
            "(samples, batch, channel, height, width)."
        )

    if prediction_samples.shape[
        0
    ] <= 1:
        raise PosteriorProductsError(
            "`prediction_samples` must contain at least two "
            "posterior realizations."
        )

    if prediction_samples.shape[
        1
    ] <= 0:
        raise PosteriorProductsError(
            "`prediction_samples` must contain at least one case."
        )

    if prediction_samples.shape[
        2
    ] <= 0:
        raise PosteriorProductsError(
            "`prediction_samples` must contain at least one channel."
        )

    if not torch.is_floating_point(
        prediction_samples
    ):
        raise TypeError(
            "`prediction_samples` must use a floating-point dtype."
        )

    if not torch.isfinite(
        prediction_samples
    ).all():
        raise PosteriorProductsError(
            "`prediction_samples` contains non-finite values."
        )


def _validate_composition_inputs(
    *,
    predictions: torch.Tensor,
    base_image: torch.Tensor,
    transferred_mask: torch.Tensor,
) -> None:
    """Validate tensors used for hard regional composition."""

    if not isinstance(
        predictions,
        torch.Tensor,
    ):
        raise TypeError(
            "`predictions` must be a torch.Tensor."
        )

    if not isinstance(
        base_image,
        torch.Tensor,
    ):
        raise TypeError(
            "`base_image` must be a torch.Tensor."
        )

    if not isinstance(
        transferred_mask,
        torch.Tensor,
    ):
        raise TypeError(
            "`transferred_mask` must be a torch.Tensor."
        )

    if predictions.ndim not in (
        4,
        5,
    ):
        raise PosteriorProductsError(
            "`predictions` must have shape "
            "(batch, channel, height, width) or "
            "(samples, batch, channel, height, width)."
        )

    if base_image.ndim != 4:
        raise PosteriorProductsError(
            "`base_image` must have shape "
            "(batch, channel, height, width)."
        )

    if transferred_mask.ndim != 4:
        raise PosteriorProductsError(
            "`transferred_mask` must have shape "
            "(batch, channel, height, width)."
        )

    if base_image.shape != transferred_mask.shape:
        raise PosteriorProductsError(
            "Base image and transferred mask must have identical shapes."
        )

    expected_shape = (
        predictions.shape[
            -4:
        ]
    )

    if expected_shape != base_image.shape:
        raise PosteriorProductsError(
            "Prediction and composition-input shapes do not match.\n\n"
            f"Prediction trailing shape: {tuple(expected_shape)}\n"
            f"Base/mask shape:          {tuple(base_image.shape)}"
        )

    for name, tensor in (
        (
            "predictions",
            predictions,
        ),
        (
            "base_image",
            base_image,
        ),
        (
            "transferred_mask",
            transferred_mask,
        ),
    ):
        if not torch.is_floating_point(
            tensor
        ):
            raise TypeError(
                f"`{name}` must use a floating-point dtype."
            )

        if not torch.isfinite(
            tensor
        ).all():
            raise PosteriorProductsError(
                f"`{name}` contains non-finite values."
            )

    if (
        predictions.device
        != base_image.device
        or predictions.device
        != transferred_mask.device
    ):
        raise PosteriorProductsError(
            "Predictions, base image, and transferred mask must be on "
            "the same device."
        )

    if (
        predictions.dtype
        != base_image.dtype
        or predictions.dtype
        != transferred_mask.dtype
    ):
        raise PosteriorProductsError(
            "Predictions, base image, and transferred mask must use "
            "the same dtype."
        )

    if (
        transferred_mask < 0
    ).any() or (
        transferred_mask > 1
    ).any():
        raise PosteriorProductsError(
            "`transferred_mask` must contain values in [0, 1]."
        )


def compute_posterior_products(
    prediction_samples: torch.Tensor,
) -> PosteriorProducts:
    """
    Compute posterior mean, unbiased variance, and standard deviation.

    Parameters
    ----------
    prediction_samples
        Posterior realization stack with shape
        ``(samples, batch, channel, height, width)``.

    Returns
    -------
    PosteriorProducts
        Raw prediction stack together with posterior summaries.
    """

    _validate_prediction_samples(
        prediction_samples
    )

    prediction_mean = prediction_samples.mean(
        dim=0
    )

    prediction_variance = prediction_samples.var(
        dim=0,
        unbiased=True,
    )

    prediction_std = torch.sqrt(
        prediction_variance
    )

    for name, tensor in (
        (
            "prediction_mean",
            prediction_mean,
        ),
        (
            "prediction_variance",
            prediction_variance,
        ),
        (
            "prediction_std",
            prediction_std,
        ),
    ):
        if not torch.isfinite(
            tensor
        ).all():
            raise PosteriorProductsError(
                f"{name} contains non-finite values."
            )

    if (
        prediction_variance < 0
    ).any():
        raise PosteriorProductsError(
            "Posterior variance contains negative values."
        )

    expected_summary_shape = prediction_samples.shape[
        1:
    ]

    for name, tensor in (
        (
            "prediction_mean",
            prediction_mean,
        ),
        (
            "prediction_variance",
            prediction_variance,
        ),
        (
            "prediction_std",
            prediction_std,
        ),
    ):
        if tensor.shape != expected_summary_shape:
            raise PosteriorProductsError(
                f"{name} has an unexpected shape.\n\n"
                f"Observed: {tuple(tensor.shape)}\n"
                f"Expected: {tuple(expected_summary_shape)}"
            )

    return PosteriorProducts(
        sample_count=int(
            prediction_samples.shape[
                0
            ]
        ),
        prediction_samples=prediction_samples,
        prediction_mean=prediction_mean,
        prediction_variance=prediction_variance,
        prediction_std=prediction_std,
    )


def reconstruct_composites(
    *,
    predictions: torch.Tensor,
    base_image: torch.Tensor,
    transferred_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Apply deterministic hard regional composition.

    ``predictions`` may contain either one prediction tensor with shape
    ``(batch, channel, height, width)`` or a posterior realization stack with
    shape ``(samples, batch, channel, height, width)``.

    Pixels outside the transferred mask are copied exactly from the base image.
    """

    _validate_composition_inputs(
        predictions=predictions,
        base_image=base_image,
        transferred_mask=transferred_mask,
    )

    if predictions.ndim == 5:
        expanded_base = base_image.unsqueeze(
            0
        )

        expanded_mask = transferred_mask.unsqueeze(
            0
        )

    else:
        expanded_base = base_image
        expanded_mask = transferred_mask

    composites = (
        predictions
        * expanded_mask
        + expanded_base
        * (
            1.0
            - expanded_mask
        )
    )

    if composites.shape != predictions.shape:
        raise PosteriorProductsError(
            "Hard-composited predictions have an unexpected shape."
        )

    if not torch.isfinite(
        composites
    ).all():
        raise PosteriorProductsError(
            "Hard-composited predictions contain non-finite values."
        )

    outside_mask = (
        expanded_mask
        == 0
    )

    expected_outside = expanded_base.expand_as(
        composites
    )

    if not torch.equal(
        composites[
            outside_mask.expand_as(
                composites
            )
        ],
        expected_outside[
            outside_mask.expand_as(
                composites
            )
        ],
    ):
        raise PosteriorProductsError(
            "Hard composition failed exact outside-mask preservation."
        )

    return composites


def reconstruct_composite_mean(
    *,
    prediction_mean: torch.Tensor,
    base_image: torch.Tensor,
    transferred_mask: torch.Tensor,
) -> torch.Tensor:
    """Apply hard regional composition to one posterior-mean prediction."""

    return reconstruct_composites(
        predictions=prediction_mean,
        base_image=base_image,
        transferred_mask=transferred_mask,
    )


__all__ = [
    "PosteriorProducts",
    "PosteriorProductsError",
    "compute_posterior_products",
    "reconstruct_composite_mean",
    "reconstruct_composites",
]
