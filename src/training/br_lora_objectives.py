"""
Variational objective components for Bayesian Regional LoRA (BR-LoRA).

The reconstruction component preserves the validated baseline masked x0 loss:

    reconstruction
        = inside-mask L1
        + outside_weight * outside-mask L1

BR-LoRA adds a parameter-normalized analytic KL term:

    normalized_kl
        = KL(q || p) / number_of_variational_parameters

with optional linear warmup:

    warmup_multiplier
        = min(step / warmup_steps, 1)

The complete objective is

    total
        = reconstruction
        + kl_weight
          * warmup_multiplier
          * normalized_kl

This module does not perform forward propagation, backward propagation,
optimizer updates, or posterior sampling. Those responsibilities remain
separate from objective construction.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from .losses import (
    DEFAULT_OUTSIDE_LOSS_WEIGHT,
    masked_x0_loss,
)


class BRLoRAObjectiveError(
    ValueError
):
    """Raised when a BR-LoRA objective input is invalid."""


@dataclass(
    frozen=True,
    slots=True,
)
class RegionalReconstructionLoss:
    """Components of the regional reconstruction loss."""

    total: Tensor
    inside: Tensor
    outside: Tensor


@dataclass(
    frozen=True,
    slots=True,
)
class BRLoRAVariationalObjective:
    """Components of the complete BR-LoRA variational objective."""

    total: Tensor

    reconstruction: Tensor
    inside: Tensor
    outside: Tensor

    kl: Tensor
    normalized_kl: Tensor

    kl_weight: float
    warmup_multiplier: float
    effective_kl_weight: float


def regional_reconstruction_loss(
    prediction: Tensor,
    target: Tensor,
    mask: Tensor,
    *,
    outside_weight: float = DEFAULT_OUTSIDE_LOSS_WEIGHT,
    denominator_epsilon: float = 1e-8,
) -> RegionalReconstructionLoss:
    """
    Compute the mask-aware reconstruction objective used by BR-LoRA.

    The calculation intentionally matches ``masked_x0_loss`` from the
    validated baseline implementation.
    """

    _validate_reconstruction_tensors(
        prediction=prediction,
        target=target,
        mask=mask,
    )

    validated_outside_weight = (
        _require_nonnegative_finite_float(
            outside_weight,
            name="outside_weight",
        )
    )

    validated_epsilon = (
        _require_positive_finite_float(
            denominator_epsilon,
            name="denominator_epsilon",
        )
    )

    absolute_error = torch.abs(
        prediction
        - target
    )

    outside_mask = (
        1.0
        - mask
    )

    inside = (
        absolute_error
        * mask
    ).sum() / (
        mask.sum()
        + validated_epsilon
    )

    outside = (
        absolute_error
        * outside_mask
    ).sum() / (
        outside_mask.sum()
        + validated_epsilon
    )

    total = (
        inside
        + validated_outside_weight
        * outside
    )

    return RegionalReconstructionLoss(
        total=total,
        inside=inside,
        outside=outside,
    )


def reconstruction_matches_baseline(
    prediction: Tensor,
    target: Tensor,
    mask: Tensor,
    *,
    outside_weight: float = DEFAULT_OUTSIDE_LOSS_WEIGHT,
) -> bool:
    """
    Return whether the BR-LoRA reconstruction total exactly matches the
    validated baseline ``masked_x0_loss`` calculation.
    """

    components = regional_reconstruction_loss(
        prediction,
        target,
        mask,
        outside_weight=outside_weight,
    )

    baseline_loss = masked_x0_loss(
        prediction,
        target,
        mask,
        outside_weight=outside_weight,
    )

    return torch.equal(
        components.total,
        baseline_loss,
    )


def linear_kl_warmup_multiplier(
    *,
    step: int,
    warmup_steps: int,
) -> float:
    """
    Return the linear KL warmup multiplier in [0, 1].

    When ``warmup_steps == 0``, warmup is disabled and the multiplier is 1.
    """

    if (
        isinstance(
            step,
            bool,
        )
        or not isinstance(
            step,
            int,
        )
    ):
        raise TypeError(
            "`step` must be an integer."
        )

    if step < 0:
        raise BRLoRAObjectiveError(
            "`step` must be non-negative."
        )

    if (
        isinstance(
            warmup_steps,
            bool,
        )
        or not isinstance(
            warmup_steps,
            int,
        )
    ):
        raise TypeError(
            "`warmup_steps` must be an integer."
        )

    if warmup_steps < 0:
        raise BRLoRAObjectiveError(
            "`warmup_steps` must be non-negative."
        )

    if warmup_steps == 0:
        return 1.0

    return min(
        float(
            step
        )
        / float(
            warmup_steps
        ),
        1.0,
    )


def normalized_variational_objective(
    reconstruction: RegionalReconstructionLoss,
    kl: Tensor,
    *,
    variational_parameter_count: int,
    kl_weight: float,
    warmup_multiplier: float = 1.0,
) -> BRLoRAVariationalObjective:
    """
    Combine regional reconstruction loss and normalized BR-LoRA KL.

    KL is normalized by the total number of variational posterior scalar
    parameters before applying the KL coefficient.
    """

    if not isinstance(
        reconstruction,
        RegionalReconstructionLoss,
    ):
        raise TypeError(
            "`reconstruction` must be a "
            "RegionalReconstructionLoss instance."
        )

    if not isinstance(
        kl,
        Tensor,
    ):
        raise TypeError(
            "`kl` must be a torch.Tensor."
        )

    if not torch.is_floating_point(
        kl
    ):
        raise BRLoRAObjectiveError(
            "`kl` must use a floating-point dtype."
        )

    if kl.ndim != 0:
        raise BRLoRAObjectiveError(
            "`kl` must be a scalar tensor."
        )

    if not torch.isfinite(
        kl
    ):
        raise BRLoRAObjectiveError(
            "`kl` must be finite."
        )

    if kl.item() < 0.0:
        raise BRLoRAObjectiveError(
            "`kl` must be non-negative."
        )

    if (
        kl.device
        != reconstruction.total.device
    ):
        raise BRLoRAObjectiveError(
            "`kl` and reconstruction loss must be "
            "on the same device."
        )

    parameter_count = (
        _require_positive_integer(
            variational_parameter_count,
            name="variational_parameter_count",
        )
    )

    validated_kl_weight = (
        _require_nonnegative_finite_float(
            kl_weight,
            name="kl_weight",
        )
    )

    validated_warmup = (
        _require_nonnegative_finite_float(
            warmup_multiplier,
            name="warmup_multiplier",
        )
    )

    if validated_warmup > 1.0:
        raise BRLoRAObjectiveError(
            "`warmup_multiplier` must not exceed 1."
        )

    normalized_kl = (
        kl
        / float(
            parameter_count
        )
    )

    effective_kl_weight = (
        validated_kl_weight
        * validated_warmup
    )

    total = (
        reconstruction.total
        + effective_kl_weight
        * normalized_kl
    )

    return BRLoRAVariationalObjective(
        total=total,

        reconstruction=(
            reconstruction.total
        ),

        inside=(
            reconstruction.inside
        ),

        outside=(
            reconstruction.outside
        ),

        kl=kl,

        normalized_kl=(
            normalized_kl
        ),

        kl_weight=(
            validated_kl_weight
        ),

        warmup_multiplier=(
            validated_warmup
        ),

        effective_kl_weight=(
            effective_kl_weight
        ),
    )


def br_lora_objective(
    prediction: Tensor,
    target: Tensor,
    mask: Tensor,
    kl: Tensor,
    *,
    variational_parameter_count: int,
    kl_weight: float,
    step: int,
    warmup_steps: int,
    outside_weight: float = DEFAULT_OUTSIDE_LOSS_WEIGHT,
) -> BRLoRAVariationalObjective:
    """
    Construct the complete BR-LoRA objective in one call.

    This is a convenience wrapper around:

    1. regional reconstruction;
    2. linear KL warmup;
    3. normalized variational objective.
    """

    reconstruction = (
        regional_reconstruction_loss(
            prediction,
            target,
            mask,
            outside_weight=outside_weight,
        )
    )

    warmup_multiplier = (
        linear_kl_warmup_multiplier(
            step=step,
            warmup_steps=warmup_steps,
        )
    )

    return normalized_variational_objective(
        reconstruction,
        kl,
        variational_parameter_count=(
            variational_parameter_count
        ),
        kl_weight=kl_weight,
        warmup_multiplier=(
            warmup_multiplier
        ),
    )


def _validate_reconstruction_tensors(
    *,
    prediction: Tensor,
    target: Tensor,
    mask: Tensor,
) -> None:
    """Validate tensors used by the regional reconstruction objective."""

    for name, tensor in (
        (
            "prediction",
            prediction,
        ),
        (
            "target",
            target,
        ),
        (
            "mask",
            mask,
        ),
    ):
        if not isinstance(
            tensor,
            Tensor,
        ):
            raise TypeError(
                f"`{name}` must be a torch.Tensor."
            )

        if not torch.is_floating_point(
            tensor
        ):
            raise BRLoRAObjectiveError(
                f"`{name}` must use a floating-point dtype."
            )

        if not torch.isfinite(
            tensor
        ).all():
            raise BRLoRAObjectiveError(
                f"`{name}` must contain only finite values."
            )

    if prediction.shape != target.shape:
        raise BRLoRAObjectiveError(
            "`prediction` and `target` must have identical shapes."
        )

    if mask.shape != target.shape:
        raise BRLoRAObjectiveError(
            "`mask` and `target` must have identical shapes."
        )

    if prediction.device != target.device:
        raise BRLoRAObjectiveError(
            "`prediction` and `target` must be on the same device."
        )

    if mask.device != target.device:
        raise BRLoRAObjectiveError(
            "`mask` and `target` must be on the same device."
        )

    if prediction.dtype != target.dtype:
        raise BRLoRAObjectiveError(
            "`prediction` and `target` must use the same dtype."
        )

    if mask.dtype != target.dtype:
        raise BRLoRAObjectiveError(
            "`mask` and `target` must use the same dtype."
        )

    if (
        torch.any(
            mask < 0.0
        )
        or torch.any(
            mask > 1.0
        )
    ):
        raise BRLoRAObjectiveError(
            "`mask` values must lie in [0, 1]."
        )


def _require_nonnegative_finite_float(
    value: float,
    *,
    name: str,
) -> float:
    """Validate a finite non-negative scalar."""

    if (
        isinstance(
            value,
            bool,
        )
        or not isinstance(
            value,
            (
                int,
                float,
            ),
        )
    ):
        raise TypeError(
            f"`{name}` must be a real scalar."
        )

    converted = float(
        value
    )

    if not math.isfinite(
        converted
    ):
        raise BRLoRAObjectiveError(
            f"`{name}` must be finite."
        )

    if converted < 0.0:
        raise BRLoRAObjectiveError(
            f"`{name}` must be non-negative."
        )

    return converted


def _require_positive_finite_float(
    value: float,
    *,
    name: str,
) -> float:
    """Validate a finite positive scalar."""

    converted = (
        _require_nonnegative_finite_float(
            value,
            name=name,
        )
    )

    if converted <= 0.0:
        raise BRLoRAObjectiveError(
            f"`{name}` must be positive."
        )

    return converted


def _require_positive_integer(
    value: int,
    *,
    name: str,
) -> int:
    """Validate a positive integer."""

    if (
        isinstance(
            value,
            bool,
        )
        or not isinstance(
            value,
            int,
        )
    ):
        raise TypeError(
            f"`{name}` must be an integer."
        )

    if value <= 0:
        raise BRLoRAObjectiveError(
            f"`{name}` must be positive."
        )

    return value


__all__ = [
    "BRLoRAObjectiveError",
    "BRLoRAVariationalObjective",
    "RegionalReconstructionLoss",
    "br_lora_objective",
    "linear_kl_warmup_multiplier",
    "normalized_variational_objective",
    "reconstruction_matches_baseline",
    "regional_reconstruction_loss",
]