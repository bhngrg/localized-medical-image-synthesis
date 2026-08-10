"""
Single optimization step for Bayesian Regional LoRA (BR-LoRA).

This module deliberately contains no epoch loops, checkpointing, logging,
validation loops, or DataLoader logic.

It reuses the already validated local baseline batch-preparation path:

    prepare_model_input()
        -> timestep sampling
        -> forward diffusion
        -> localized noisy input
        -> four-channel model input

and adds only the BR-LoRA-specific operations:

    posterior sampling / posterior-mean mode
    model forward pass
    analytic KL divergence
    normalized variational objective
    backward propagation
    optional gradient clipping
    one optimizer update
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from src.diffusion import DiffusionSchedule
from src.models.adapters import (
    disable_variational_sampling,
    enable_variational_sampling,
    iter_variational_lora_modules,
    variational_lora_kl_divergence,
    variational_lora_parameter_count,
)

from .br_lora_objectives import (
    BRLoRAVariationalObjective,
    br_lora_objective,
)

from .trainer import (
    prepare_model_input,
)


class BRLoRAStepError(
    ValueError
):
    """Raised when a BR-LoRA optimization-step input is invalid."""


@dataclass(
    frozen=True,
    slots=True,
)
class BRLoRAStepResult:
    """Outputs from one BR-LoRA optimization step."""

    objective: BRLoRAVariationalObjective

    prediction: Tensor

    timestep: Tensor
    condition: Tensor
    target: Tensor
    mask: Tensor

    variational_parameter_count: int

    gradient_norm: float | None

    sample_posterior: bool


def train_br_lora_step(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: dict,
    schedule: DiffusionSchedule,
    device: torch.device,
    step: int,
    kl_weight: float,
    warmup_steps: int,
    outside_weight: float = 0.05,
    max_grad_norm: float | None = 1.0,
    sample_posterior: bool = True,
) -> BRLoRAStepResult:
    """
    Run exactly one BR-LoRA optimizer update.

    Parameters
    ----------
    model
        AppearanceX0UNet containing already-configured BR-LoRA adapters.
    optimizer
        Optimizer operating on the trainable BR-LoRA posterior parameters.
    batch
        Batch produced by the existing BraTS H5 data pipeline.
    schedule
        Validated local DiffusionSchedule.
    device
        Device used for tensors and model execution.
    step
        Global optimization-step index used for KL warmup.
    kl_weight
        Final KL coefficient after warmup.
    warmup_steps
        Number of optimization steps used for linear KL warmup.
    outside_weight
        Weight applied to reconstruction error outside the lesion mask.
    max_grad_norm
        Optional maximum gradient norm. ``None`` disables clipping.
    sample_posterior
        True uses reparameterized Bayesian adapter realizations.
        False uses posterior means during the model forward pass.

        Variational training should normally use True.
    """

    _validate_step_inputs(
        model=model,
        optimizer=optimizer,
        batch=batch,
        schedule=schedule,
        device=device,
        step=step,
        max_grad_norm=max_grad_norm,
        sample_posterior=sample_posterior,
    )

    variational_modules = (
        iter_variational_lora_modules(
            model
        )
    )

    if not variational_modules:
        raise BRLoRAStepError(
            "The model contains no BR-LoRA adapters."
        )

    parameter_count = (
        variational_lora_parameter_count(
            model,
            trainable_only=False,
        )
    )

    if parameter_count <= 0:
        raise BRLoRAStepError(
            "BR-LoRA variational parameter count must be positive."
        )

    trainable_parameters = tuple(
        parameter
        for parameter
        in model.parameters()
        if parameter.requires_grad
    )

    if not trainable_parameters:
        raise BRLoRAStepError(
            "The BR-LoRA model contains no trainable parameters."
        )

    model.train()

    if sample_posterior:

        enable_variational_sampling(
            model
        )

    else:

        disable_variational_sampling(
            model
        )

    optimizer.zero_grad(
        set_to_none=True
    )

    #
    # Reuse the exact validated baseline batch-preparation path.
    #

    (
        model_input,
        timestep,
        condition,
        target,
    ) = prepare_model_input(
        batch=batch,
        schedule=schedule,
        device=device,
    )

    mask = batch[
        "mask"
    ].to(
        device
    )

    #
    # BR-LoRA forward pass.
    #

    prediction = model(
        model_input,
        timestep,
        condition,
    )

    if not isinstance(
        prediction,
        Tensor,
    ):
        raise BRLoRAStepError(
            "The model must return a torch.Tensor."
        )

    if prediction.shape != target.shape:
        raise BRLoRAStepError(
            "Prediction and target shapes must match.\n"
            f"Prediction: {tuple(prediction.shape)}\n"
            f"Target:     {tuple(target.shape)}"
        )

    if not torch.isfinite(
        prediction
    ).all():
        raise BRLoRAStepError(
            "Prediction contains non-finite values."
        )

    #
    # Analytic KL over every BR-LoRA posterior.
    #

    kl = variational_lora_kl_divergence(
        model,
        reduction="sum",
    )

    #
    # Complete BR-LoRA variational objective.
    #

    objective = br_lora_objective(
        prediction,
        target,
        mask,
        kl,
        variational_parameter_count=(
            parameter_count
        ),
        kl_weight=kl_weight,
        step=step,
        warmup_steps=warmup_steps,
        outside_weight=outside_weight,
    )

    if not torch.isfinite(
        objective.total
    ):
        raise BRLoRAStepError(
            "BR-LoRA objective is non-finite."
        )

    #
    # Backward pass.
    #

    objective.total.backward()

    #
    # Optional gradient clipping.
    #

    gradient_norm: float | None = None

    if max_grad_norm is not None:

        norm_tensor = (
            torch.nn.utils.clip_grad_norm_(
                trainable_parameters,
                max_norm=max_grad_norm,
            )
        )

        gradient_norm = float(
            norm_tensor.detach().cpu().item()
        )

        if not math.isfinite(
            gradient_norm
        ):
            raise BRLoRAStepError(
                "Gradient norm is non-finite."
            )

    #
    # One optimizer update.
    #

    optimizer.step()

    return BRLoRAStepResult(
        objective=objective,

        prediction=prediction,

        timestep=timestep,
        condition=condition,
        target=target,
        mask=mask,

        variational_parameter_count=(
            parameter_count
        ),

        gradient_norm=gradient_norm,

        sample_posterior=(
            sample_posterior
        ),
    )


def _validate_step_inputs(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: dict,
    schedule: DiffusionSchedule,
    device: torch.device,
    step: int,
    max_grad_norm: float | None,
    sample_posterior: bool,
) -> None:
    """Validate static inputs for one BR-LoRA optimization step."""

    if not isinstance(
        model,
        nn.Module,
    ):
        raise TypeError(
            "`model` must be a torch.nn.Module."
        )

    if not isinstance(
        optimizer,
        torch.optim.Optimizer,
    ):
        raise TypeError(
            "`optimizer` must be a torch.optim.Optimizer."
        )

    if not isinstance(
        batch,
        dict,
    ):
        raise TypeError(
            "`batch` must be a dictionary."
        )

    required_batch_keys = (
        "x0",
        "known",
        "mask",
        "donor_patch",
        "cond",
    )

    missing_keys = tuple(
        key
        for key in required_batch_keys
        if key not in batch
    )

    if missing_keys:
        raise BRLoRAStepError(
            "Batch is missing required keys: "
            + ", ".join(
                missing_keys
            )
        )

    if not isinstance(
        schedule,
        DiffusionSchedule,
    ):
        raise TypeError(
            "`schedule` must be a DiffusionSchedule."
        )

    if not isinstance(
        device,
        torch.device,
    ):
        raise TypeError(
            "`device` must be a torch.device."
        )

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
        raise BRLoRAStepError(
            "`step` must be non-negative."
        )

    if not isinstance(
        sample_posterior,
        bool,
    ):
        raise TypeError(
            "`sample_posterior` must be a bool."
        )

    if max_grad_norm is not None:

        if (
            isinstance(
                max_grad_norm,
                bool,
            )
            or not isinstance(
                max_grad_norm,
                (
                    int,
                    float,
                ),
            )
        ):
            raise TypeError(
                "`max_grad_norm` must be a real scalar or None."
            )

        converted_max_grad_norm = float(
            max_grad_norm
        )

        if (
            not math.isfinite(
                converted_max_grad_norm
            )
            or converted_max_grad_norm <= 0.0
        ):
            raise BRLoRAStepError(
                "`max_grad_norm` must be finite and positive."
            )


__all__ = [
    "BRLoRAStepError",
    "BRLoRAStepResult",
    "train_br_lora_step",
]