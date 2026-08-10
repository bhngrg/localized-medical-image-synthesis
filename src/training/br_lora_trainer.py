"""
Epoch-level orchestration for Bayesian Regional LoRA (BR-LoRA).

This module builds on the already validated atomic BR-LoRA optimization step.

Training
--------
Each training batch calls ``train_br_lora_step`` directly. The epoch runner
only manages:

    batch iteration
    global-step progression
    sample-weighted metric aggregation

Validation
----------
Validation performs the same forward-diffusion and BR-LoRA objective
calculation without gradients or optimizer updates.

Posterior-mean validation is the default so checkpoint selection can later use
a stable deterministic validation quantity rather than Monte Carlo variation.

This module deliberately contains no checkpointing, file I/O, experiment
logging, or command-line logic.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
import math

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
    br_lora_objective,
)

from .br_lora_step import (
    BRLoRAStepResult,
    train_br_lora_step,
)

from .trainer import (
    prepare_model_input,
)


class BRLoRATrainerError(
    ValueError
):
    """Raised when BR-LoRA epoch orchestration is invalid."""


@dataclass(
    frozen=True,
    slots=True,
)
class BRLoRAEpochMetrics:
    """Sample-weighted metrics from one completed epoch."""

    loss: float

    reconstruction: float
    inside: float
    outside: float

    kl: float
    normalized_kl: float

    samples: int
    batches: int


@dataclass(
    frozen=True,
    slots=True,
)
class BRLoRATrainEpochResult:
    """Result of one BR-LoRA training epoch."""

    metrics: BRLoRAEpochMetrics

    start_step: int
    next_step: int

    batches_processed: int


@dataclass(
    frozen=True,
    slots=True,
)
class BRLoRAValidationEpochResult:
    """Result of one BR-LoRA validation epoch."""

    metrics: BRLoRAEpochMetrics

    step: int
    batches_processed: int

    sample_posterior: bool


@dataclass(
    slots=True,
)
class _EpochAccumulator:
    """Accumulate sample-weighted scalar metrics."""

    weighted_loss: float = 0.0

    weighted_reconstruction: float = 0.0
    weighted_inside: float = 0.0
    weighted_outside: float = 0.0

    weighted_kl: float = 0.0
    weighted_normalized_kl: float = 0.0

    samples: int = 0
    batches: int = 0

    def update_from_step(
        self,
        result: BRLoRAStepResult,
    ) -> None:
        """Add one training-step result."""

        batch_size = int(
            result.target.shape[
                0
            ]
        )

        self._update(
            loss=float(
                result.objective.total
                .detach()
                .cpu()
                .item()
            ),

            reconstruction=float(
                result.objective.reconstruction
                .detach()
                .cpu()
                .item()
            ),

            inside=float(
                result.objective.inside
                .detach()
                .cpu()
                .item()
            ),

            outside=float(
                result.objective.outside
                .detach()
                .cpu()
                .item()
            ),

            kl=float(
                result.objective.kl
                .detach()
                .cpu()
                .item()
            ),

            normalized_kl=float(
                result.objective.normalized_kl
                .detach()
                .cpu()
                .item()
            ),

            batch_size=batch_size,
        )

    def update_from_validation(
        self,
        *,
        objective,
        batch_size: int,
    ) -> None:
        """Add one validation objective."""

        self._update(
            loss=float(
                objective.total
                .detach()
                .cpu()
                .item()
            ),

            reconstruction=float(
                objective.reconstruction
                .detach()
                .cpu()
                .item()
            ),

            inside=float(
                objective.inside
                .detach()
                .cpu()
                .item()
            ),

            outside=float(
                objective.outside
                .detach()
                .cpu()
                .item()
            ),

            kl=float(
                objective.kl
                .detach()
                .cpu()
                .item()
            ),

            normalized_kl=float(
                objective.normalized_kl
                .detach()
                .cpu()
                .item()
            ),

            batch_size=batch_size,
        )

    def _update(
        self,
        *,
        loss: float,
        reconstruction: float,
        inside: float,
        outside: float,
        kl: float,
        normalized_kl: float,
        batch_size: int,
    ) -> None:
        """Add one batch of detached scalar metrics."""

        if batch_size <= 0:
            raise BRLoRATrainerError(
                "Batch size must be positive."
            )

        values = (
            loss,
            reconstruction,
            inside,
            outside,
            kl,
            normalized_kl,
        )

        if not all(
            math.isfinite(
                value
            )
            for value in values
        ):
            raise BRLoRATrainerError(
                "Epoch metrics contain non-finite values."
            )

        weight = float(
            batch_size
        )

        self.weighted_loss += (
            loss
            * weight
        )

        self.weighted_reconstruction += (
            reconstruction
            * weight
        )

        self.weighted_inside += (
            inside
            * weight
        )

        self.weighted_outside += (
            outside
            * weight
        )

        self.weighted_kl += (
            kl
            * weight
        )

        self.weighted_normalized_kl += (
            normalized_kl
            * weight
        )

        self.samples += (
            batch_size
        )

        self.batches += 1

    def compute(
        self,
    ) -> BRLoRAEpochMetrics:
        """Return sample-weighted epoch averages."""

        if self.batches <= 0:
            raise BRLoRATrainerError(
                "Cannot compute metrics for an empty epoch."
            )

        if self.samples <= 0:
            raise BRLoRATrainerError(
                "Epoch contains no samples."
            )

        denominator = float(
            self.samples
        )

        return BRLoRAEpochMetrics(
            loss=(
                self.weighted_loss
                / denominator
            ),

            reconstruction=(
                self.weighted_reconstruction
                / denominator
            ),

            inside=(
                self.weighted_inside
                / denominator
            ),

            outside=(
                self.weighted_outside
                / denominator
            ),

            kl=(
                self.weighted_kl
                / denominator
            ),

            normalized_kl=(
                self.weighted_normalized_kl
                / denominator
            ),

            samples=self.samples,
            batches=self.batches,
        )


def train_br_lora_epoch(
    *,
    model: nn.Module,
    batches: Iterable[dict],
    optimizer: torch.optim.Optimizer,
    schedule: DiffusionSchedule,
    device: torch.device,
    start_step: int,
    kl_weight: float,
    warmup_steps: int,
    outside_weight: float = 0.05,
    max_grad_norm: float | None = 1.0,
    sample_posterior: bool = True,
) -> BRLoRATrainEpochResult:
    """
    Train BR-LoRA for exactly one epoch.

    ``start_step`` is the global optimization-step index before the first
    batch. One is added after each completed optimizer update.
    """

    _validate_nonnegative_integer(
        start_step,
        name="start_step",
    )

    accumulator = (
        _EpochAccumulator()
    )

    step = start_step

    for batch in batches:

        result = train_br_lora_step(
            model=model,
            optimizer=optimizer,
            batch=batch,
            schedule=schedule,
            device=device,
            step=step,
            kl_weight=kl_weight,
            warmup_steps=warmup_steps,
            outside_weight=outside_weight,
            max_grad_norm=max_grad_norm,
            sample_posterior=sample_posterior,
        )

        accumulator.update_from_step(
            result
        )

        step += 1

    metrics = (
        accumulator.compute()
    )

    return BRLoRATrainEpochResult(
        metrics=metrics,

        start_step=start_step,
        next_step=step,

        batches_processed=(
            accumulator.batches
        ),
    )


def validate_br_lora_epoch(
    *,
    model: nn.Module,
    batches: Iterable[dict],
    schedule: DiffusionSchedule,
    device: torch.device,
    step: int,
    kl_weight: float,
    warmup_steps: int,
    outside_weight: float = 0.05,
    sample_posterior: bool = False,
) -> BRLoRAValidationEpochResult:
    """
    Evaluate BR-LoRA for exactly one validation epoch.

    No gradients or optimizer updates are performed.

    By default, validation uses posterior means:

        sample_posterior=False

    This produces a stable validation criterion for later checkpoint
    selection. Posterior-sampled validation remains available explicitly.
    """

    _validate_nonnegative_integer(
        step,
        name="step",
    )

    if not isinstance(
        sample_posterior,
        bool,
    ):
        raise TypeError(
            "`sample_posterior` must be a bool."
        )

    variational_modules = (
        iter_variational_lora_modules(
            model
        )
    )

    if not variational_modules:
        raise BRLoRATrainerError(
            "The model contains no BR-LoRA adapters."
        )

    parameter_count = (
        variational_lora_parameter_count(
            model,
            trainable_only=False,
        )
    )

    if parameter_count <= 0:
        raise BRLoRATrainerError(
            "BR-LoRA variational parameter count must be positive."
        )

    model.eval()

    if sample_posterior:

        enable_variational_sampling(
            model
        )

    else:

        disable_variational_sampling(
            model
        )

    accumulator = (
        _EpochAccumulator()
    )

    with torch.no_grad():

        for batch in batches:

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

            prediction = model(
                model_input,
                timestep,
                condition,
            )

            _validate_prediction(
                prediction=prediction,
                target=target,
            )

            kl = (
                variational_lora_kl_divergence(
                    model,
                    reduction="sum",
                )
            )

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
                raise BRLoRATrainerError(
                    "Validation objective is non-finite."
                )

            accumulator.update_from_validation(
                objective=objective,
                batch_size=int(
                    target.shape[
                        0
                    ]
                ),
            )

    metrics = (
        accumulator.compute()
    )

    return BRLoRAValidationEpochResult(
        metrics=metrics,

        step=step,

        batches_processed=(
            accumulator.batches
        ),

        sample_posterior=(
            sample_posterior
        ),
    )


def _validate_prediction(
    *,
    prediction: Tensor,
    target: Tensor,
) -> None:
    """Validate one model prediction during validation."""

    if not isinstance(
        prediction,
        Tensor,
    ):
        raise BRLoRATrainerError(
            "The model must return a torch.Tensor."
        )

    if prediction.shape != target.shape:
        raise BRLoRATrainerError(
            "Prediction and target shapes must match.\n"
            f"Prediction: {tuple(prediction.shape)}\n"
            f"Target:     {tuple(target.shape)}"
        )

    if prediction.device != target.device:
        raise BRLoRATrainerError(
            "Prediction and target must be on the same device."
        )

    if prediction.dtype != target.dtype:
        raise BRLoRATrainerError(
            "Prediction and target must use the same dtype."
        )

    if not torch.isfinite(
        prediction
    ).all():
        raise BRLoRATrainerError(
            "Prediction contains non-finite values."
        )


def _validate_nonnegative_integer(
    value: int,
    *,
    name: str,
) -> None:
    """Validate a non-negative integer."""

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

    if value < 0:
        raise BRLoRATrainerError(
            f"`{name}` must be non-negative."
        )


__all__ = [
    "BRLoRAEpochMetrics",
    "BRLoRATrainEpochResult",
    "BRLoRATrainerError",
    "BRLoRAValidationEpochResult",
    "train_br_lora_epoch",
    "validate_br_lora_epoch",
]