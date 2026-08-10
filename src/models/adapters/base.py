"""
Common utilities for model adaptation and parameter accounting.

This module provides small shared helpers used by deterministic LoRA and
Bayesian Regional LoRA (BR-LoRA).

The validated AppearanceX0UNet backbone remains unchanged. Adaptation modules
are responsible only for controlling trainable parameters and replacing
selected layers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from torch import nn


class AdaptationError(ValueError):
    """Raised when an adaptation configuration is invalid."""


@dataclass(
    frozen=True,
    slots=True,
)
class AdaptationReport:
    """Summary of a configured adaptation strategy."""

    method: str

    adapted_module_names: tuple[
        str,
        ...,
    ]

    trainable_parameter_names: tuple[
        str,
        ...,
    ]

    trainable_parameters: int
    total_parameters: int

    @property
    def trainable_fraction(
        self,
    ) -> float:
        """Return the fraction of model parameters that are trainable."""
        if self.total_parameters == 0:
            return 0.0

        return (
            self.trainable_parameters
            / self.total_parameters
        )

    @property
    def trainable_percent(
        self,
    ) -> float:
        """Return the percentage of model parameters that are trainable."""
        return (
            100.0
            * self.trainable_fraction
        )


def freeze_module(
    module: nn.Module,
) -> None:
    """Disable gradients for every parameter in a module."""
    for parameter in module.parameters():
        parameter.requires_grad_(
            False
        )


def unfreeze_module(
    module: nn.Module,
) -> None:
    """Enable gradients for every parameter in a module."""
    for parameter in module.parameters():
        parameter.requires_grad_(
            True
        )


def trainable_parameter_names(
    module: nn.Module,
) -> tuple[
    str,
    ...,
]:
    """Return trainable parameter names in stable model order."""
    return tuple(
        name
        for name, parameter
        in module.named_parameters()
        if parameter.requires_grad
    )


def count_parameters(
    module: nn.Module,
    *,
    trainable_only: bool = False,
) -> int:
    """Count scalar parameters in a model."""
    return sum(
        parameter.numel()
        for parameter
        in module.parameters()
        if (
            not trainable_only
            or parameter.requires_grad
        )
    )


def make_adaptation_report(
    module: nn.Module,
    *,
    method: str,
    adapted_module_names: Iterable[
        str
    ] = (),
) -> AdaptationReport:
    """Create a parameter-accounting report for an adapted model."""
    return AdaptationReport(
        method=method,

        adapted_module_names=tuple(
            adapted_module_names
        ),

        trainable_parameter_names=(
            trainable_parameter_names(
                module
            )
        ),

        trainable_parameters=(
            count_parameters(
                module,
                trainable_only=True,
            )
        ),

        total_parameters=(
            count_parameters(
                module
            )
        ),
    )


__all__ = [
    "AdaptationError",
    "AdaptationReport",
    "count_parameters",
    "freeze_module",
    "make_adaptation_report",
    "trainable_parameter_names",
    "unfreeze_module",
]