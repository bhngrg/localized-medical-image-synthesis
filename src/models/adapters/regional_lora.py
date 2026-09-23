"""Deterministic Regional LoRA configuration.

This module defines the method-level construction used by deterministic
Regional LoRA and by BR-LoRA before deterministic adapters are converted to
their variational counterparts.

The low-level convolutional LoRA implementation remains in ``lora.py``.
"""

from __future__ import annotations

import torch
from torch import nn

from .base import (
    freeze_module,
    trainable_parameter_names,
)
from .lora import (
    inject_lora,
    iter_lora_modules,
)


def configure_regional_lora(
    model: nn.Module,
    *,
    target_layers: tuple[str, ...],
    rank: int,
    alpha: float,
    dropout: float = 0.0,
) -> tuple[str, ...]:
    """Freeze a backbone and construct fresh deterministic Regional LoRA."""

    if not target_layers:
        raise ValueError(
            "`target_layers` must contain at least one layer."
        )

    freeze_module(
        model
    )

    injected = inject_lora(
        model,
        rank=rank,
        alpha=alpha,
        dropout=dropout,
        exact_names=target_layers,
    )

    if injected != target_layers:
        raise RuntimeError(
            "Fresh Regional LoRA injection inventory does not match "
            "the configured target-layer sequence.\n"
            f"Configured: {target_layers}\n"
            f"Injected:   {injected}"
        )

    deterministic_modules = iter_lora_modules(
        model
    )

    deterministic_names = tuple(
        name
        for name, _
        in deterministic_modules
    )

    if deterministic_names != target_layers:
        raise RuntimeError(
            "Deterministic Regional LoRA inventory does not match the "
            "configured target-layer sequence."
        )

    for name, module in deterministic_modules:
        nonzero_b = int(
            torch.count_nonzero(
                module.lora_B.weight
            ).item()
        )

        if nonzero_b != 0:
            raise RuntimeError(
                "Fresh deterministic LoRA B factors must be exactly zero; "
                f"{name!r} contains {nonzero_b} nonzero elements."
            )

    expected_trainable_names = tuple(
        parameter_name
        for layer_name in target_layers
        for parameter_name in (
            f"{layer_name}.lora_A.weight",
            f"{layer_name}.lora_B.weight",
        )
    )

    actual_trainable_names = trainable_parameter_names(
        model
    )

    if actual_trainable_names != expected_trainable_names:
        raise RuntimeError(
            "Regional LoRA trainable-parameter inventory does not match "
            "the expected adapter-only sequence.\n"
            f"Expected: {expected_trainable_names}\n"
            f"Actual:   {actual_trainable_names}"
        )

    return injected


__all__ = [
    "configure_regional_lora",
]
