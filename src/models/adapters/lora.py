"""
Deterministic convolutional LoRA used to initialize BR-LoRA.

This module wraps selected Conv2d layers in the validated AppearanceX0UNet
backbone. The frozen base convolution is preserved exactly. Fresh LoRA
adapters use Kaiming initialization for A and exact zeros for B, so the
initial adapter update is identically zero.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from .base import AdaptationError
from .selection import (
    replace_named_module,
    select_named_modules,
)


class LoRAError(AdaptationError):
    """Raised when convolutional LoRA configuration is invalid."""


class LoRAConv2d(nn.Module):
    """LoRA wrapper for a frozen 2D convolution."""

    def __init__(
        self,
        conv: nn.Conv2d,
        *,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if not isinstance(
            conv,
            nn.Conv2d,
        ):
            raise LoRAError(
                "`conv` must be an instance of nn.Conv2d."
            )

        if rank <= 0:
            raise LoRAError(
                f"`rank` must be positive; received {rank}."
            )

        if (
            not math.isfinite(
                alpha
            )
            or alpha <= 0.0
        ):
            raise LoRAError(
                "`alpha` must be finite and positive."
            )

        if not 0.0 <= dropout < 1.0:
            raise LoRAError(
                "`dropout` must satisfy 0 <= dropout < 1."
            )

        if conv.groups != 1:
            raise LoRAError(
                "LoRAConv2d currently supports only "
                "Conv2d layers with groups=1."
            )

        self.conv = conv

        self.rank = int(
            rank
        )

        self.alpha = float(
            alpha
        )

        self.scale = (
            self.alpha
            / self.rank
        )

        self.dropout_probability = float(
            dropout
        )

        if dropout == 0.0:
            self.lora_dropout: nn.Module = nn.Identity()

        else:
            self.lora_dropout = nn.Dropout2d(
                p=dropout
            )

        self.lora_A = nn.Conv2d(
            in_channels=conv.in_channels,
            out_channels=self.rank,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=False,
            device=conv.weight.device,
            dtype=conv.weight.dtype,
        )

        self.lora_B = nn.Conv2d(
            in_channels=self.rank,
            out_channels=conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            groups=1,
            bias=False,
            padding_mode=conv.padding_mode,
            device=conv.weight.device,
            dtype=conv.weight.dtype,
        )

        self.adapters_enabled = True
        self.merged = False

        self._freeze_base_layer()
        self.reset_lora_parameters()

    @property
    def scaling(
        self,
    ) -> float:
        """Return the LoRA scaling factor."""

        return self.scale

    @property
    def weight(
        self,
    ) -> nn.Parameter:
        """Expose the frozen base convolution weight."""

        return self.conv.weight

    @property
    def bias(
        self,
    ) -> nn.Parameter | None:
        """Expose the frozen base convolution bias."""

        return self.conv.bias

    def _freeze_base_layer(
        self,
    ) -> None:
        """Freeze every parameter in the wrapped convolution."""

        for parameter in self.conv.parameters():
            parameter.requires_grad_(
                False
            )

    def reset_lora_parameters(
        self,
    ) -> None:
        """
        Initialize LoRA with an exactly zero initial update.

        A uses Kaiming initialization and B is initialized to exact zeros.
        """

        nn.init.kaiming_uniform_(
            self.lora_A.weight,
            a=math.sqrt(
                5
            ),
        )

        nn.init.zeros_(
            self.lora_B.weight
        )

    def adapter_weight(
        self,
    ) -> Tensor:
        """Return the dense convolutional LoRA weight update."""

        a_weight = self.lora_A.weight[
            :,
            :,
            0,
            0,
        ]

        update = torch.einsum(
            "orhw,ri->oihw",
            self.lora_B.weight,
            a_weight,
        )

        return (
            update
            * self.scale
        )

    def enable_adapters(
        self,
    ) -> None:
        """Enable the LoRA contribution."""

        self.adapters_enabled = True

    def disable_adapters(
        self,
    ) -> None:
        """Disable the LoRA contribution."""

        self.adapters_enabled = False

    def merge(
        self,
    ) -> None:
        """Merge the adapter update into the base convolution."""

        if self.merged:
            return

        with torch.no_grad():

            self.conv.weight.add_(
                self.adapter_weight().to(
                    device=self.conv.weight.device,
                    dtype=self.conv.weight.dtype,
                )
            )

        self.merged = True

    def unmerge(
        self,
    ) -> None:
        """Remove a previously merged adapter update."""

        if not self.merged:
            return

        with torch.no_grad():

            self.conv.weight.sub_(
                self.adapter_weight().to(
                    device=self.conv.weight.device,
                    dtype=self.conv.weight.dtype,
                )
            )

        self.merged = False

    def forward(
        self,
        inputs: Tensor,
    ) -> Tensor:
        """Apply the frozen convolution and optional LoRA update."""

        base_output = self.conv(
            inputs
        )

        if (
            self.merged
            or not self.adapters_enabled
        ):
            return base_output

        adapter_output = self.lora_B(
            self.lora_A(
                self.lora_dropout(
                    inputs
                )
            )
        )

        return (
            base_output
            + adapter_output
            * self.scale
        )


def iter_lora_modules(
    model: nn.Module,
) -> tuple[
    tuple[
        str,
        LoRAConv2d,
    ],
    ...,
]:
    """Return convolutional LoRA modules in stable model order."""

    return tuple(
        (
            name,
            module,
        )
        for name, module
        in model.named_modules()
        if isinstance(
            module,
            LoRAConv2d,
        )
    )


def inject_lora(
    model: nn.Module,
    *,
    rank: int,
    alpha: float,
    dropout: float = 0.0,
    exact_names: tuple[
        str,
        ...,
    ] = (),
    suffixes: tuple[
        str,
        ...,
    ] = (),
    regex_patterns: tuple[
        str,
        ...,
    ] = (),
    exclude_patterns: tuple[
        str,
        ...,
    ] = (),
) -> tuple[
    str,
    ...,
]:
    """
    Replace selected Conv2d layers with frozen-base LoRA wrappers.

    Selection is resolved before replacements occur so module order remains
    deterministic.
    """

    already_adapted = select_named_modules(
        model,
        exact_names=exact_names,
        suffixes=suffixes,
        regex_patterns=regex_patterns,
        module_types=(
            LoRAConv2d,
        ),
        exclude_patterns=exclude_patterns,
        require_match=False,
    )

    if already_adapted:
        raise LoRAError(
            "The selected modules already contain LoRA adapters: "
            + ", ".join(
                name
                for name, _
                in already_adapted
            )
        )

    selected = select_named_modules(
        model,
        exact_names=exact_names,
        suffixes=suffixes,
        regex_patterns=regex_patterns,
        module_types=(
            nn.Conv2d,
        ),
        exclude_patterns=exclude_patterns,
        require_match=True,
    )

    selected_names = tuple(
        name
        for name, _
        in selected
    )

    for name, module in selected:

        replacement = LoRAConv2d(
            module,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
        )

        replacement.train(
            module.training
        )

        replace_named_module(
            model,
            name,
            replacement,
        )

    return selected_names


def deterministic_lora_parameter_count(
    model: nn.Module,
) -> int:
    """Count scalar parameters in LoRA A and B across all adapters."""

    return sum(
        module.lora_A.weight.numel()
        + module.lora_B.weight.numel()

        for _, module
        in iter_lora_modules(
            model
        )
    )


__all__ = [
    "LoRAConv2d",
    "LoRAError",
    "deterministic_lora_parameter_count",
    "inject_lora",
    "iter_lora_modules",
]