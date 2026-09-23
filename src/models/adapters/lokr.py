"""Deterministic convolutional LoKr adaptation.

LoKr represents each convolutional weight update as a sum of Kronecker
products in flattened weight space while keeping the pretrained convolution
parameters frozen.

Fresh adapters use Kaiming initialization for the A factors and exact zeros
for the B factors, so the initial LoKr update is exactly zero.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .base import (
    AdaptationError,
    freeze_module,
    trainable_parameter_names,
)
from .selection import (
    replace_named_module,
    select_named_modules,
)


class LoKrError(AdaptationError):
    """Raised when convolutional LoKr configuration is invalid."""


def balanced_factor_pair(
    value: int,
) -> tuple[int, int]:
    """Return the closest integer factor pair whose product is ``value``."""

    if value <= 0:
        raise LoKrError(
            f"`value` must be positive; received {value}."
        )

    root = math.isqrt(
        value
    )

    for left in range(
        root,
        0,
        -1,
    ):
        if value % left == 0:
            return (
                left,
                value // left,
            )

    raise RuntimeError(
        f"Could not factor positive integer {value}."
    )


class LoKrConv2d(nn.Module):
    """LoKr wrapper for a frozen 2D convolution."""

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
            raise LoKrError(
                "`conv` must be an instance of nn.Conv2d."
            )

        if rank <= 0:
            raise LoKrError(
                f"`rank` must be positive; received {rank}."
            )

        if (
            not math.isfinite(
                alpha
            )
            or alpha <= 0.0
        ):
            raise LoKrError(
                "`alpha` must be finite and positive."
            )

        if not 0.0 <= dropout < 1.0:
            raise LoKrError(
                "`dropout` must satisfy 0 <= dropout < 1."
            )

        if conv.groups != 1:
            raise LoKrError(
                "LoKrConv2d currently supports only "
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
            self.lokr_dropout: nn.Module = nn.Identity()

        else:
            self.lokr_dropout = nn.Dropout2d(
                p=dropout
            )

        kernel_h, kernel_w = (
            conv.kernel_size
        )

        flat_in = (
            conv.in_channels
            * kernel_h
            * kernel_w
        )

        (
            self.out_left,
            self.out_right,
        ) = balanced_factor_pair(
            conv.out_channels
        )

        (
            self.in_left,
            self.in_right,
        ) = balanced_factor_pair(
            flat_in
        )

        self.lokr_A = nn.Parameter(
            torch.empty(
                self.rank,
                self.out_left,
                self.in_left,
                device=conv.weight.device,
                dtype=conv.weight.dtype,
            )
        )

        self.lokr_B = nn.Parameter(
            torch.empty(
                self.rank,
                self.out_right,
                self.in_right,
                device=conv.weight.device,
                dtype=conv.weight.dtype,
            )
        )

        self._freeze_base_layer()
        self.reset_lokr_parameters()

    @property
    def scaling(
        self,
    ) -> float:
        """Return the Kronecker update scaling factor."""

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

    def reset_lokr_parameters(
        self,
    ) -> None:
        """Initialize the Kronecker update to exactly zero."""

        for index in range(
            self.rank
        ):
            nn.init.kaiming_uniform_(
                self.lokr_A[
                    index
                ],
                a=math.sqrt(
                    5
                ),
            )

        nn.init.zeros_(
            self.lokr_B
        )

    def adapter_weight(
        self,
    ) -> Tensor:
        """Return the dense Kronecker-structured weight update."""

        pieces = tuple(
            torch.kron(
                self.lokr_A[
                    index
                ],
                self.lokr_B[
                    index
                ],
            )
            for index in range(
                self.rank
            )
        )

        delta_flat = torch.stack(
            pieces,
            dim=0,
        ).sum(
            dim=0
        )

        delta_weight = delta_flat.view_as(
            self.conv.weight
        )

        return (
            delta_weight
            * self.scale
        )

    def effective_weight(
        self,
    ) -> Tensor:
        """Return the frozen base weight plus the LoKr update."""

        return (
            self.conv.weight
            + self.adapter_weight()
        )

    def forward(
        self,
        inputs: Tensor,
    ) -> Tensor:
        """Apply the frozen convolution plus its LoKr update."""

        base_output = self.conv(
            inputs
        )

        adapter_output = F.conv2d(
            self.lokr_dropout(
                inputs
            ),
            self.adapter_weight(),
            None,
            stride=self.conv.stride,
            padding=self.conv.padding,
            dilation=self.conv.dilation,
            groups=self.conv.groups,
        )

        return (
            base_output
            + adapter_output
        )


def iter_lokr_modules(
    model: nn.Module,
) -> tuple[
    tuple[
        str,
        LoKrConv2d,
    ],
    ...,
]:
    """Return convolutional LoKr modules in stable model order."""

    return tuple(
        (
            name,
            module,
        )
        for name, module
        in model.named_modules()
        if isinstance(
            module,
            LoKrConv2d,
        )
    )


def inject_lokr(
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
) -> tuple[str, ...]:
    """Replace selected Conv2d layers with frozen-base LoKr wrappers."""

    already_adapted = select_named_modules(
        model,
        exact_names=exact_names,
        suffixes=suffixes,
        regex_patterns=regex_patterns,
        module_types=(
            LoKrConv2d,
        ),
        exclude_patterns=exclude_patterns,
        require_match=False,
    )

    if already_adapted:
        raise LoKrError(
            "The selected modules already contain LoKr adapters: "
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
        replacement = LoKrConv2d(
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


def deterministic_lokr_parameter_count(
    model: nn.Module,
) -> int:
    """Count scalar trainable parameters across all LoKr adapters."""

    return sum(
        module.lokr_A.numel()
        + module.lokr_B.numel()

        for _, module
        in iter_lokr_modules(
            model
        )
    )


def configure_lokr(
    model: nn.Module,
    *,
    target_layers: tuple[str, ...],
    rank: int,
    alpha: float,
    dropout: float = 0.0,
) -> tuple[str, ...]:
    """Freeze a backbone and construct fresh deterministic LoKr."""

    if not target_layers:
        raise ValueError(
            "`target_layers` must contain at least one layer."
        )

    freeze_module(
        model
    )

    injected = inject_lokr(
        model,
        rank=rank,
        alpha=alpha,
        dropout=dropout,
        exact_names=target_layers,
    )

    if injected != target_layers:
        raise RuntimeError(
            "Fresh LoKr injection inventory does not match "
            "the configured target-layer sequence.\n"
            f"Configured: {target_layers}\n"
            f"Injected:   {injected}"
        )

    lokr_modules = iter_lokr_modules(
        model
    )

    lokr_names = tuple(
        name
        for name, _
        in lokr_modules
    )

    if lokr_names != target_layers:
        raise RuntimeError(
            "Deterministic LoKr inventory does not match the "
            "configured target-layer sequence."
        )

    for name, module in lokr_modules:
        nonzero_b = int(
            torch.count_nonzero(
                module.lokr_B
            ).item()
        )

        if nonzero_b != 0:
            raise RuntimeError(
                "Fresh deterministic LoKr B factors must be exactly zero; "
                f"{name!r} contains {nonzero_b} nonzero elements."
            )

    expected_trainable_names = tuple(
        parameter_name
        for layer_name in target_layers
        for parameter_name in (
            f"{layer_name}.lokr_A",
            f"{layer_name}.lokr_B",
        )
    )

    actual_trainable_names = trainable_parameter_names(
        model
    )

    if actual_trainable_names != expected_trainable_names:
        raise RuntimeError(
            "LoKr trainable-parameter inventory does not match "
            "the expected adapter-only sequence.\n"
            f"Expected: {expected_trainable_names}\n"
            f"Actual:   {actual_trainable_names}"
        )

    return injected


__all__ = [
    "LoKrConv2d",
    "LoKrError",
    "balanced_factor_pair",
    "configure_lokr",
    "deterministic_lokr_parameter_count",
    "inject_lokr",
    "iter_lokr_modules",
]
