"""
Deterministic convolutional DoRA adaptation.

DoRA decomposes adaptation into a trainable weight magnitude and a
low-rank directional update while keeping the pretrained convolution
parameters frozen.

Fresh adapters use Kaiming initialization for A and exact zeros for B.
The magnitude parameter is initialized from the frozen base convolution,
so the initial effective DoRA weight reproduces the pretrained weight.
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


class DoRAError(AdaptationError):
    """Raised when convolutional DoRA configuration is invalid."""


class DoRAConv2d(nn.Module):
    """DoRA wrapper for a frozen 2D convolution."""

    def __init__(
        self,
        conv: nn.Conv2d,
        *,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
        eps: float = 1.0e-6,
    ) -> None:
        super().__init__()

        if not isinstance(
            conv,
            nn.Conv2d,
        ):
            raise DoRAError(
                "`conv` must be an instance of nn.Conv2d."
            )

        if rank <= 0:
            raise DoRAError(
                f"`rank` must be positive; received {rank}."
            )

        if (
            not math.isfinite(
                alpha
            )
            or alpha <= 0.0
        ):
            raise DoRAError(
                "`alpha` must be finite and positive."
            )

        if not 0.0 <= dropout < 1.0:
            raise DoRAError(
                "`dropout` must satisfy 0 <= dropout < 1."
            )

        if (
            not math.isfinite(
                eps
            )
            or eps <= 0.0
        ):
            raise DoRAError(
                "`eps` must be finite and positive."
            )

        if conv.groups != 1:
            raise DoRAError(
                "DoRAConv2d currently supports only "
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
            self.dora_dropout: nn.Module = nn.Identity()

        else:
            self.dora_dropout = nn.Dropout2d(
                p=dropout
            )

        self.eps = float(
            eps
        )

        self.dora_A = nn.Conv2d(
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

        self.dora_B = nn.Conv2d(
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

        self._freeze_base_layer()

        with torch.no_grad():
            initial_magnitude = torch.linalg.vector_norm(
                self.conv.weight.detach().flatten(
                    start_dim=1
                ),
                dim=1,
            )

        self.magnitude = nn.Parameter(
            initial_magnitude.clone()
        )

        self.reset_dora_parameters()

    @property
    def scaling(
        self,
    ) -> float:
        """Return the low-rank update scaling factor."""

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

    def reset_dora_parameters(
        self,
    ) -> None:
        """Initialize the directional update to exactly zero."""

        nn.init.kaiming_uniform_(
            self.dora_A.weight,
            a=math.sqrt(
                5
            ),
        )

        nn.init.zeros_(
            self.dora_B.weight
        )

    def adapter_weight(
        self,
    ) -> Tensor:
        """Return the dense low-rank directional weight update."""

        a_weight = self.dora_A.weight[
            :,
            :,
            0,
            0,
        ]

        update = torch.einsum(
            "orhw,ri->oihw",
            self.dora_B.weight,
            a_weight,
        )

        return (
            update
            * self.scale
        )

    def effective_weight(
        self,
    ) -> Tensor:
        """Return the magnitude-scaled normalized DoRA weight."""

        directional_weight = (
            self.conv.weight
            + self.adapter_weight()
        )

        direction_norm = torch.linalg.vector_norm(
            directional_weight.flatten(
                start_dim=1
            ),
            dim=1,
        ).clamp_min(
            self.eps
        )

        normalized_weight = (
            directional_weight
            / direction_norm[
                :,
                None,
                None,
                None,
            ]
        )

        return (
            normalized_weight
            * self.magnitude[
                :,
                None,
                None,
                None,
            ]
        )

    def forward(
        self,
        inputs: Tensor,
    ) -> Tensor:
        """Apply DoRA with dropout restricted to the low-rank branch."""

        directional_weight = (
            self.conv.weight
            + self.adapter_weight()
        )

        direction_norm = torch.linalg.vector_norm(
            directional_weight.flatten(
                start_dim=1
            ),
            dim=1,
        ).clamp_min(
            self.eps
        )

        magnitude_scale = (
            self.magnitude
            / direction_norm
        )

        base_output = F.conv2d(
            inputs,
            self.conv.weight,
            None,
            stride=self.conv.stride,
            padding=self.conv.padding,
            dilation=self.conv.dilation,
            groups=self.conv.groups,
        )

        adapter_output = self.dora_B(
            self.dora_A(
                self.dora_dropout(
                    inputs
                )
            )
        )

        directional_output = (
            base_output
            + adapter_output
            * self.scale
        )

        output = (
            directional_output
            * magnitude_scale[
                None,
                :,
                None,
                None,
            ]
        )

        if self.conv.bias is not None:
            output = (
                output
                + self.conv.bias[
                    None,
                    :,
                    None,
                    None,
                ]
            )

        return output


def iter_dora_modules(
    model: nn.Module,
) -> tuple[
    tuple[
        str,
        DoRAConv2d,
    ],
    ...,
]:
    """Return convolutional DoRA modules in stable model order."""

    return tuple(
        (
            name,
            module,
        )
        for name, module
        in model.named_modules()
        if isinstance(
            module,
            DoRAConv2d,
        )
    )


def inject_dora(
    model: nn.Module,
    *,
    rank: int,
    alpha: float,
    dropout: float = 0.0,
    eps: float = 1.0e-6,
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
    """Replace selected Conv2d layers with frozen-base DoRA wrappers."""

    already_adapted = select_named_modules(
        model,
        exact_names=exact_names,
        suffixes=suffixes,
        regex_patterns=regex_patterns,
        module_types=(
            DoRAConv2d,
        ),
        exclude_patterns=exclude_patterns,
        require_match=False,
    )

    if already_adapted:
        raise DoRAError(
            "The selected modules already contain DoRA adapters: "
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

        replacement = DoRAConv2d(
            module,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            eps=eps,
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


def deterministic_dora_parameter_count(
    model: nn.Module,
) -> int:
    """Count scalar trainable parameters across all DoRA adapters."""

    return sum(
        module.dora_A.weight.numel()
        + module.dora_B.weight.numel()
        + module.magnitude.numel()

        for _, module
        in iter_dora_modules(
            model
        )
    )


def configure_dora(
    model: nn.Module,
    *,
    target_layers: tuple[str, ...],
    rank: int,
    alpha: float,
    dropout: float = 0.0,
    eps: float = 1.0e-6,
) -> tuple[str, ...]:
    """Freeze a backbone and construct fresh deterministic DoRA."""

    if not target_layers:
        raise ValueError(
            "`target_layers` must contain at least one layer."
        )

    freeze_module(
        model
    )

    injected = inject_dora(
        model,
        rank=rank,
        alpha=alpha,
        dropout=dropout,
        eps=eps,
        exact_names=target_layers,
    )

    if injected != target_layers:
        raise RuntimeError(
            "Fresh DoRA injection inventory does not match "
            "the configured target-layer sequence.\n"
            f"Configured: {target_layers}\n"
            f"Injected:   {injected}"
        )

    dora_modules = iter_dora_modules(
        model
    )

    dora_names = tuple(
        name
        for name, _
        in dora_modules
    )

    if dora_names != target_layers:
        raise RuntimeError(
            "Deterministic DoRA inventory does not match the "
            "configured target-layer sequence."
        )

    for name, module in dora_modules:
        nonzero_b = int(
            torch.count_nonzero(
                module.dora_B.weight
            ).item()
        )

        if nonzero_b != 0:
            raise RuntimeError(
                "Fresh deterministic DoRA B factors must be exactly zero; "
                f"{name!r} contains {nonzero_b} nonzero elements."
            )

    expected_trainable_names = tuple(
        parameter_name
        for layer_name in target_layers
        for parameter_name in (
            f"{layer_name}.magnitude",
            f"{layer_name}.dora_A.weight",
            f"{layer_name}.dora_B.weight",
        )
    )

    actual_trainable_names = trainable_parameter_names(
        model
    )

    if actual_trainable_names != expected_trainable_names:
        raise RuntimeError(
            "DoRA trainable-parameter inventory does not match "
            "the expected adapter-only sequence.\n"
            f"Expected: {expected_trainable_names}\n"
            f"Actual:   {actual_trainable_names}"
        )

    return injected


__all__ = [
    "DoRAConv2d",
    "DoRAError",
    "configure_dora",
    "deterministic_dora_parameter_count",
    "inject_dora",
    "iter_dora_modules",
]
