"""
Variational Bayesian convolutional LoRA for BR-LoRA.

This module converts deterministic convolutional LoRA adapters into
mean-field Gaussian adapters while preserving the frozen base convolution.

Each LoRA factor is represented by a DiagonalGaussianParameter:

    A ~ q(A)
    B ~ q(B)

The forward pass supports two modes:

    sample_posterior = True
        Draw a Bayesian realization of A and B.

    sample_posterior = False
        Use posterior means as deterministic point estimates.

The full fitted posterior is always retained in the module state regardless
of which forward mode is selected.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .base import AdaptationError
from .lora import (
    LoRAConv2d,
    iter_lora_modules,
)
from .selection import (
    replace_named_module,
)
from .variational import (
    DiagonalGaussianParameter,
)


class VariationalLoRAError(
    AdaptationError
):
    """Raised when a variational LoRA adapter is invalid."""


class VariationalLoRAConv2d(
    nn.Module
):
    """Mean-field variational extension of convolutional LoRA."""

    def __init__(
        self,
        conv: nn.Conv2d,
        *,
        rank: int,
        alpha: float,
        initial_a_mean: Tensor,
        initial_b_mean: Tensor,
        initial_std: float,
        prior_mean: float = 0.0,
        prior_std: float = 1.0,
        minimum_std: float = 1e-8,
        dropout: float = 0.0,
        sample_posterior: bool = True,
    ) -> None:
        super().__init__()

        if not isinstance(
            conv,
            nn.Conv2d,
        ):
            raise VariationalLoRAError(
                "`conv` must be an instance of nn.Conv2d."
            )

        if rank <= 0:
            raise VariationalLoRAError(
                f"`rank` must be positive; received {rank}."
            )

        if (
            not math.isfinite(
                alpha
            )
            or alpha <= 0.0
        ):
            raise VariationalLoRAError(
                "`alpha` must be finite and positive."
            )

        if not 0.0 <= dropout < 1.0:
            raise VariationalLoRAError(
                "`dropout` must satisfy 0 <= dropout < 1."
            )

        if not isinstance(
            sample_posterior,
            bool,
        ):
            raise VariationalLoRAError(
                "`sample_posterior` must be a bool."
            )

        if conv.groups != 1:
            raise VariationalLoRAError(
                "VariationalLoRAConv2d currently supports "
                "only Conv2d layers with groups=1."
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

        expected_a_shape = (
            self.rank,
            conv.in_channels,
            1,
            1,
        )

        expected_b_shape = (
            conv.out_channels,
            self.rank,
            *conv.kernel_size,
        )

        _validate_initial_tensor(
            initial_a_mean,
            expected_shape=expected_a_shape,
            reference=conv.weight,
            name="initial_a_mean",
        )

        _validate_initial_tensor(
            initial_b_mean,
            expected_shape=expected_b_shape,
            reference=conv.weight,
            name="initial_b_mean",
        )

        if dropout == 0.0:
            self.lora_dropout: nn.Module = nn.Identity()

        else:
            self.lora_dropout = nn.Dropout2d(
                p=dropout
            )

        self.lora_A = DiagonalGaussianParameter(
            initial_a_mean,
            initial_std=initial_std,
            prior_mean=prior_mean,
            prior_std=prior_std,
            minimum_std=minimum_std,
        )

        self.lora_B = DiagonalGaussianParameter(
            initial_b_mean,
            initial_std=initial_std,
            prior_mean=prior_mean,
            prior_std=prior_std,
            minimum_std=minimum_std,
        )

        self.adapters_enabled = True

        self.posterior_sampling_enabled = (
            sample_posterior
        )

        self._freeze_base_layer()

    @classmethod
    def from_deterministic(
        cls,
        module: LoRAConv2d,
        *,
        initial_std: float,
        prior_mean: float = 0.0,
        prior_std: float = 1.0,
        minimum_std: float = 1e-8,
        sample_posterior: bool = True,
    ) -> "VariationalLoRAConv2d":
        """Construct BR-LoRA from one deterministic LoRA adapter."""

        if not isinstance(
            module,
            LoRAConv2d,
        ):
            raise VariationalLoRAError(
                "`module` must be an instance of LoRAConv2d."
            )

        converted = cls(
            module.conv,
            rank=module.rank,
            alpha=module.alpha,
            initial_a_mean=module.lora_A.weight,
            initial_b_mean=module.lora_B.weight,
            initial_std=initial_std,
            prior_mean=prior_mean,
            prior_std=prior_std,
            minimum_std=minimum_std,
            dropout=module.dropout_probability,
            sample_posterior=sample_posterior,
        )

        converted.train(
            module.training
        )

        if not module.adapters_enabled:
            converted.disable_adapters()

        return converted

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
        """Freeze every parameter in the wrapped base convolution."""

        for parameter in self.conv.parameters():
            parameter.requires_grad_(
                False
            )

    def enable_adapters(
        self,
    ) -> None:
        """Enable the BR-LoRA contribution."""

        self.adapters_enabled = True

    def disable_adapters(
        self,
    ) -> None:
        """Disable the BR-LoRA contribution."""

        self.adapters_enabled = False

    def enable_posterior_sampling(
        self,
    ) -> None:
        """Use Bayesian posterior draws during forward propagation."""

        self.posterior_sampling_enabled = True

    def disable_posterior_sampling(
        self,
    ) -> None:
        """Use posterior means during forward propagation."""

        self.posterior_sampling_enabled = False

    def kl_divergence(
        self,
        *,
        reduction: str = "sum",
    ) -> Tensor:
        """Return the combined KL divergence for LoRA A and B."""

        if reduction == "none":
            raise VariationalLoRAError(
                "Combined adapter KL does not support reduction='none'."
            )

        return (
            self.lora_A.kl_divergence(
                reduction=reduction
            )
            + self.lora_B.kl_divergence(
                reduction=reduction
            )
        )

    def posterior_mean_adapter_weight(
        self,
    ) -> Tensor:
        """Return the dense adapter update formed from posterior means."""

        a_weight = self.lora_A.posterior_mean[
            :,
            :,
            0,
            0,
        ]

        update = torch.einsum(
            "orhw,ri->oihw",
            self.lora_B.posterior_mean,
            a_weight,
        )

        return (
            update
            * self.scale
        )

    def forward(
        self,
        inputs: Tensor,
    ) -> Tensor:
        """Apply the frozen convolution and BR-LoRA update."""

        base_output = self.conv(
            inputs
        )

        if not self.adapters_enabled:
            return base_output

        sample = (
            self.posterior_sampling_enabled
        )

        a_weight = self.lora_A.rsample(
            sample=sample
        )

        b_weight = self.lora_B.rsample(
            sample=sample
        )

        adapter_inputs = self.lora_dropout(
            inputs
        )

        rank_features = F.conv2d(
            adapter_inputs,
            a_weight,
            bias=None,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
        )

        adapter_output = _conv2d_with_weight(
            rank_features,
            weight=b_weight,
            reference=self.conv,
        )

        return (
            base_output
            + adapter_output
            * self.scale
        )


def _validate_initial_tensor(
    value: Tensor,
    *,
    expected_shape: tuple[
        int,
        ...,
    ],
    reference: Tensor,
    name: str,
) -> None:
    """Validate tensor shape, device, and dtype for posterior initialization."""

    if not isinstance(
        value,
        Tensor,
    ):
        raise VariationalLoRAError(
            f"`{name}` must be a torch.Tensor."
        )

    if tuple(
        value.shape
    ) != expected_shape:
        raise VariationalLoRAError(
            f"`{name}` has shape {tuple(value.shape)}; "
            f"expected {expected_shape}."
        )

    if value.device != reference.device:
        raise VariationalLoRAError(
            f"`{name}` must be on device {reference.device}; "
            f"received {value.device}."
        )

    if value.dtype != reference.dtype:
        raise VariationalLoRAError(
            f"`{name}` must have dtype {reference.dtype}; "
            f"received {value.dtype}."
        )


def _conv2d_with_weight(
    inputs: Tensor,
    *,
    weight: Tensor,
    reference: nn.Conv2d,
) -> Tensor:
    """Apply reference Conv2d geometry using an externally supplied weight."""

    if reference.padding_mode == "zeros":
        return F.conv2d(
            inputs,
            weight,
            bias=None,
            stride=reference.stride,
            padding=reference.padding,
            dilation=reference.dilation,
            groups=1,
        )

    pad_height, pad_width = (
        reference.padding
    )

    padded = F.pad(
        inputs,
        (
            pad_width,
            pad_width,
            pad_height,
            pad_height,
        ),
        mode=reference.padding_mode,
    )

    return F.conv2d(
        padded,
        weight,
        bias=None,
        stride=reference.stride,
        padding=0,
        dilation=reference.dilation,
        groups=1,
    )


def iter_variational_lora_modules(
    model: nn.Module,
) -> tuple[
    tuple[
        str,
        VariationalLoRAConv2d,
    ],
    ...,
]:
    """Return BR-LoRA modules in stable model order."""

    return tuple(
        (
            name,
            module,
        )
        for name, module
        in model.named_modules()
        if isinstance(
            module,
            VariationalLoRAConv2d,
        )
    )


def convert_lora_to_variational(
    model: nn.Module,
    *,
    initial_std: float,
    prior_mean: float = 0.0,
    prior_std: float = 1.0,
    minimum_std: float = 1e-8,
    target_names: tuple[
        str,
        ...,
    ] | None = None,
    sample_posterior: bool = True,
) -> tuple[
    str,
    ...,
]:
    """Replace deterministic LoRA modules with BR-LoRA modules."""

    existing_variational = (
        iter_variational_lora_modules(
            model
        )
    )

    if existing_variational:
        raise VariationalLoRAError(
            "The model already contains variational LoRA adapters: "
            + ", ".join(
                name
                for name, _
                in existing_variational
            )
        )

    deterministic_modules = (
        iter_lora_modules(
            model
        )
    )

    if not deterministic_modules:
        raise VariationalLoRAError(
            "The model contains no deterministic LoRA modules to convert."
        )

    available = {
        name: module
        for name, module
        in deterministic_modules
    }

    if target_names is None:
        selected = (
            deterministic_modules
        )

    else:
        requested = tuple(
            target_names
        )

        if not requested:
            raise VariationalLoRAError(
                "`target_names` must contain at least one module name."
            )

        if (
            len(
                set(
                    requested
                )
            )
            != len(
                requested
            )
        ):
            raise VariationalLoRAError(
                "`target_names` must not contain duplicate names."
            )

        missing = tuple(
            name
            for name in requested
            if name not in available
        )

        if missing:
            raise VariationalLoRAError(
                "Requested deterministic LoRA modules were not found: "
                + ", ".join(
                    missing
                )
            )

        requested_set = frozenset(
            requested
        )

        selected = tuple(
            (
                name,
                module,
            )
            for name, module
            in deterministic_modules
            if name in requested_set
        )

    merged_names = tuple(
        name
        for name, module
        in selected
        if module.merged
    )

    if merged_names:
        raise VariationalLoRAError(
            "Merged deterministic LoRA modules cannot be converted. "
            "Unmerge them first: "
            + ", ".join(
                merged_names
            )
        )

    selected_names = tuple(
        name
        for name, _
        in selected
    )

    for name, module in selected:

        replacement = (
            VariationalLoRAConv2d.from_deterministic(
                module,
                initial_std=initial_std,
                prior_mean=prior_mean,
                prior_std=prior_std,
                minimum_std=minimum_std,
                sample_posterior=sample_posterior,
            )
        )

        replace_named_module(
            model,
            name,
            replacement,
        )

    return selected_names


def enable_variational_lora(
    model: nn.Module,
) -> None:
    """Enable every BR-LoRA adapter."""

    for _, module in iter_variational_lora_modules(
        model
    ):
        module.enable_adapters()


def disable_variational_lora(
    model: nn.Module,
) -> None:
    """Disable every BR-LoRA adapter."""

    for _, module in iter_variational_lora_modules(
        model
    ):
        module.disable_adapters()


def enable_variational_sampling(
    model: nn.Module,
) -> None:
    """Enable genuine Bayesian realizations for the full model."""

    for _, module in iter_variational_lora_modules(
        model
    ):
        module.enable_posterior_sampling()


def disable_variational_sampling(
    model: nn.Module,
) -> None:
    """Use posterior means for every BR-LoRA adapter."""

    for _, module in iter_variational_lora_modules(
        model
    ):
        module.disable_posterior_sampling()


def variational_lora_parameter_count(
    model: nn.Module,
    *,
    trainable_only: bool = False,
) -> int:
    """
    Count BR-LoRA posterior parameters.

    Includes posterior mean and posterior rho for both A and B.
    """

    if not isinstance(
        trainable_only,
        bool,
    ):
        raise TypeError(
            "`trainable_only` must be a bool."
        )

    count = 0

    for _, module in iter_variational_lora_modules(
        model
    ):

        posterior_parameters = (
            module.lora_A.posterior_mean,
            module.lora_A.posterior_rho,
            module.lora_B.posterior_mean,
            module.lora_B.posterior_rho,
        )

        count += sum(
            parameter.numel()

            for parameter
            in posterior_parameters

            if (
                not trainable_only
                or parameter.requires_grad
            )
        )

    return count


def variational_lora_kl_divergence(
    model: nn.Module,
    *,
    reduction: str = "sum",
) -> Tensor:
    """Return total KL divergence across all BR-LoRA adapters."""

    modules = iter_variational_lora_modules(
        model
    )

    if not modules:
        raise VariationalLoRAError(
            "The model contains no variational LoRA modules."
        )

    component_kls = tuple(
        module.kl_divergence(
            reduction=reduction
        )
        for _, module
        in modules
    )

    total = component_kls[
        0
    ]

    for component in component_kls[
        1:
    ]:
        total = (
            total
            + component
        )

    return total


__all__ = [
    "VariationalLoRAConv2d",
    "VariationalLoRAError",
    "convert_lora_to_variational",
    "disable_variational_lora",
    "disable_variational_sampling",
    "enable_variational_lora",
    "enable_variational_sampling",
    "iter_variational_lora_modules",
    "variational_lora_kl_divergence",
    "variational_lora_parameter_count",
]