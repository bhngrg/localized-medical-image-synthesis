"""
Posterior-mean and posterior-sampling inference for Bayesian Regional LoRA.

This module provides the first BR-LoRA inference layer without defining any
reliability score or interpretation.

Two inference modes are supported:

posterior mean
    Every variational LoRA factor uses its fitted posterior mean. No adapter
    sampling occurs.

posterior sampling
    Each model forward draws one independent joint realization of all fitted
    variational LoRA factors. A shared prepared diffusion input can be reused
    across realizations so variation is attributable to the BR-LoRA posterior
    rather than newly sampled diffusion noise.

The fitted BR-LoRA checkpoint is self-contained: it stores the full frozen
backbone together with the variational adapter state. Inference therefore
reconstructs the model architecture from checkpoint metadata, creates the
adapter structure, and restores the full checkpoint state strictly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from src.diffusion import DiffusionSchedule
from src.models import AppearanceX0UNet
from src.models.adapters import (
    convert_lora_to_variational,
    disable_variational_sampling,
    enable_variational_sampling,
    freeze_module,
    inject_lora,
    iter_variational_lora_modules,
    variational_lora_parameter_count,
)
from src.training.br_lora_checkpoint import (
    load_br_lora_checkpoint,
    restore_br_lora_checkpoint,
)


class BRLoRAInferenceError(
    ValueError
):
    """Raised when BR-LoRA inference inputs or checkpoint state are invalid."""


@dataclass(
    frozen=True,
    slots=True,
)
class LoadedBRLoRA:
    """A strictly restored fitted BR-LoRA model and checkpoint metadata."""

    model: AppearanceX0UNet
    checkpoint: dict[str, Any]

    variational_module_names: tuple[
        str,
        ...,
    ]

    variational_parameter_count: int


@dataclass(
    frozen=True,
    slots=True,
)
class PreparedBRLoRAInference:
    """One fixed diffusion input reused by posterior-mean/sample inference."""

    model_input: Tensor

    target: Tensor
    known: Tensor
    mask: Tensor
    donor_patch: Tensor
    condition: Tensor

    timestep: Tensor
    diffusion_noise: Tensor
    x_t: Tensor


@dataclass(
    frozen=True,
    slots=True,
)
class BRLoRAPosteriorMeanResult:
    """Posterior-mean BR-LoRA prediction for one prepared inference input."""

    prediction: Tensor
    composite: Tensor

    prepared: PreparedBRLoRAInference


@dataclass(
    frozen=True,
    slots=True,
)
class BRLoRAPosteriorSamplesResult:
    """Raw BR-LoRA posterior realizations and direct Monte Carlo summaries."""

    prediction_samples: Tensor
    composite_samples: Tensor

    prediction_mean: Tensor
    prediction_variance: Tensor
    prediction_std: Tensor

    composite_mean: Tensor
    composite_variance: Tensor
    composite_std: Tensor

    prepared: PreparedBRLoRAInference

    posterior_samples: int


def load_fitted_br_lora(
    checkpoint_path: str | Path,
    *,
    device: torch.device,
) -> LoadedBRLoRA:
    """
    Reconstruct and strictly restore one fitted BR-LoRA checkpoint.

    No optimizer or RNG state is restored for ordinary inference.
    """

    if not isinstance(
        device,
        torch.device,
    ):
        raise TypeError(
            "`device` must be a torch.device."
        )

    path = Path(
        checkpoint_path
    ).expanduser().resolve()

    if not path.is_file():
        raise FileNotFoundError(
            f"BR-LoRA checkpoint not found:\n{path}"
        )

    checkpoint = load_br_lora_checkpoint(
        path,
        map_location=device,
    )

    model_config = checkpoint.get(
        "model_config"
    )

    br_lora_config = checkpoint.get(
        "br_lora_config"
    )

    if not isinstance(
        model_config,
        Mapping,
    ):
        raise BRLoRAInferenceError(
            "BR-LoRA checkpoint must contain mapping-valued model_config."
        )

    if not isinstance(
        br_lora_config,
        Mapping,
    ):
        raise BRLoRAInferenceError(
            "BR-LoRA checkpoint must contain mapping-valued br_lora_config."
        )

    model = _build_backbone_from_config(
        model_config,
        device=device,
    )

    target_layers = _target_layers_from_config(
        br_lora_config
    )

    freeze_module(
        model
    )

    injected_names = inject_lora(
        model,
        rank=_require_positive_integer(
            br_lora_config.get(
                "rank",
                4,
            ),
            name="br_lora.rank",
        ),
        alpha=_require_positive_finite_float(
            br_lora_config.get(
                "alpha",
                8.0,
            ),
            name="br_lora.alpha",
        ),
        dropout=_require_probability_below_one(
            br_lora_config.get(
                "dropout",
                0.0,
            ),
            name="br_lora.dropout",
        ),
        exact_names=target_layers,
    )

    if injected_names != target_layers:
        raise BRLoRAInferenceError(
            "Deterministic LoRA injection inventory does not match "
            "checkpoint target layers."
        )

    converted_names = convert_lora_to_variational(
        model,
        initial_std=_require_positive_finite_float(
            br_lora_config.get(
                "initial_std",
                0.01,
            ),
            name="br_lora.initial_std",
        ),
        prior_mean=_require_finite_float(
            br_lora_config.get(
                "prior_mean",
                0.0,
            ),
            name="br_lora.prior_mean",
        ),
        prior_std=_require_positive_finite_float(
            br_lora_config.get(
                "prior_std",
                1.0,
            ),
            name="br_lora.prior_std",
        ),
        minimum_std=_require_positive_finite_float(
            br_lora_config.get(
                "minimum_std",
                1.0e-8,
            ),
            name="br_lora.minimum_std",
        ),
        target_names=target_layers,
        sample_posterior=False,
    )

    if converted_names != target_layers:
        raise BRLoRAInferenceError(
            "Variational LoRA conversion inventory does not match "
            "checkpoint target layers."
        )

    restored = restore_br_lora_checkpoint(
        payload=checkpoint,
        model=model,
        optimizer=None,
        strict=True,
        restore_rng=False,
    )

    load_result = restored[
        "load_result"
    ]

    if (
        load_result.missing_keys
        or load_result.unexpected_keys
    ):
        raise BRLoRAInferenceError(
            "Strict BR-LoRA checkpoint restoration reported "
            "state-dict mismatches."
        )

    model.eval()

    disable_variational_sampling(
        model
    )

    module_names = tuple(
        name
        for name, _
        in iter_variational_lora_modules(
            model
        )
    )

    checkpoint_module_names = tuple(
        checkpoint[
            "variational_module_names"
        ]
    )

    if module_names != checkpoint_module_names:
        raise BRLoRAInferenceError(
            "Restored BR-LoRA module inventory does not match checkpoint."
        )

    parameter_count = (
        variational_lora_parameter_count(
            model,
            trainable_only=False,
        )
    )

    checkpoint_parameter_count = int(
        checkpoint[
            "variational_parameter_count"
        ]
    )

    if parameter_count != checkpoint_parameter_count:
        raise BRLoRAInferenceError(
            "Restored BR-LoRA posterior parameter count does not match "
            "checkpoint metadata."
        )

    return LoadedBRLoRA(
        model=model,
        checkpoint=checkpoint,
        variational_module_names=(
            module_names
        ),
        variational_parameter_count=(
            parameter_count
        ),
    )


def prepare_br_lora_batch(
    batch: dict,
    *,
    schedule: DiffusionSchedule,
    device: torch.device,
    timestep_fraction: float = 0.75,
    max_samples: int | None = None,
    diffusion_noise: Tensor | None = None,
) -> PreparedBRLoRAInference:
    """
    Prepare one fixed reconstruction-style diffusion input.

    This mirrors the validated baseline ``reconstruct_batch`` construction but
    stops before model forward propagation. Reusing the returned object across
    multiple posterior forwards keeps timestep and diffusion noise fixed.

    When ``diffusion_noise`` is None, exactly one fresh Gaussian noise tensor is
    drawn here.
    """

    if not isinstance(
        batch,
        dict,
    ):
        raise TypeError(
            "`batch` must be a dictionary."
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

    timestep_fraction_value = _require_fraction(
        timestep_fraction,
        name="timestep_fraction",
    )

    if max_samples is not None:
        max_samples = _require_positive_integer(
            max_samples,
            name="max_samples",
        )

    required_keys = (
        "x0",
        "known",
        "mask",
        "donor_patch",
        "cond",
    )

    missing_keys = tuple(
        key
        for key in required_keys
        if key not in batch
    )

    if missing_keys:
        raise BRLoRAInferenceError(
            "Batch is missing required keys: "
            + ", ".join(
                missing_keys
            )
        )

    def sliced(
        key: str,
    ) -> Tensor:
        value = batch[
            key
        ]

        if not isinstance(
            value,
            Tensor,
        ):
            raise TypeError(
                f"batch[{key!r}] must be a torch.Tensor."
            )

        if max_samples is not None:
            value = value[
                :max_samples
            ]

        return value.to(
            device
        )

    target = sliced(
        "x0"
    )

    known = sliced(
        "known"
    )

    mask = sliced(
        "mask"
    )

    donor_patch = sliced(
        "donor_patch"
    )

    condition = sliced(
        "cond"
    )

    if target.ndim != 4:
        raise BRLoRAInferenceError(
            "`x0` must have shape [B, C, H, W]."
        )

    if target.shape[0] <= 0:
        raise BRLoRAInferenceError(
            "Prepared batch must contain at least one sample."
        )

    for name, value in (
        (
            "known",
            known,
        ),
        (
            "mask",
            mask,
        ),
        (
            "donor_patch",
            donor_patch,
        ),
    ):
        if value.shape != target.shape:
            raise BRLoRAInferenceError(
                f"`{name}` must have the same shape as `x0`."
            )

    if condition.ndim != 2:
        raise BRLoRAInferenceError(
            "`cond` must have shape [B, cond_dim]."
        )

    if condition.shape[0] != target.shape[0]:
        raise BRLoRAInferenceError(
            "`cond` batch size must equal `x0` batch size."
        )

    t_value = int(
        timestep_fraction_value
        * schedule.timesteps
    )

    timestep = torch.full(
        (
            target.shape[
                0
            ],
        ),
        t_value,
        device=device,
        dtype=torch.long,
    )

    if diffusion_noise is None:
        noise = torch.randn_like(
            target
        )

    else:
        if not isinstance(
            diffusion_noise,
            Tensor,
        ):
            raise TypeError(
                "`diffusion_noise` must be a torch.Tensor or None."
            )

        if diffusion_noise.shape != target.shape:
            raise BRLoRAInferenceError(
                "`diffusion_noise` must have the same shape as `x0`."
            )

        noise = diffusion_noise.to(
            device=device,
            dtype=target.dtype,
        )

    if not torch.isfinite(
        noise
    ).all():
        raise BRLoRAInferenceError(
            "`diffusion_noise` must contain only finite values."
        )

    x_t_full = schedule.q_sample(
        x0=target,
        t=timestep,
        noise=noise,
    )

    x_t = (
        target
        * (
            1.0
            - mask
        )
        + x_t_full
        * mask
    )

    model_input = torch.cat(
        [
            x_t,
            known,
            mask,
            donor_patch,
        ],
        dim=1,
    )

    return PreparedBRLoRAInference(
        model_input=model_input,
        target=target,
        known=known,
        mask=mask,
        donor_patch=donor_patch,
        condition=condition,
        timestep=timestep,
        diffusion_noise=noise,
        x_t=x_t,
    )


@torch.inference_mode()
def posterior_mean_inference(
    *,
    model: nn.Module,
    prepared: PreparedBRLoRAInference,
) -> BRLoRAPosteriorMeanResult:
    """
    Run deterministic inference using fitted BR-LoRA posterior means.

    This is not the same operation as averaging posterior-sampled predictions.
    """

    _validate_inference_model_and_prepared(
        model=model,
        prepared=prepared,
    )

    model.eval()

    disable_variational_sampling(
        model
    )

    prediction = model(
        prepared.model_input,
        prepared.timestep,
        prepared.condition,
    )

    _validate_prediction(
        prediction,
        target=prepared.target,
    )

    composite = _hard_composite(
        base=prepared.target,
        prediction=prediction,
        mask=prepared.mask,
    )

    return BRLoRAPosteriorMeanResult(
        prediction=prediction,
        composite=composite,
        prepared=prepared,
    )


@torch.inference_mode()
def posterior_sample_inference(
    *,
    model: nn.Module,
    prepared: PreparedBRLoRAInference,
    posterior_samples: int,
) -> BRLoRAPosteriorSamplesResult:
    """
    Draw independent BR-LoRA posterior realizations for one fixed input.

    The same ``prepared`` model input, timestep, and diffusion noise are reused
    for every realization. Consequently, differences across the returned sample
    stack arise from sampled BR-LoRA adapter parameters.

    Variance uses the direct Monte Carlo second central moment
    (``correction=0`` / population convention). No reliability interpretation
    is attached to these summaries.
    """

    _validate_inference_model_and_prepared(
        model=model,
        prepared=prepared,
    )

    sample_count = _require_positive_integer(
        posterior_samples,
        name="posterior_samples",
    )

    model.eval()

    enable_variational_sampling(
        model
    )

    predictions: list[
        Tensor
    ] = []

    composites: list[
        Tensor
    ] = []

    for _ in range(
        sample_count
    ):
        prediction = model(
            prepared.model_input,
            prepared.timestep,
            prepared.condition,
        )

        _validate_prediction(
            prediction,
            target=prepared.target,
        )

        composite = _hard_composite(
            base=prepared.target,
            prediction=prediction,
            mask=prepared.mask,
        )

        predictions.append(
            prediction
        )

        composites.append(
            composite
        )

    prediction_samples = torch.stack(
        predictions,
        dim=0,
    )

    composite_samples = torch.stack(
        composites,
        dim=0,
    )

    prediction_mean = (
        prediction_samples.mean(
            dim=0
        )
    )

    prediction_variance = (
        prediction_samples.var(
            dim=0,
            correction=0,
        )
    )

    prediction_std = torch.sqrt(
        prediction_variance
    )

    composite_mean = (
        composite_samples.mean(
            dim=0
        )
    )

    composite_variance = (
        composite_samples.var(
            dim=0,
            correction=0,
        )
    )

    composite_std = torch.sqrt(
        composite_variance
    )

    return BRLoRAPosteriorSamplesResult(
        prediction_samples=(
            prediction_samples
        ),
        composite_samples=(
            composite_samples
        ),
        prediction_mean=(
            prediction_mean
        ),
        prediction_variance=(
            prediction_variance
        ),
        prediction_std=(
            prediction_std
        ),
        composite_mean=(
            composite_mean
        ),
        composite_variance=(
            composite_variance
        ),
        composite_std=(
            composite_std
        ),
        prepared=prepared,
        posterior_samples=(
            sample_count
        ),
    )


def _build_backbone_from_config(
    config: Mapping[str, Any],
    *,
    device: torch.device,
) -> AppearanceX0UNet:
    """Construct the validated local backbone from checkpoint metadata."""

    return AppearanceX0UNet(
        in_ch=_require_positive_integer(
            config.get(
                "in_channels",
                4,
            ),
            name="model_config.in_channels",
        ),
        out_ch=_require_positive_integer(
            config.get(
                "out_channels",
                1,
            ),
            name="model_config.out_channels",
        ),
        base=_require_positive_integer(
            config.get(
                "base_channels",
                32,
            ),
            name="model_config.base_channels",
        ),
        time_dim=_require_positive_integer(
            config.get(
                "time_dim",
                128,
            ),
            name="model_config.time_dim",
        ),
        cond_dim=_require_positive_integer(
            config.get(
                "cond_dim",
                4,
            ),
            name="model_config.cond_dim",
        ),
    ).to(
        device
    )


def _target_layers_from_config(
    config: Mapping[str, Any],
) -> tuple[
    str,
    ...,
]:
    """Return validated BR-LoRA target layers from checkpoint metadata."""

    value = config.get(
        "target_layers"
    )

    if not isinstance(
        value,
        (
            list,
            tuple,
        ),
    ):
        raise BRLoRAInferenceError(
            "br_lora_config.target_layers must be a list or tuple."
        )

    names = tuple(
        str(
            name
        )
        for name in value
    )

    if not names:
        raise BRLoRAInferenceError(
            "br_lora_config.target_layers must not be empty."
        )

    if len(
        set(
            names
        )
    ) != len(
        names
    ):
        raise BRLoRAInferenceError(
            "br_lora_config.target_layers must not contain duplicates."
        )

    return names


def _validate_inference_model_and_prepared(
    *,
    model: nn.Module,
    prepared: PreparedBRLoRAInference,
) -> None:
    """Validate the shared BR-LoRA inference contract."""

    if not isinstance(
        model,
        nn.Module,
    ):
        raise TypeError(
            "`model` must be a torch.nn.Module."
        )

    if not isinstance(
        prepared,
        PreparedBRLoRAInference,
    ):
        raise TypeError(
            "`prepared` must be a PreparedBRLoRAInference."
        )

    modules = (
        iter_variational_lora_modules(
            model
        )
    )

    if not modules:
        raise BRLoRAInferenceError(
            "The model contains no BR-LoRA variational adapters."
        )

    if (
        prepared.model_input.device
        != next(
            model.parameters()
        ).device
    ):
        raise BRLoRAInferenceError(
            "Prepared inference tensors and model must be on the same device."
        )


def _validate_prediction(
    prediction: Tensor,
    *,
    target: Tensor,
) -> None:
    """Validate one BR-LoRA model prediction."""

    if not isinstance(
        prediction,
        Tensor,
    ):
        raise BRLoRAInferenceError(
            "BR-LoRA model must return a torch.Tensor."
        )

    if prediction.shape != target.shape:
        raise BRLoRAInferenceError(
            "BR-LoRA prediction and target shapes must match."
        )

    if not torch.isfinite(
        prediction
    ).all():
        raise BRLoRAInferenceError(
            "BR-LoRA prediction contains non-finite values."
        )


def _hard_composite(
    *,
    base: Tensor,
    prediction: Tensor,
    mask: Tensor,
) -> Tensor:
    """Apply exact hard regional composition."""

    return (
        base
        * (
            1.0
            - mask
        )
        + prediction
        * mask
    )


def _require_finite_float(
    value: Any,
    *,
    name: str,
) -> float:
    """Validate one finite real scalar."""

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
        raise BRLoRAInferenceError(
            f"`{name}` must be finite."
        )

    return converted


def _require_positive_finite_float(
    value: Any,
    *,
    name: str,
) -> float:
    """Validate one finite positive real scalar."""

    converted = _require_finite_float(
        value,
        name=name,
    )

    if converted <= 0.0:
        raise BRLoRAInferenceError(
            f"`{name}` must be positive."
        )

    return converted


def _require_probability_below_one(
    value: Any,
    *,
    name: str,
) -> float:
    """Validate a scalar in [0, 1)."""

    converted = _require_finite_float(
        value,
        name=name,
    )

    if not (
        0.0
        <= converted
        < 1.0
    ):
        raise BRLoRAInferenceError(
            f"`{name}` must satisfy 0 <= value < 1."
        )

    return converted


def _require_fraction(
    value: Any,
    *,
    name: str,
) -> float:
    """Validate one timestep fraction in [0, 1)."""

    return _require_probability_below_one(
        value,
        name=name,
    )


def _require_positive_integer(
    value: Any,
    *,
    name: str,
) -> int:
    """Validate one positive integer."""

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
        raise BRLoRAInferenceError(
            f"`{name}` must be positive."
        )

    return value


__all__ = [
    "BRLoRAInferenceError",
    "BRLoRAPosteriorMeanResult",
    "BRLoRAPosteriorSamplesResult",
    "LoadedBRLoRA",
    "PreparedBRLoRAInference",
    "load_fitted_br_lora",
    "posterior_mean_inference",
    "posterior_sample_inference",
    "prepare_br_lora_batch",
]
