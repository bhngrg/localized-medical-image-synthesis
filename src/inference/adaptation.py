"""
Inference utilities for fitted deterministic adaptation comparators.

Supported methods:
- Regional LoRA
- DoRA
- LoKr
- BitFit

Fitted models are reconstructed entirely from checkpoint metadata. Ordinary
inference restores model state only; optimizer and RNG state are not restored.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import Tensor, nn

from src.diffusion import DiffusionSchedule
from src.models.adapters import (
    configure_bitfit,
    configure_dora,
    configure_lokr,
    configure_regional_lora,
    count_parameters,
    iter_dora_modules,
    iter_lokr_modules,
    iter_lora_modules,
    trainable_parameter_names,
)
from src.training.adaptation_setup import (
    build_backbone,
)


SUPPORTED_ADAPTATION_METHODS = (
    "regional_lora",
    "dora",
    "lokr",
    "bitfit",
)


class AdaptationInferenceError(RuntimeError):
    """Raised when deterministic adaptation inference is invalid."""


@dataclass(frozen=True)
class LoadedAdaptation:
    """Strictly restored fitted deterministic adaptation model."""

    model: nn.Module
    checkpoint: dict[str, Any]
    adaptation_method: str
    adapted_module_names: tuple[str, ...]
    trainable_parameter_count: int
    trainable_tensor_count: int


@dataclass(frozen=True)
class PreparedAdaptationInference:
    """Fixed diffusion input for deterministic adaptation inference."""

    model_input: Tensor
    target: Tensor
    known: Tensor
    mask: Tensor
    donor_patch: Tensor
    condition: Tensor
    timestep: Tensor
    diffusion_noise: Tensor
    x_t: Tensor


@dataclass(frozen=True)
class AdaptationInferenceResult:
    """One deterministic prediction and its hard regional composite."""

    prediction: Tensor
    composite: Tensor
    prepared: PreparedAdaptationInference


def load_fitted_adaptation(
    checkpoint_path: str | Path,
    *,
    device: torch.device,
) -> LoadedAdaptation:
    """
    Reconstruct and strictly restore one fitted deterministic comparator.

    The checkpoint's model and adaptation metadata are authoritative.
    Optimizer and RNG state are intentionally ignored for inference.
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
            f"Deterministic adaptation checkpoint not found:\n{path}"
        )

    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=False,
    )

    if not isinstance(
        checkpoint,
        dict,
    ):
        raise AdaptationInferenceError(
            "Deterministic adaptation checkpoint must contain a dictionary."
        )

    required = {
        "model_state_dict",
        "adaptation_method",
        "adaptation_config",
        "model_config",
        "trainable_parameter_count",
        "trainable_tensor_count",
    }

    missing = sorted(
        required
        - set(
            checkpoint
        )
    )

    if missing:
        raise AdaptationInferenceError(
            "Deterministic adaptation checkpoint is missing required key(s): "
            + ", ".join(
                missing
            )
        )

    method = checkpoint[
        "adaptation_method"
    ]

    if not isinstance(
        method,
        str,
    ):
        raise AdaptationInferenceError(
            "Checkpoint adaptation_method must be a string."
        )

    if method not in SUPPORTED_ADAPTATION_METHODS:
        raise AdaptationInferenceError(
            f"Unsupported deterministic adaptation method: {method!r}"
        )

    adaptation_config = checkpoint[
        "adaptation_config"
    ]

    if not isinstance(
        adaptation_config,
        Mapping,
    ):
        raise AdaptationInferenceError(
            "Checkpoint adaptation_config must be a mapping."
        )

    model_config = checkpoint[
        "model_config"
    ]

    if not isinstance(
        model_config,
        Mapping,
    ):
        raise AdaptationInferenceError(
            "Checkpoint model_config must be a mapping."
        )

    model = build_backbone(
        model_cfg=dict(
            model_config
        ),
        device=device,
    )

    adapted_module_names = _configure_from_checkpoint(
        model=model,
        method=method,
        config=adaptation_config,
    )

    load_result = model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ],
        strict=True,
    )

    if (
        load_result.missing_keys
        or load_result.unexpected_keys
    ):
        raise AdaptationInferenceError(
            "Strict deterministic adaptation checkpoint restoration "
            "reported state-dict mismatches."
        )

    _validate_restored_inventory(
        model=model,
        method=method,
        expected_module_names=adapted_module_names,
    )

    actual_parameter_count = count_parameters(
        model,
        trainable_only=True,
    )

    expected_parameter_count = _require_nonnegative_integer(
        checkpoint[
            "trainable_parameter_count"
        ],
        name="trainable_parameter_count",
    )

    if (
        actual_parameter_count
        != expected_parameter_count
    ):
        raise AdaptationInferenceError(
            "Restored trainable parameter count does not match checkpoint "
            "metadata.\n"
            f"Checkpoint: {expected_parameter_count}\n"
            f"Current:    {actual_parameter_count}"
        )

    actual_tensor_count = len(
        trainable_parameter_names(
            model
        )
    )

    expected_tensor_count = _require_nonnegative_integer(
        checkpoint[
            "trainable_tensor_count"
        ],
        name="trainable_tensor_count",
    )

    if (
        actual_tensor_count
        != expected_tensor_count
    ):
        raise AdaptationInferenceError(
            "Restored trainable tensor count does not match checkpoint "
            "metadata.\n"
            f"Checkpoint: {expected_tensor_count}\n"
            f"Current:    {actual_tensor_count}"
        )

    model.eval()

    return LoadedAdaptation(
        model=model,
        checkpoint=checkpoint,
        adaptation_method=method,
        adapted_module_names=adapted_module_names,
        trainable_parameter_count=actual_parameter_count,
        trainable_tensor_count=actual_tensor_count,
    )


def prepare_adaptation_batch(
    batch: dict,
    *,
    schedule: DiffusionSchedule,
    device: torch.device,
    timestep_fraction: float = 0.75,
    max_samples: int | None = None,
    diffusion_noise: Tensor | None = None,
) -> PreparedAdaptationInference:
    """
    Prepare one fixed reconstruction-style diffusion input.

    When ``diffusion_noise`` is omitted, exactly one fresh Gaussian tensor is
    drawn. Supplying a noise tensor permits identical diffusion inputs across
    deterministic and Bayesian comparator inference.
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
        raise AdaptationInferenceError(
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
        raise AdaptationInferenceError(
            "`x0` must have shape [B, C, H, W]."
        )

    if target.shape[0] <= 0:
        raise AdaptationInferenceError(
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
            raise AdaptationInferenceError(
                f"`{name}` must have the same shape as `x0`."
            )

    if condition.ndim != 2:
        raise AdaptationInferenceError(
            "`cond` must have shape [B, cond_dim]."
        )

    if condition.shape[0] != target.shape[0]:
        raise AdaptationInferenceError(
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
            raise AdaptationInferenceError(
                "`diffusion_noise` must have the same shape as `x0`."
            )

        noise = diffusion_noise.to(
            device=device,
            dtype=target.dtype,
        )

    if not torch.isfinite(
        noise
    ).all():
        raise AdaptationInferenceError(
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

    return PreparedAdaptationInference(
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
def adaptation_inference(
    *,
    model: nn.Module,
    prepared: PreparedAdaptationInference,
) -> AdaptationInferenceResult:
    """Run one deterministic adapted-model forward and hard composition."""

    if not isinstance(
        model,
        nn.Module,
    ):
        raise TypeError(
            "`model` must be a torch.nn.Module."
        )

    if not isinstance(
        prepared,
        PreparedAdaptationInference,
    ):
        raise TypeError(
            "`prepared` must be a PreparedAdaptationInference."
        )

    model_device = next(
        model.parameters()
    ).device

    if (
        prepared.model_input.device
        != model_device
    ):
        raise AdaptationInferenceError(
            "Prepared inference tensors and model must be on the same device."
        )

    model.eval()

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

    return AdaptationInferenceResult(
        prediction=prediction,
        composite=composite,
        prepared=prepared,
    )


def _configure_from_checkpoint(
    *,
    model: nn.Module,
    method: str,
    config: Mapping[str, Any],
) -> tuple[str, ...]:
    """Reconstruct the exact deterministic adaptation topology."""

    if method == "bitfit":
        configure_bitfit(
            model
        )

        return ()

    target_layers = _target_layers_from_config(
        config,
        method=method,
    )

    rank = _require_positive_integer(
        config.get(
            "rank"
        ),
        name=f"{method}.rank",
    )

    alpha = _require_positive_finite_float(
        config.get(
            "alpha"
        ),
        name=f"{method}.alpha",
    )

    dropout = _require_probability_below_one(
        config.get(
            "dropout",
            0.0,
        ),
        name=f"{method}.dropout",
    )

    if method == "regional_lora":
        return configure_regional_lora(
            model,
            target_layers=target_layers,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
        )

    if method == "lokr":
        return configure_lokr(
            model,
            target_layers=target_layers,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
        )

    if method == "dora":
        eps = _require_positive_finite_float(
            config.get(
                "eps",
                1.0e-6,
            ),
            name="dora.eps",
        )

        return configure_dora(
            model,
            target_layers=target_layers,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            eps=eps,
        )

    raise AdaptationInferenceError(
        f"Unsupported deterministic adaptation method: {method!r}"
    )


def _validate_restored_inventory(
    *,
    model: nn.Module,
    method: str,
    expected_module_names: tuple[str, ...],
) -> None:
    """Verify method-specific adapted-module inventory after restoration."""

    if method == "regional_lora":
        current = tuple(
            name
            for name, _
            in iter_lora_modules(
                model
            )
        )

    elif method == "dora":
        current = tuple(
            name
            for name, _
            in iter_dora_modules(
                model
            )
        )

    elif method == "lokr":
        current = tuple(
            name
            for name, _
            in iter_lokr_modules(
                model
            )
        )

    elif method == "bitfit":
        current = ()

    else:
        raise AdaptationInferenceError(
            f"Unsupported deterministic adaptation method: {method!r}"
        )

    if current != expected_module_names:
        raise AdaptationInferenceError(
            "Restored deterministic adaptation module inventory does not "
            "match checkpoint configuration.\n"
            f"Expected: {expected_module_names}\n"
            f"Current:  {current}"
        )


def _target_layers_from_config(
    config: Mapping[str, Any],
    *,
    method: str,
) -> tuple[str, ...]:
    """Return validated target layers from checkpoint adaptation metadata."""

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
        raise AdaptationInferenceError(
            f"{method}.target_layers must be a list or tuple."
        )

    names = tuple(
        str(
            name
        )
        for name in value
    )

    if not names:
        raise AdaptationInferenceError(
            f"{method}.target_layers must not be empty."
        )

    if len(
        set(
            names
        )
    ) != len(
        names
    ):
        raise AdaptationInferenceError(
            f"{method}.target_layers must not contain duplicates."
        )

    return names


def _validate_prediction(
    prediction: Tensor,
    *,
    target: Tensor,
) -> None:
    """Validate one deterministic comparator prediction."""

    if not isinstance(
        prediction,
        Tensor,
    ):
        raise AdaptationInferenceError(
            "Deterministic adaptation model must return a torch.Tensor."
        )

    if prediction.shape != target.shape:
        raise AdaptationInferenceError(
            "Deterministic adaptation prediction and target shapes must match."
        )

    if not torch.isfinite(
        prediction
    ).all():
        raise AdaptationInferenceError(
            "Deterministic adaptation prediction contains non-finite values."
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


def _require_positive_integer(
    value: Any,
    *,
    name: str,
) -> int:
    """Return one strictly positive integer."""

    if (
        isinstance(
            value,
            bool,
        )
        or not isinstance(
            value,
            int,
        )
        or value <= 0
    ):
        raise AdaptationInferenceError(
            f"{name} must be a positive integer."
        )

    return value


def _require_nonnegative_integer(
    value: Any,
    *,
    name: str,
) -> int:
    """Return one non-negative integer."""

    if (
        isinstance(
            value,
            bool,
        )
        or not isinstance(
            value,
            int,
        )
        or value < 0
    ):
        raise AdaptationInferenceError(
            f"{name} must be a non-negative integer."
        )

    return value


def _require_positive_finite_float(
    value: Any,
    *,
    name: str,
) -> float:
    """Return one strictly positive finite float."""

    try:
        result = float(
            value
        )

    except (
        TypeError,
        ValueError,
    ) as error:
        raise AdaptationInferenceError(
            f"{name} must be a finite positive number."
        ) from error

    if (
        not math.isfinite(
            result
        )
        or result <= 0.0
    ):
        raise AdaptationInferenceError(
            f"{name} must be a finite positive number."
        )

    return result


def _require_probability_below_one(
    value: Any,
    *,
    name: str,
) -> float:
    """Return one probability in [0, 1)."""

    try:
        result = float(
            value
        )

    except (
        TypeError,
        ValueError,
    ) as error:
        raise AdaptationInferenceError(
            f"{name} must satisfy 0 <= value < 1."
        ) from error

    if (
        not math.isfinite(
            result
        )
        or result < 0.0
        or result >= 1.0
    ):
        raise AdaptationInferenceError(
            f"{name} must satisfy 0 <= value < 1."
        )

    return result


def _require_fraction(
    value: Any,
    *,
    name: str,
) -> float:
    """Return one finite fraction in [0, 1)."""

    return _require_probability_below_one(
        value,
        name=name,
    )


__all__ = [
    "AdaptationInferenceError",
    "AdaptationInferenceResult",
    "LoadedAdaptation",
    "PreparedAdaptationInference",
    "SUPPORTED_ADAPTATION_METHODS",
    "adaptation_inference",
    "load_fitted_adaptation",
    "prepare_adaptation_batch",
]
