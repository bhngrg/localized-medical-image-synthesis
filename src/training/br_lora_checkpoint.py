"""
Checkpoint utilities for Bayesian Regional LoRA (BR-LoRA).

BR-LoRA checkpoints preserve the complete fitted model state, including

    frozen backbone parameters
    posterior means
    posterior rho parameters
    prior buffers

together with optimizer state, training progress, configuration metadata,
optional training history, and random-number-generator state.

Individual posterior realizations are not stored. They can be regenerated
from the fitted posterior after checkpoint restoration.

This module is deliberately independent of epoch orchestration and command-line
interfaces.
"""

from __future__ import annotations

import copy
import math
from pathlib import Path
import random
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn

from src.models.adapters import (
    iter_variational_lora_modules,
    variational_lora_parameter_count,
)


BR_LORA_CHECKPOINT_SCHEMA_VERSION = 1

BR_LORA_TRAINING_MODE = (
    "bayesian_regional_lora"
)


class BRLoRACheckpointError(
    ValueError
):
    """Raised when a BR-LoRA checkpoint is invalid."""


def capture_rng_state() -> dict[str, Any]:
    """
    Capture random-number-generator state used by the training workflow.

    Python, NumPy, CPU Torch, CUDA, and Apple MPS streams are retained when
    available so checkpoint resumption can reproduce stochastic training.
    """

    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": None,
        "torch_mps": None,
    }

    if torch.cuda.is_available():
        state[
            "torch_cuda"
        ] = torch.cuda.get_rng_state_all()

    if (
        hasattr(
            torch,
            "mps",
        )
        and torch.backends.mps.is_available()
        and hasattr(
            torch.mps,
            "get_rng_state",
        )
    ):
        state[
            "torch_mps"
        ] = torch.mps.get_rng_state()

    return state


def restore_rng_state(
    state: Mapping[str, Any],
) -> None:
    """Restore RNG state previously produced by ``capture_rng_state``."""

    if not isinstance(
        state,
        Mapping,
    ):
        raise TypeError(
            "`state` must be a mapping."
        )

    required_keys = (
        "python",
        "numpy",
        "torch_cpu",
        "torch_cuda",
        "torch_mps",
    )

    missing_keys = tuple(
        key
        for key in required_keys
        if key not in state
    )

    if missing_keys:
        raise BRLoRACheckpointError(
            "RNG state is missing required keys: "
            + ", ".join(
                missing_keys
            )
        )

    random.setstate(
        state[
            "python"
        ]
    )

    np.random.set_state(
        state[
            "numpy"
        ]
    )

    torch_cpu_state = state[
        "torch_cpu"
    ]

    if not isinstance(
        torch_cpu_state,
        torch.Tensor,
    ):
        raise BRLoRACheckpointError(
            "`torch_cpu` RNG state must be a torch.Tensor."
        )

    torch.set_rng_state(
        torch_cpu_state.cpu()
    )

    torch_cuda_state = state[
        "torch_cuda"
    ]

    if (
        torch_cuda_state is not None
        and torch.cuda.is_available()
    ):
        if not isinstance(
            torch_cuda_state,
            (
                list,
                tuple,
            ),
        ):
            raise BRLoRACheckpointError(
                "`torch_cuda` RNG state must be a sequence or None."
            )

        torch.cuda.set_rng_state_all(
            list(
                torch_cuda_state
            )
        )

    torch_mps_state = state[
        "torch_mps"
    ]

    if (
        torch_mps_state is not None
        and hasattr(
            torch,
            "mps",
        )
        and torch.backends.mps.is_available()
        and hasattr(
            torch.mps,
            "set_rng_state",
        )
    ):
        if not isinstance(
            torch_mps_state,
            torch.Tensor,
        ):
            raise BRLoRACheckpointError(
                "`torch_mps` RNG state must be a torch.Tensor or None."
            )

        torch.mps.set_rng_state(
            torch_mps_state.cpu()
        )


def build_br_lora_checkpoint_payload(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    completed_epochs: int,
    global_step: int,
    best_validation_loss: float | None,
    br_lora_config: Mapping[str, Any],
    training_config: Mapping[str, Any],
    model_config: Mapping[str, Any] | None = None,
    data_config: Mapping[str, Any] | None = None,
    history: list[dict[str, Any]] | None = None,
    include_rng_state: bool = True,
) -> dict[str, Any]:
    """
    Build one complete BR-LoRA checkpoint payload.

    ``model.state_dict()`` is intentionally stored in full. This guarantees
    that the frozen backbone and fitted Bayesian adapter posterior are restored
    together rather than relying on a separately located backbone checkpoint.
    """

    _validate_model(
        model
    )

    if not isinstance(
        optimizer,
        torch.optim.Optimizer,
    ):
        raise TypeError(
            "`optimizer` must be a torch.optim.Optimizer."
        )

    _validate_nonnegative_integer(
        completed_epochs,
        name="completed_epochs",
    )

    _validate_nonnegative_integer(
        global_step,
        name="global_step",
    )

    validated_best_loss = (
        _validate_optional_finite_float(
            best_validation_loss,
            name="best_validation_loss",
        )
    )

    validated_br_lora_config = (
        _copy_mapping(
            br_lora_config,
            name="br_lora_config",
        )
    )

    validated_training_config = (
        _copy_mapping(
            training_config,
            name="training_config",
        )
    )

    validated_model_config = (
        None
        if model_config is None
        else _copy_mapping(
            model_config,
            name="model_config",
        )
    )

    validated_data_config = (
        None
        if data_config is None
        else _copy_mapping(
            data_config,
            name="data_config",
        )
    )

    validated_history = (
        []
        if history is None
        else _copy_history(
            history
        )
    )

    if not isinstance(
        include_rng_state,
        bool,
    ):
        raise TypeError(
            "`include_rng_state` must be a bool."
        )

    variational_modules = (
        iter_variational_lora_modules(
            model
        )
    )

    variational_module_names = tuple(
        name
        for name, _
        in variational_modules
    )

    variational_parameter_count = (
        variational_lora_parameter_count(
            model,
            trainable_only=False,
        )
    )

    trainable_parameter_names = tuple(
        name
        for name, parameter
        in model.named_parameters()
        if parameter.requires_grad
    )

    payload: dict[str, Any] = {
        "schema_version": (
            BR_LORA_CHECKPOINT_SCHEMA_VERSION
        ),

        "training_mode": (
            BR_LORA_TRAINING_MODE
        ),

        "model_state_dict": copy.deepcopy(
            model.state_dict()
        ),

        "optimizer_state_dict": copy.deepcopy(
            optimizer.state_dict()
        ),

        "completed_epochs": (
            completed_epochs
        ),

        "global_step": (
            global_step
        ),

        "best_validation_loss": (
            validated_best_loss
        ),

        "br_lora_config": (
            validated_br_lora_config
        ),

        "training_config": (
            validated_training_config
        ),

        "model_config": (
            validated_model_config
        ),

        "data_config": (
            validated_data_config
        ),

        "history": (
            validated_history
        ),

        "variational_module_names": (
            variational_module_names
        ),

        "variational_parameter_count": (
            variational_parameter_count
        ),

        "trainable_parameter_names": (
            trainable_parameter_names
        ),

        "rng_state": (
            capture_rng_state()
            if include_rng_state
            else None
        ),
    }

    return payload


def save_br_lora_checkpoint(
    path: str | Path,
    payload: Mapping[str, Any],
) -> None:
    """
    Save a BR-LoRA checkpoint atomically.

    The destination directory is created immediately before writing.
    """

    checkpoint_path = Path(
        path
    )

    checkpoint_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    _validate_checkpoint_payload(
        payload
    )

    temporary_path = (
        checkpoint_path.with_suffix(
            checkpoint_path.suffix
            + ".tmp"
        )
    )

    torch.save(
        dict(
            payload
        ),
        temporary_path,
    )

    temporary_path.replace(
        checkpoint_path
    )


def load_br_lora_checkpoint(
    path: str | Path,
    *,
    map_location: (
        str
        | torch.device
        | None
    ) = None,
) -> dict[str, Any]:
    """Load and validate one BR-LoRA checkpoint payload."""

    checkpoint_path = (
        Path(
            path
        )
        .expanduser()
        .resolve()
    )

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"BR-LoRA checkpoint not found:\n{checkpoint_path}"
        )

    payload = torch.load(
        checkpoint_path,
        map_location=map_location,
        weights_only=False,
    )

    if not isinstance(
        payload,
        dict,
    ):
        raise BRLoRACheckpointError(
            "BR-LoRA checkpoint must contain a dictionary."
        )

    _validate_checkpoint_payload(
        payload
    )

    return payload


def restore_br_lora_checkpoint(
    *,
    payload: Mapping[str, Any],
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    strict: bool = True,
    restore_rng: bool = False,
) -> dict[str, Any]:
    """
    Restore BR-LoRA model state and optional optimizer/RNG state.

    The BR-LoRA adapter structure must already exist in ``model`` before this
    function is called. This function restores state; it does not inject or
    construct adapters.
    """

    _validate_checkpoint_payload(
        payload
    )

    _validate_model(
        model
    )

    if not isinstance(
        strict,
        bool,
    ):
        raise TypeError(
            "`strict` must be a bool."
        )

    if not isinstance(
        restore_rng,
        bool,
    ):
        raise TypeError(
            "`restore_rng` must be a bool."
        )

    expected_module_names = tuple(
        payload[
            "variational_module_names"
        ]
    )

    current_module_names = tuple(
        name
        for name, _
        in iter_variational_lora_modules(
            model
        )
    )

    if (
        current_module_names
        != expected_module_names
    ):
        raise BRLoRACheckpointError(
            "BR-LoRA adapter inventory does not match checkpoint.\n"
            f"Checkpoint: {expected_module_names}\n"
            f"Current:    {current_module_names}"
        )

    current_parameter_count = (
        variational_lora_parameter_count(
            model,
            trainable_only=False,
        )
    )

    checkpoint_parameter_count = int(
        payload[
            "variational_parameter_count"
        ]
    )

    if (
        current_parameter_count
        != checkpoint_parameter_count
    ):
        raise BRLoRACheckpointError(
            "BR-LoRA variational parameter count does not match "
            "checkpoint.\n"
            f"Checkpoint: {checkpoint_parameter_count}\n"
            f"Current:    {current_parameter_count}"
        )

    load_result = model.load_state_dict(
        payload[
            "model_state_dict"
        ],
        strict=strict,
    )

    if optimizer is not None:

        if not isinstance(
            optimizer,
            torch.optim.Optimizer,
        ):
            raise TypeError(
                "`optimizer` must be a torch.optim.Optimizer or None."
            )

        optimizer.load_state_dict(
            payload[
                "optimizer_state_dict"
            ]
        )

    if restore_rng:

        rng_state = payload[
            "rng_state"
        ]

        if rng_state is None:
            raise BRLoRACheckpointError(
                "Checkpoint does not contain RNG state."
            )

        restore_rng_state(
            rng_state
        )

    return {
        "load_result": (
            load_result
        ),

        "completed_epochs": int(
            payload[
                "completed_epochs"
            ]
        ),

        "global_step": int(
            payload[
                "global_step"
            ]
        ),

        "best_validation_loss": (
            payload[
                "best_validation_loss"
            ]
        ),

        "history": list(
            payload[
                "history"
            ]
        ),

        "br_lora_config": dict(
            payload[
                "br_lora_config"
            ]
        ),

        "training_config": dict(
            payload[
                "training_config"
            ]
        ),

        "model_config": (
            None
            if payload[
                "model_config"
            ] is None
            else dict(
                payload[
                    "model_config"
                ]
            )
        ),

        "data_config": (
            None
            if payload[
                "data_config"
            ] is None
            else dict(
                payload[
                    "data_config"
                ]
            )
        ),
    }


def _validate_checkpoint_payload(
    payload: Mapping[str, Any],
) -> None:
    """Validate the structural contract of a BR-LoRA checkpoint."""

    if not isinstance(
        payload,
        Mapping,
    ):
        raise TypeError(
            "`payload` must be a mapping."
        )

    required_keys = (
        "schema_version",
        "training_mode",
        "model_state_dict",
        "optimizer_state_dict",
        "completed_epochs",
        "global_step",
        "best_validation_loss",
        "br_lora_config",
        "training_config",
        "model_config",
        "data_config",
        "history",
        "variational_module_names",
        "variational_parameter_count",
        "trainable_parameter_names",
        "rng_state",
    )

    missing_keys = tuple(
        key
        for key in required_keys
        if key not in payload
    )

    if missing_keys:
        raise BRLoRACheckpointError(
            "BR-LoRA checkpoint is missing required keys: "
            + ", ".join(
                missing_keys
            )
        )

    if (
        payload[
            "schema_version"
        ]
        != BR_LORA_CHECKPOINT_SCHEMA_VERSION
    ):
        raise BRLoRACheckpointError(
            "Unsupported BR-LoRA checkpoint schema version: "
            f"{payload['schema_version']!r}"
        )

    if (
        payload[
            "training_mode"
        ]
        != BR_LORA_TRAINING_MODE
    ):
        raise BRLoRACheckpointError(
            "Unexpected training mode in BR-LoRA checkpoint: "
            f"{payload['training_mode']!r}"
        )

    if not isinstance(
        payload[
            "model_state_dict"
        ],
        Mapping,
    ):
        raise BRLoRACheckpointError(
            "`model_state_dict` must be a mapping."
        )

    if not isinstance(
        payload[
            "optimizer_state_dict"
        ],
        Mapping,
    ):
        raise BRLoRACheckpointError(
            "`optimizer_state_dict` must be a mapping."
        )

    _validate_nonnegative_integer(
        payload[
            "completed_epochs"
        ],
        name="completed_epochs",
    )

    _validate_nonnegative_integer(
        payload[
            "global_step"
        ],
        name="global_step",
    )

    _validate_optional_finite_float(
        payload[
            "best_validation_loss"
        ],
        name="best_validation_loss",
    )

    for name in (
        "br_lora_config",
        "training_config",
    ):

        if not isinstance(
            payload[
                name
            ],
            Mapping,
        ):
            raise BRLoRACheckpointError(
                f"`{name}` must be a mapping."
            )

    for name in (
        "model_config",
        "data_config",
    ):

        value = payload[
            name
        ]

        if (
            value is not None
            and not isinstance(
                value,
                Mapping,
            )
        ):
            raise BRLoRACheckpointError(
                f"`{name}` must be a mapping or None."
            )

    history = payload[
        "history"
    ]

    if not isinstance(
        history,
        list,
    ):
        raise BRLoRACheckpointError(
            "`history` must be a list."
        )

    if not all(
        isinstance(
            record,
            dict,
        )
        for record in history
    ):
        raise BRLoRACheckpointError(
            "Every history record must be a dictionary."
        )

    module_names = payload[
        "variational_module_names"
    ]

    if not isinstance(
        module_names,
        (
            tuple,
            list,
        ),
    ):
        raise BRLoRACheckpointError(
            "`variational_module_names` must be a sequence."
        )

    if not module_names:
        raise BRLoRACheckpointError(
            "BR-LoRA checkpoint contains no variational modules."
        )

    if not all(
        isinstance(
            name,
            str,
        )
        and bool(
            name
        )
        for name in module_names
    ):
        raise BRLoRACheckpointError(
            "Every variational module name must be a non-empty string."
        )

    parameter_count = payload[
        "variational_parameter_count"
    ]

    if (
        isinstance(
            parameter_count,
            bool,
        )
        or not isinstance(
            parameter_count,
            int,
        )
        or parameter_count <= 0
    ):
        raise BRLoRACheckpointError(
            "`variational_parameter_count` must be a positive integer."
        )

    trainable_names = payload[
        "trainable_parameter_names"
    ]

    if not isinstance(
        trainable_names,
        (
            tuple,
            list,
        ),
    ):
        raise BRLoRACheckpointError(
            "`trainable_parameter_names` must be a sequence."
        )

    if not trainable_names:
        raise BRLoRACheckpointError(
            "BR-LoRA checkpoint contains no trainable parameters."
        )

    if not all(
        isinstance(
            name,
            str,
        )
        and bool(
            name
        )
        for name in trainable_names
    ):
        raise BRLoRACheckpointError(
            "Every trainable parameter name must be a non-empty string."
        )

    rng_state = payload[
        "rng_state"
    ]

    if (
        rng_state is not None
        and not isinstance(
            rng_state,
            Mapping,
        )
    ):
        raise BRLoRACheckpointError(
            "`rng_state` must be a mapping or None."
        )


def _validate_model(
    model: nn.Module,
) -> None:
    """Require a model containing BR-LoRA adapters."""

    if not isinstance(
        model,
        nn.Module,
    ):
        raise TypeError(
            "`model` must be a torch.nn.Module."
        )

    modules = (
        iter_variational_lora_modules(
            model
        )
    )

    if not modules:
        raise BRLoRACheckpointError(
            "The model contains no BR-LoRA adapters."
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
        raise BRLoRACheckpointError(
            f"`{name}` must be non-negative."
        )


def _validate_optional_finite_float(
    value: float | None,
    *,
    name: str,
) -> float | None:
    """Validate an optional finite scalar."""

    if value is None:
        return None

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
            f"`{name}` must be a real scalar or None."
        )

    converted = float(
        value
    )

    if not math.isfinite(
        converted
    ):
        raise BRLoRACheckpointError(
            f"`{name}` must be finite when provided."
        )

    return converted


def _copy_mapping(
    value: Mapping[str, Any],
    *,
    name: str,
) -> dict[str, Any]:
    """Validate and shallow-copy one configuration mapping."""

    if not isinstance(
        value,
        Mapping,
    ):
        raise TypeError(
            f"`{name}` must be a mapping."
        )

    return dict(
        value
    )


def _copy_history(
    history: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Validate and copy training-history records."""

    if not isinstance(
        history,
        list,
    ):
        raise TypeError(
            "`history` must be a list."
        )

    copied: list[
        dict[str, Any]
    ] = []

    for index, record in enumerate(
        history
    ):

        if not isinstance(
            record,
            dict,
        ):
            raise TypeError(
                "Every history record must be a dictionary; "
                f"record {index} is {type(record).__name__}."
            )

        copied.append(
            dict(
                record
            )
        )

    return copied


__all__ = [
    "BR_LORA_CHECKPOINT_SCHEMA_VERSION",
    "BR_LORA_TRAINING_MODE",
    "BRLoRACheckpointError",
    "build_br_lora_checkpoint_payload",
    "capture_rng_state",
    "load_br_lora_checkpoint",
    "restore_br_lora_checkpoint",
    "restore_rng_state",
    "save_br_lora_checkpoint",
]