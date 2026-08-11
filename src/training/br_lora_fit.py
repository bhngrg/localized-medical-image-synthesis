"""
Multi-epoch orchestration primitives for Bayesian Regional LoRA (BR-LoRA).

Two explicit protocols are supported through ``data_config["split_mode"]``:

internal
    Train on the internal optimization split, validate after each epoch,
    save ``latest.pt`` every epoch, and select ``best.pt`` by validation loss.

full_train
    Train on all supplied training data for a fixed epoch target, perform no
    validation, save ``latest.pt`` every epoch for exact resume, and write
    ``final.pt`` only when the configured epoch target is reached.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.diffusion import DiffusionSchedule

from .br_lora_checkpoint import (
    build_br_lora_checkpoint_payload,
    load_br_lora_checkpoint,
    save_br_lora_checkpoint,
)
from .br_lora_trainer import (
    BRLoRATrainEpochResult,
    BRLoRAValidationEpochResult,
    train_br_lora_epoch,
    validate_br_lora_epoch,
)

INTERNAL_SPLIT_MODE = "internal"
FULL_TRAIN_SPLIT_MODE = "full_train"
SUPPORTED_SPLIT_MODES = (
    INTERNAL_SPLIT_MODE,
    FULL_TRAIN_SPLIT_MODE,
)


class BRLoRAFitError(ValueError):
    """Raised when BR-LoRA fit orchestration inputs are invalid."""


@dataclass(frozen=True, slots=True)
class BRLoRAFitConfig:
    """Static scientific settings for one BR-LoRA training run."""

    epochs: int
    kl_weight: float = 1.0e-6
    kl_warmup_steps: int = 1000
    outside_loss_weight: float = 0.05
    max_grad_norm: float | None = 1.0
    training_sample_posterior: bool = True
    validation_sample_posterior: bool = False

    def __post_init__(self) -> None:
        _require_positive_integer(self.epochs, name="epochs")
        _require_nonnegative_finite_float(self.kl_weight, name="kl_weight")
        _require_nonnegative_integer(
            self.kl_warmup_steps,
            name="kl_warmup_steps",
        )
        _require_nonnegative_finite_float(
            self.outside_loss_weight,
            name="outside_loss_weight",
        )

        if self.max_grad_norm is not None:
            _require_positive_finite_float(
                self.max_grad_norm,
                name="max_grad_norm",
            )

        if not isinstance(self.training_sample_posterior, bool):
            raise TypeError(
                "`training_sample_posterior` must be a bool."
            )

        if not isinstance(self.validation_sample_posterior, bool):
            raise TypeError(
                "`validation_sample_posterior` must be a bool."
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "epochs": self.epochs,
            "kl_weight": self.kl_weight,
            "kl_warmup_steps": self.kl_warmup_steps,
            "outside_loss_weight": self.outside_loss_weight,
            "max_grad_norm": self.max_grad_norm,
            "training_sample_posterior": self.training_sample_posterior,
            "validation_sample_posterior": self.validation_sample_posterior,
        }


@dataclass(frozen=True, slots=True)
class BRLoRAFitState:
    """Progress state required to continue BR-LoRA training."""

    completed_epochs: int = 0
    global_step: int = 0
    best_validation_loss: float | None = None
    best_completed_epoch: int | None = None

    def __post_init__(self) -> None:
        _require_nonnegative_integer(
            self.completed_epochs,
            name="completed_epochs",
        )
        _require_nonnegative_integer(
            self.global_step,
            name="global_step",
        )

        if self.best_validation_loss is not None:
            _require_finite_float(
                self.best_validation_loss,
                name="best_validation_loss",
            )

        if self.best_completed_epoch is not None:
            _require_positive_integer(
                self.best_completed_epoch,
                name="best_completed_epoch",
            )
            if self.best_completed_epoch > self.completed_epochs:
                raise BRLoRAFitError(
                    "`best_completed_epoch` cannot exceed "
                    "`completed_epochs`."
                )

        if (
            self.best_validation_loss is None
            and self.best_completed_epoch is not None
        ):
            raise BRLoRAFitError(
                "`best_completed_epoch` requires `best_validation_loss`."
            )

        if (
            self.best_validation_loss is not None
            and self.best_completed_epoch is None
        ):
            raise BRLoRAFitError(
                "`best_validation_loss` requires `best_completed_epoch`."
            )


@dataclass(frozen=True, slots=True)
class BRLoRAEpochRecord:
    """Serializable metrics from one completed BR-LoRA epoch."""

    completed_epoch: int
    global_step: int

    training_samples: int
    training_batches: int

    training_loss: float
    training_reconstruction: float
    training_inside: float
    training_outside: float
    training_kl: float
    training_normalized_kl: float

    validation_samples: int | None = None
    validation_batches: int | None = None

    validation_loss: float | None = None
    validation_reconstruction: float | None = None
    validation_inside: float | None = None
    validation_outside: float | None = None
    validation_kl: float | None = None
    validation_normalized_kl: float | None = None

    validation_sample_posterior: bool | None = None

    def __post_init__(self) -> None:
        _require_positive_integer(
            self.completed_epoch,
            name="completed_epoch",
        )
        _require_positive_integer(
            self.global_step,
            name="global_step",
        )
        _require_positive_integer(
            self.training_samples,
            name="training_samples",
        )
        _require_positive_integer(
            self.training_batches,
            name="training_batches",
        )

        for name, value in (
            ("training_loss", self.training_loss),
            ("training_reconstruction", self.training_reconstruction),
            ("training_inside", self.training_inside),
            ("training_outside", self.training_outside),
            ("training_kl", self.training_kl),
            ("training_normalized_kl", self.training_normalized_kl),
        ):
            _require_finite_float(value, name=name)

        validation_values = (
            self.validation_samples,
            self.validation_batches,
            self.validation_loss,
            self.validation_reconstruction,
            self.validation_inside,
            self.validation_outside,
            self.validation_kl,
            self.validation_normalized_kl,
            self.validation_sample_posterior,
        )
        present = tuple(value is not None for value in validation_values)

        if any(present) and not all(present):
            raise BRLoRAFitError(
                "Validation fields must either all be populated or all be None."
            )

        if all(present):
            _require_positive_integer(
                self.validation_samples,
                name="validation_samples",
            )
            _require_positive_integer(
                self.validation_batches,
                name="validation_batches",
            )
            for name, value in (
                ("validation_loss", self.validation_loss),
                ("validation_reconstruction", self.validation_reconstruction),
                ("validation_inside", self.validation_inside),
                ("validation_outside", self.validation_outside),
                ("validation_kl", self.validation_kl),
                (
                    "validation_normalized_kl",
                    self.validation_normalized_kl,
                ),
            ):
                _require_finite_float(value, name=name)

            if not isinstance(self.validation_sample_posterior, bool):
                raise TypeError(
                    "`validation_sample_posterior` must be a bool "
                    "when validation metrics are present."
                )

    @property
    def has_validation(self) -> bool:
        return self.validation_loss is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "completed_epoch": self.completed_epoch,
            "global_step": self.global_step,
            "training_samples": self.training_samples,
            "training_batches": self.training_batches,
            "training_loss": self.training_loss,
            "training_reconstruction": self.training_reconstruction,
            "training_inside": self.training_inside,
            "training_outside": self.training_outside,
            "training_kl": self.training_kl,
            "training_normalized_kl": self.training_normalized_kl,
            "validation_samples": self.validation_samples,
            "validation_batches": self.validation_batches,
            "validation_loss": self.validation_loss,
            "validation_reconstruction": self.validation_reconstruction,
            "validation_inside": self.validation_inside,
            "validation_outside": self.validation_outside,
            "validation_kl": self.validation_kl,
            "validation_normalized_kl": self.validation_normalized_kl,
            "validation_sample_posterior": self.validation_sample_posterior,
        }


@dataclass(frozen=True, slots=True)
class BRLoRAFitResult:
    """Final structured result returned by BR-LoRA orchestration."""

    state: BRLoRAFitState
    history: tuple[BRLoRAEpochRecord, ...]
    split_mode: str

    latest_checkpoint_path: str
    best_checkpoint_path: str | None = None
    final_checkpoint_path: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.state, BRLoRAFitState):
            raise TypeError("`state` must be a BRLoRAFitState.")

        if not isinstance(self.history, tuple):
            raise TypeError("`history` must be a tuple.")

        if not all(
            isinstance(record, BRLoRAEpochRecord)
            for record in self.history
        ):
            raise TypeError(
                "Every history entry must be a BRLoRAEpochRecord."
            )

        if len(self.history) != self.state.completed_epochs:
            raise BRLoRAFitError(
                "History length must equal completed epoch count."
            )

        _validate_split_mode(self.split_mode)

        if not isinstance(self.latest_checkpoint_path, str):
            raise TypeError(
                "`latest_checkpoint_path` must be a string."
            )

        if (
            self.best_checkpoint_path is not None
            and not isinstance(self.best_checkpoint_path, str)
        ):
            raise TypeError(
                "`best_checkpoint_path` must be a string or None."
            )

        if (
            self.final_checkpoint_path is not None
            and not isinstance(self.final_checkpoint_path, str)
        ):
            raise TypeError(
                "`final_checkpoint_path` must be a string or None."
            )

        if self.split_mode == INTERNAL_SPLIT_MODE:
            if self.best_checkpoint_path is None:
                raise BRLoRAFitError(
                    "Internal training requires a best checkpoint path."
                )
            if self.final_checkpoint_path is not None:
                raise BRLoRAFitError(
                    "Internal training must not report a final checkpoint path."
                )
        else:
            if self.best_checkpoint_path is not None:
                raise BRLoRAFitError(
                    "Full training must not report a best checkpoint path."
                )
            if self.final_checkpoint_path is None:
                raise BRLoRAFitError(
                    "Full training requires a final checkpoint path."
                )


def history_to_dicts(
    history: tuple[BRLoRAEpochRecord, ...],
) -> list[dict[str, Any]]:
    if not isinstance(history, tuple):
        raise TypeError("`history` must be a tuple.")

    records: list[dict[str, Any]] = []

    for record in history:
        if not isinstance(record, BRLoRAEpochRecord):
            raise TypeError(
                "Every history entry must be a BRLoRAEpochRecord."
            )
        records.append(record.to_dict())

    return records


def history_from_dicts(
    history: list[dict[str, Any]],
) -> tuple[BRLoRAEpochRecord, ...]:
    if not isinstance(history, list):
        raise TypeError("`history` must be a list.")

    records: list[BRLoRAEpochRecord] = []

    for index, record in enumerate(history):
        if not isinstance(record, dict):
            raise TypeError(
                "Every history entry must be a dictionary; "
                f"entry {index} is {type(record).__name__}."
            )

        try:
            restored = BRLoRAEpochRecord(**record)
        except (TypeError, ValueError) as error:
            raise BRLoRAFitError(
                "Invalid BR-LoRA history record at "
                f"index {index}: {error}"
            ) from error

        records.append(restored)

    restored_history = tuple(records)
    _validate_history_sequence(restored_history)
    _validate_history_mode_consistency(restored_history)
    return restored_history


def build_epoch_record(
    *,
    completed_epoch: int,
    training_result: BRLoRATrainEpochResult,
    validation_result: BRLoRAValidationEpochResult,
) -> BRLoRAEpochRecord:
    """Build one internal-validation history record."""

    _require_positive_integer(completed_epoch, name="completed_epoch")

    if not isinstance(training_result, BRLoRATrainEpochResult):
        raise TypeError(
            "`training_result` must be a BRLoRATrainEpochResult."
        )

    if not isinstance(validation_result, BRLoRAValidationEpochResult):
        raise TypeError(
            "`validation_result` must be a BRLoRAValidationEpochResult."
        )

    if validation_result.step != training_result.next_step:
        raise BRLoRAFitError(
            "Validation step must equal the training result's "
            "next global step."
        )

    training_metrics = training_result.metrics
    validation_metrics = validation_result.metrics

    if training_metrics.batches != training_result.batches_processed:
        raise BRLoRAFitError(
            "Training metrics and training result disagree about "
            "the number of processed batches."
        )

    if validation_metrics.batches != validation_result.batches_processed:
        raise BRLoRAFitError(
            "Validation metrics and validation result disagree about "
            "the number of processed batches."
        )

    return BRLoRAEpochRecord(
        completed_epoch=completed_epoch,
        global_step=training_result.next_step,
        training_samples=training_metrics.samples,
        training_batches=training_metrics.batches,
        training_loss=training_metrics.loss,
        training_reconstruction=training_metrics.reconstruction,
        training_inside=training_metrics.inside,
        training_outside=training_metrics.outside,
        training_kl=training_metrics.kl,
        training_normalized_kl=training_metrics.normalized_kl,
        validation_samples=validation_metrics.samples,
        validation_batches=validation_metrics.batches,
        validation_loss=validation_metrics.loss,
        validation_reconstruction=validation_metrics.reconstruction,
        validation_inside=validation_metrics.inside,
        validation_outside=validation_metrics.outside,
        validation_kl=validation_metrics.kl,
        validation_normalized_kl=validation_metrics.normalized_kl,
        validation_sample_posterior=validation_result.sample_posterior,
    )


def build_full_train_epoch_record(
    *,
    completed_epoch: int,
    training_result: BRLoRATrainEpochResult,
) -> BRLoRAEpochRecord:
    """Build one full-training record without fabricated validation."""

    _require_positive_integer(completed_epoch, name="completed_epoch")

    if not isinstance(training_result, BRLoRATrainEpochResult):
        raise TypeError(
            "`training_result` must be a BRLoRATrainEpochResult."
        )

    metrics = training_result.metrics

    if metrics.batches != training_result.batches_processed:
        raise BRLoRAFitError(
            "Training metrics and training result disagree about "
            "the number of processed batches."
        )

    return BRLoRAEpochRecord(
        completed_epoch=completed_epoch,
        global_step=training_result.next_step,
        training_samples=metrics.samples,
        training_batches=metrics.batches,
        training_loss=metrics.loss,
        training_reconstruction=metrics.reconstruction,
        training_inside=metrics.inside,
        training_outside=metrics.outside,
        training_kl=metrics.kl,
        training_normalized_kl=metrics.normalized_kl,
    )


def advance_fit_state(
    *,
    previous_state: BRLoRAFitState,
    record: BRLoRAEpochRecord,
) -> tuple[BRLoRAFitState, bool]:
    """Advance internal-validation fit state."""

    _validate_state_record_sequence(
        previous_state=previous_state,
        record=record,
    )

    if not record.has_validation:
        raise BRLoRAFitError(
            "`advance_fit_state` requires validation metrics."
        )

    improved = (
        previous_state.best_validation_loss is None
        or record.validation_loss < previous_state.best_validation_loss
    )

    if improved:
        best_validation_loss = record.validation_loss
        best_completed_epoch = record.completed_epoch
    else:
        best_validation_loss = previous_state.best_validation_loss
        best_completed_epoch = previous_state.best_completed_epoch

    return (
        BRLoRAFitState(
            completed_epochs=record.completed_epoch,
            global_step=record.global_step,
            best_validation_loss=best_validation_loss,
            best_completed_epoch=best_completed_epoch,
        ),
        improved,
    )


def advance_full_train_state(
    *,
    previous_state: BRLoRAFitState,
    record: BRLoRAEpochRecord,
) -> BRLoRAFitState:
    """Advance fixed-epoch full-training state."""

    _validate_state_record_sequence(
        previous_state=previous_state,
        record=record,
    )

    if record.has_validation:
        raise BRLoRAFitError(
            "Full-training state requires a training-only record."
        )

    if (
        previous_state.best_validation_loss is not None
        or previous_state.best_completed_epoch is not None
    ):
        raise BRLoRAFitError(
            "Full-training state must not contain validation-best metadata."
        )

    return BRLoRAFitState(
        completed_epochs=record.completed_epoch,
        global_step=record.global_step,
        best_validation_loss=None,
        best_completed_epoch=None,
    )


def fit_br_lora(
    *,
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader | None,
    optimizer: torch.optim.Optimizer,
    schedule: DiffusionSchedule,
    device: torch.device,
    config: BRLoRAFitConfig,
    output_dir: str | Path,
    br_lora_config: dict[str, Any],
    model_config: dict[str, Any],
    data_config: dict[str, Any],
    initial_state: BRLoRAFitState | None = None,
    initial_history: tuple[BRLoRAEpochRecord, ...] = (),
    epoch_callback: Callable[
        [BRLoRAEpochRecord, BRLoRAFitState, bool | None],
        None,
    ] | None = None,
) -> BRLoRAFitResult:
    """
    Run BR-LoRA using ``data_config["split_mode"]`` as protocol source.
    """

    split_mode = _resolve_split_mode(data_config)

    _validate_fit_inputs(
        model=model,
        train_loader=train_loader,
        validation_loader=validation_loader,
        optimizer=optimizer,
        schedule=schedule,
        device=device,
        config=config,
        output_dir=output_dir,
        br_lora_config=br_lora_config,
        model_config=model_config,
        data_config=data_config,
        split_mode=split_mode,
        initial_state=initial_state,
        initial_history=initial_history,
        epoch_callback=epoch_callback,
    )

    state = BRLoRAFitState() if initial_state is None else initial_state
    history = list(initial_history)

    output_path = Path(output_dir)
    latest_checkpoint_path = output_path / "latest.pt"
    best_checkpoint_path = output_path / "best.pt"
    final_checkpoint_path = output_path / "final.pt"

    while state.completed_epochs < config.epochs:
        completed_epoch = state.completed_epochs + 1

        training_result = train_br_lora_epoch(
            model=model,
            batches=train_loader,
            optimizer=optimizer,
            schedule=schedule,
            device=device,
            start_step=state.global_step,
            kl_weight=config.kl_weight,
            warmup_steps=config.kl_warmup_steps,
            outside_weight=config.outside_loss_weight,
            max_grad_norm=config.max_grad_norm,
            sample_posterior=config.training_sample_posterior,
        )

        if split_mode == INTERNAL_SPLIT_MODE:
            assert validation_loader is not None

            validation_result = validate_br_lora_epoch(
                model=model,
                batches=validation_loader,
                schedule=schedule,
                device=device,
                step=training_result.next_step,
                kl_weight=config.kl_weight,
                warmup_steps=config.kl_warmup_steps,
                outside_weight=config.outside_loss_weight,
                sample_posterior=config.validation_sample_posterior,
            )

            record = build_epoch_record(
                completed_epoch=completed_epoch,
                training_result=training_result,
                validation_result=validation_result,
            )

            next_state, improved = advance_fit_state(
                previous_state=state,
                record=record,
            )

        else:
            record = build_full_train_epoch_record(
                completed_epoch=completed_epoch,
                training_result=training_result,
            )

            next_state = advance_full_train_state(
                previous_state=state,
                record=record,
            )

            improved = None

        history.append(record)
        history_tuple = tuple(history)

        _validate_history_sequence(history_tuple)
        _validate_history_mode_consistency(
            history_tuple,
            expected_split_mode=split_mode,
        )

        if len(history_tuple) != next_state.completed_epochs:
            raise BRLoRAFitError(
                "History length and completed epoch count diverged."
            )

        checkpoint_payload = build_br_lora_checkpoint_payload(
            model=model,
            optimizer=optimizer,
            completed_epochs=next_state.completed_epochs,
            global_step=next_state.global_step,
            best_validation_loss=next_state.best_validation_loss,
            br_lora_config=br_lora_config,
            training_config=config.to_dict(),
            model_config=model_config,
            data_config=data_config,
            history=history_to_dicts(history_tuple),
            include_rng_state=True,
        )

        save_br_lora_checkpoint(
            latest_checkpoint_path,
            checkpoint_payload,
        )

        if split_mode == INTERNAL_SPLIT_MODE and improved:
            save_br_lora_checkpoint(
                best_checkpoint_path,
                checkpoint_payload,
            )

        state = next_state

        if epoch_callback is not None:
            epoch_callback(record, state, improved)

    if not latest_checkpoint_path.is_file():
        raise BRLoRAFitError(
            "BR-LoRA fit completed without a latest checkpoint."
        )

    if split_mode == INTERNAL_SPLIT_MODE:
        if best_checkpoint_path.is_file():
            best_payload = load_br_lora_checkpoint(
                best_checkpoint_path,
                map_location=device,
            )
            best_payload["training_config"] = config.to_dict()
            save_br_lora_checkpoint(
                best_checkpoint_path,
                best_payload,
            )

        if (
            state.best_validation_loss is not None
            and not best_checkpoint_path.is_file()
        ):
            raise BRLoRAFitError(
                "Internal BR-LoRA fit has a best validation state but "
                "no best checkpoint."
            )

        return BRLoRAFitResult(
            state=state,
            history=tuple(history),
            split_mode=split_mode,
            latest_checkpoint_path=str(latest_checkpoint_path),
            best_checkpoint_path=str(best_checkpoint_path),
            final_checkpoint_path=None,
        )

    if state.completed_epochs != config.epochs:
        raise BRLoRAFitError(
            "Full-training BR-LoRA fit did not reach its configured epoch target."
        )

    final_payload = load_br_lora_checkpoint(
        latest_checkpoint_path,
        map_location=device,
    )
    save_br_lora_checkpoint(
        final_checkpoint_path,
        final_payload,
    )

    if not final_checkpoint_path.is_file():
        raise BRLoRAFitError(
            "Full-training BR-LoRA fit completed without a final checkpoint."
        )

    return BRLoRAFitResult(
        state=state,
        history=tuple(history),
        split_mode=split_mode,
        latest_checkpoint_path=str(latest_checkpoint_path),
        best_checkpoint_path=None,
        final_checkpoint_path=str(final_checkpoint_path),
    )


def _validate_fit_inputs(
    *,
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader | None,
    optimizer: torch.optim.Optimizer,
    schedule: DiffusionSchedule,
    device: torch.device,
    config: BRLoRAFitConfig,
    output_dir: str | Path,
    br_lora_config: dict[str, Any],
    model_config: dict[str, Any],
    data_config: dict[str, Any],
    split_mode: str,
    initial_state: BRLoRAFitState | None,
    initial_history: tuple[BRLoRAEpochRecord, ...],
    epoch_callback: object,
) -> None:
    if not isinstance(model, nn.Module):
        raise TypeError("`model` must be a torch.nn.Module.")

    if not isinstance(train_loader, DataLoader):
        raise TypeError("`train_loader` must be a DataLoader.")

    _validate_split_mode(split_mode)

    if split_mode == INTERNAL_SPLIT_MODE:
        if not isinstance(validation_loader, DataLoader):
            raise TypeError(
                "Internal BR-LoRA training requires "
                "`validation_loader` to be a DataLoader."
            )
    elif validation_loader is not None:
        raise BRLoRAFitError(
            "Full-training BR-LoRA requires `validation_loader=None`; "
            "no internal validation criterion may be fabricated."
        )

    if not isinstance(optimizer, torch.optim.Optimizer):
        raise TypeError("`optimizer` must be a torch.optim.Optimizer.")

    if not isinstance(schedule, DiffusionSchedule):
        raise TypeError("`schedule` must be a DiffusionSchedule.")

    if not isinstance(device, torch.device):
        raise TypeError("`device` must be a torch.device.")

    if not isinstance(config, BRLoRAFitConfig):
        raise TypeError("`config` must be a BRLoRAFitConfig.")

    if not isinstance(output_dir, (str, Path)):
        raise TypeError("`output_dir` must be a string or Path.")

    for name, value in (
        ("br_lora_config", br_lora_config),
        ("model_config", model_config),
        ("data_config", data_config),
    ):
        if not isinstance(value, dict):
            raise TypeError(f"`{name}` must be a dictionary.")

    if initial_state is not None and not isinstance(
        initial_state,
        BRLoRAFitState,
    ):
        raise TypeError(
            "`initial_state` must be a BRLoRAFitState or None."
        )

    if not isinstance(initial_history, tuple):
        raise TypeError("`initial_history` must be a tuple.")

    if not all(
        isinstance(record, BRLoRAEpochRecord)
        for record in initial_history
    ):
        raise TypeError(
            "Every initial history entry must be a BRLoRAEpochRecord."
        )

    _validate_history_sequence(initial_history)
    _validate_history_mode_consistency(
        initial_history,
        expected_split_mode=split_mode,
    )

    state = BRLoRAFitState() if initial_state is None else initial_state

    if len(initial_history) != state.completed_epochs:
        raise BRLoRAFitError(
            "Initial history length must equal the initial completed "
            "epoch count."
        )

    if initial_history:
        final_record = initial_history[-1]

        if final_record.completed_epoch != state.completed_epochs:
            raise BRLoRAFitError(
                "Initial history and state disagree about the last "
                "completed epoch."
            )

        if final_record.global_step != state.global_step:
            raise BRLoRAFitError(
                "Initial history and state disagree about the global step."
            )

    if split_mode == FULL_TRAIN_SPLIT_MODE and (
        state.best_validation_loss is not None
        or state.best_completed_epoch is not None
    ):
        raise BRLoRAFitError(
            "Full-training resume state must not contain "
            "validation-best metadata."
        )

    if state.completed_epochs > config.epochs:
        raise BRLoRAFitError(
            "Initial completed epoch count exceeds configured epochs."
        )

    if epoch_callback is not None and not callable(epoch_callback):
        raise TypeError(
            "`epoch_callback` must be callable or None."
        )


def _validate_state_record_sequence(
    *,
    previous_state: BRLoRAFitState,
    record: BRLoRAEpochRecord,
) -> None:
    if not isinstance(previous_state, BRLoRAFitState):
        raise TypeError(
            "`previous_state` must be a BRLoRAFitState."
        )

    if not isinstance(record, BRLoRAEpochRecord):
        raise TypeError(
            "`record` must be a BRLoRAEpochRecord."
        )

    expected_epoch = previous_state.completed_epochs + 1

    if record.completed_epoch != expected_epoch:
        raise BRLoRAFitError(
            "Epoch record does not immediately follow the previous state."
        )

    if record.global_step <= previous_state.global_step:
        raise BRLoRAFitError(
            "Epoch record global step must exceed the previous global step."
        )


def _resolve_split_mode(
    data_config: dict[str, Any],
) -> str:
    if not isinstance(data_config, dict):
        raise TypeError("`data_config` must be a dictionary.")

    split_mode = str(
        data_config.get(
            "split_mode",
            INTERNAL_SPLIT_MODE,
        )
    )
    return _validate_split_mode(split_mode)


def _validate_split_mode(
    split_mode: str,
) -> str:
    if not isinstance(split_mode, str):
        raise TypeError("`split_mode` must be a string.")

    if split_mode not in SUPPORTED_SPLIT_MODES:
        raise BRLoRAFitError(
            "split_mode must be 'internal' or 'full_train'."
        )

    return split_mode


def _validate_history_sequence(
    history: tuple[BRLoRAEpochRecord, ...],
) -> None:
    previous_epoch = 0
    previous_step = -1

    for record in history:
        if record.completed_epoch != previous_epoch + 1:
            raise BRLoRAFitError(
                "History epochs must be contiguous starting at 1."
            )

        if record.global_step <= previous_step:
            raise BRLoRAFitError(
                "History global steps must increase strictly."
            )

        previous_epoch = record.completed_epoch
        previous_step = record.global_step


def _validate_history_mode_consistency(
    history: tuple[BRLoRAEpochRecord, ...],
    *,
    expected_split_mode: str | None = None,
) -> None:
    if not history:
        return

    first_has_validation = history[0].has_validation

    if any(
        record.has_validation != first_has_validation
        for record in history
    ):
        raise BRLoRAFitError(
            "BR-LoRA history cannot mix internal-validation and "
            "full-training records."
        )

    if expected_split_mode is not None:
        _validate_split_mode(expected_split_mode)

        expected_has_validation = (
            expected_split_mode == INTERNAL_SPLIT_MODE
        )

        if first_has_validation != expected_has_validation:
            raise BRLoRAFitError(
                "BR-LoRA history does not match the configured split mode."
            )


def _require_finite_float(
    value: float,
    *,
    name: str,
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
    ):
        raise TypeError(
            f"`{name}` must be a real scalar."
        )

    converted = float(value)

    if not math.isfinite(converted):
        raise BRLoRAFitError(
            f"`{name}` must be finite."
        )

    return converted


def _require_nonnegative_finite_float(
    value: float,
    *,
    name: str,
) -> float:
    converted = _require_finite_float(value, name=name)

    if converted < 0.0:
        raise BRLoRAFitError(
            f"`{name}` must be non-negative."
        )

    return converted


def _require_positive_finite_float(
    value: float,
    *,
    name: str,
) -> float:
    converted = _require_finite_float(value, name=name)

    if converted <= 0.0:
        raise BRLoRAFitError(
            f"`{name}` must be positive."
        )

    return converted


def _require_nonnegative_integer(
    value: int,
    *,
    name: str,
) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
    ):
        raise TypeError(
            f"`{name}` must be an integer."
        )

    if value < 0:
        raise BRLoRAFitError(
            f"`{name}` must be non-negative."
        )

    return value


def _require_positive_integer(
    value: int,
    *,
    name: str,
) -> int:
    validated = _require_nonnegative_integer(
        value,
        name=name,
    )

    if validated <= 0:
        raise BRLoRAFitError(
            f"`{name}` must be positive."
        )

    return validated


__all__ = [
    "BRLoRAEpochRecord",
    "BRLoRAFitConfig",
    "BRLoRAFitError",
    "BRLoRAFitResult",
    "BRLoRAFitState",
    "FULL_TRAIN_SPLIT_MODE",
    "INTERNAL_SPLIT_MODE",
    "advance_fit_state",
    "advance_full_train_state",
    "build_epoch_record",
    "build_full_train_epoch_record",
    "fit_br_lora",
    "history_from_dicts",
    "history_to_dicts",
]
