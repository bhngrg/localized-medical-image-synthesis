"""
Multi-epoch orchestration primitives for Bayesian Regional LoRA (BR-LoRA).

This module coordinates already validated BR-LoRA components without
reimplementing adapter mathematics, the variational objective, optimization
steps, or epoch-level training logic.

The first development increment defines:

    training configuration
    mutable training state
    epoch-history records
    fit-result structures
    epoch-result to history-record conversion
    fit-state advancement
    history serialization helpers

Checkpoint writing, best-model selection, resume behavior, and the complete
multi-epoch fit loop are added in later increments.
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


class BRLoRAFitError(
    ValueError
):
    """Raised when BR-LoRA fit orchestration inputs are invalid."""


@dataclass(
    frozen=True,
    slots=True,
)
class BRLoRAFitConfig:
    """Static scientific settings for one BR-LoRA training run."""

    epochs: int

    kl_weight: float = 1.0e-6
    kl_warmup_steps: int = 1000

    outside_loss_weight: float = 0.05
    max_grad_norm: float | None = 1.0

    training_sample_posterior: bool = True
    validation_sample_posterior: bool = False

    def __post_init__(
        self,
    ) -> None:
        _require_positive_integer(
            self.epochs,
            name="epochs",
        )

        _require_nonnegative_finite_float(
            self.kl_weight,
            name="kl_weight",
        )

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

        if not isinstance(
            self.training_sample_posterior,
            bool,
        ):
            raise TypeError(
                "`training_sample_posterior` must be a bool."
            )

        if not isinstance(
            self.validation_sample_posterior,
            bool,
        ):
            raise TypeError(
                "`validation_sample_posterior` must be a bool."
            )

    def to_dict(
        self,
    ) -> dict[str, Any]:
        """Return a plain dictionary suitable for checkpoint metadata."""

        return {
            "epochs": self.epochs,
            "kl_weight": self.kl_weight,
            "kl_warmup_steps": (
                self.kl_warmup_steps
            ),
            "outside_loss_weight": (
                self.outside_loss_weight
            ),
            "max_grad_norm": (
                self.max_grad_norm
            ),
            "training_sample_posterior": (
                self.training_sample_posterior
            ),
            "validation_sample_posterior": (
                self.validation_sample_posterior
            ),
        }


@dataclass(
    frozen=True,
    slots=True,
)
class BRLoRAFitState:
    """Progress state required to continue BR-LoRA training."""

    completed_epochs: int = 0
    global_step: int = 0

    best_validation_loss: (
        float
        | None
    ) = None

    best_completed_epoch: (
        int
        | None
    ) = None

    def __post_init__(
        self,
    ) -> None:
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

            if (
                self.best_completed_epoch
                > self.completed_epochs
            ):
                raise BRLoRAFitError(
                    "`best_completed_epoch` cannot exceed "
                    "`completed_epochs`."
                )

        if (
            self.best_validation_loss is None
            and self.best_completed_epoch is not None
        ):
            raise BRLoRAFitError(
                "`best_completed_epoch` requires "
                "`best_validation_loss`."
            )

        if (
            self.best_validation_loss is not None
            and self.best_completed_epoch is None
        ):
            raise BRLoRAFitError(
                "`best_validation_loss` requires "
                "`best_completed_epoch`."
            )


@dataclass(
    frozen=True,
    slots=True,
)
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

    validation_samples: int
    validation_batches: int

    validation_loss: float
    validation_reconstruction: float
    validation_inside: float
    validation_outside: float
    validation_kl: float
    validation_normalized_kl: float

    validation_sample_posterior: bool

    def __post_init__(
        self,
    ) -> None:
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

        _require_positive_integer(
            self.validation_samples,
            name="validation_samples",
        )

        _require_positive_integer(
            self.validation_batches,
            name="validation_batches",
        )

        for name, value in (
            (
                "training_loss",
                self.training_loss,
            ),
            (
                "training_reconstruction",
                self.training_reconstruction,
            ),
            (
                "training_inside",
                self.training_inside,
            ),
            (
                "training_outside",
                self.training_outside,
            ),
            (
                "training_kl",
                self.training_kl,
            ),
            (
                "training_normalized_kl",
                self.training_normalized_kl,
            ),
            (
                "validation_loss",
                self.validation_loss,
            ),
            (
                "validation_reconstruction",
                self.validation_reconstruction,
            ),
            (
                "validation_inside",
                self.validation_inside,
            ),
            (
                "validation_outside",
                self.validation_outside,
            ),
            (
                "validation_kl",
                self.validation_kl,
            ),
            (
                "validation_normalized_kl",
                self.validation_normalized_kl,
            ),
        ):
            _require_finite_float(
                value,
                name=name,
            )

        if not isinstance(
            self.validation_sample_posterior,
            bool,
        ):
            raise TypeError(
                "`validation_sample_posterior` must be a bool."
            )

    def to_dict(
        self,
    ) -> dict[str, Any]:
        """Return a JSON/checkpoint-friendly representation."""

        return {
            "completed_epoch": (
                self.completed_epoch
            ),
            "global_step": (
                self.global_step
            ),
            "training_samples": (
                self.training_samples
            ),
            "training_batches": (
                self.training_batches
            ),
            "training_loss": (
                self.training_loss
            ),
            "training_reconstruction": (
                self.training_reconstruction
            ),
            "training_inside": (
                self.training_inside
            ),
            "training_outside": (
                self.training_outside
            ),
            "training_kl": (
                self.training_kl
            ),
            "training_normalized_kl": (
                self.training_normalized_kl
            ),
            "validation_samples": (
                self.validation_samples
            ),
            "validation_batches": (
                self.validation_batches
            ),
            "validation_loss": (
                self.validation_loss
            ),
            "validation_reconstruction": (
                self.validation_reconstruction
            ),
            "validation_inside": (
                self.validation_inside
            ),
            "validation_outside": (
                self.validation_outside
            ),
            "validation_kl": (
                self.validation_kl
            ),
            "validation_normalized_kl": (
                self.validation_normalized_kl
            ),
            "validation_sample_posterior": (
                self.validation_sample_posterior
            ),
        }


@dataclass(
    frozen=True,
    slots=True,
)
class BRLoRAFitResult:
    """Final structured result returned by BR-LoRA orchestration."""

    state: BRLoRAFitState

    history: tuple[
        BRLoRAEpochRecord,
        ...,
    ]

    latest_checkpoint_path: str
    best_checkpoint_path: str

    def __post_init__(
        self,
    ) -> None:
        if not isinstance(
            self.state,
            BRLoRAFitState,
        ):
            raise TypeError(
                "`state` must be a BRLoRAFitState."
            )

        if not isinstance(
            self.history,
            tuple,
        ):
            raise TypeError(
                "`history` must be a tuple."
            )

        if not all(
            isinstance(
                record,
                BRLoRAEpochRecord,
            )
            for record in self.history
        ):
            raise TypeError(
                "Every history entry must be "
                "a BRLoRAEpochRecord."
            )

        if len(
            self.history
        ) != self.state.completed_epochs:
            raise BRLoRAFitError(
                "History length must equal completed epoch count."
            )

        if not isinstance(
            self.latest_checkpoint_path,
            str,
        ):
            raise TypeError(
                "`latest_checkpoint_path` must be a string."
            )

        if not isinstance(
            self.best_checkpoint_path,
            str,
        ):
            raise TypeError(
                "`best_checkpoint_path` must be a string."
            )


def history_to_dicts(
    history: tuple[
        BRLoRAEpochRecord,
        ...,
    ],
) -> list[
    dict[str, Any]
]:
    """Convert typed epoch history into checkpoint-friendly dictionaries."""

    if not isinstance(
        history,
        tuple,
    ):
        raise TypeError(
            "`history` must be a tuple."
        )

    records: list[
        dict[str, Any]
    ] = []

    for record in history:

        if not isinstance(
            record,
            BRLoRAEpochRecord,
        ):
            raise TypeError(
                "Every history entry must be "
                "a BRLoRAEpochRecord."
            )

        records.append(
            record.to_dict()
        )

    return records


def history_from_dicts(
    history: list[
        dict[str, Any]
    ],
) -> tuple[
    BRLoRAEpochRecord,
    ...,
]:
    """Restore typed BR-LoRA epoch history from checkpoint dictionaries."""

    if not isinstance(
        history,
        list,
    ):
        raise TypeError(
            "`history` must be a list."
        )

    records: list[
        BRLoRAEpochRecord
    ] = []

    for index, record in enumerate(
        history
    ):

        if not isinstance(
            record,
            dict,
        ):
            raise TypeError(
                "Every history entry must be a dictionary; "
                f"entry {index} is "
                f"{type(record).__name__}."
            )

        try:
            restored = BRLoRAEpochRecord(
                **record
            )

        except TypeError as error:
            raise BRLoRAFitError(
                "Invalid BR-LoRA history record at "
                f"index {index}: {error}"
            ) from error

        records.append(
            restored
        )

    _validate_history_sequence(
        tuple(
            records
        )
    )

    return tuple(
        records
    )


def build_epoch_record(
    *,
    completed_epoch: int,
    training_result: BRLoRATrainEpochResult,
    validation_result: BRLoRAValidationEpochResult,
) -> BRLoRAEpochRecord:
    """
    Build one fit-history record from completed BR-LoRA epoch results.

    Training advances the global optimization step. Validation must use that
    same step and must not advance it.
    """

    _require_positive_integer(
        completed_epoch,
        name="completed_epoch",
    )

    if not isinstance(
        training_result,
        BRLoRATrainEpochResult,
    ):
        raise TypeError(
            "`training_result` must be a BRLoRATrainEpochResult."
        )

    if not isinstance(
        validation_result,
        BRLoRAValidationEpochResult,
    ):
        raise TypeError(
            "`validation_result` must be a "
            "BRLoRAValidationEpochResult."
        )

    if (
        validation_result.step
        != training_result.next_step
    ):
        raise BRLoRAFitError(
            "Validation step must equal the training result's "
            "next global step.\n"
            f"Training next step: {training_result.next_step}\n"
            f"Validation step:    {validation_result.step}"
        )

    training_metrics = (
        training_result.metrics
    )

    validation_metrics = (
        validation_result.metrics
    )

    if (
        training_metrics.batches
        != training_result.batches_processed
    ):
        raise BRLoRAFitError(
            "Training metrics and training result disagree about "
            "the number of processed batches."
        )

    if (
        validation_metrics.batches
        != validation_result.batches_processed
    ):
        raise BRLoRAFitError(
            "Validation metrics and validation result disagree about "
            "the number of processed batches."
        )

    return BRLoRAEpochRecord(
        completed_epoch=completed_epoch,
        global_step=(
            training_result.next_step
        ),
        training_samples=(
            training_metrics.samples
        ),
        training_batches=(
            training_metrics.batches
        ),
        training_loss=(
            training_metrics.loss
        ),
        training_reconstruction=(
            training_metrics.reconstruction
        ),
        training_inside=(
            training_metrics.inside
        ),
        training_outside=(
            training_metrics.outside
        ),
        training_kl=(
            training_metrics.kl
        ),
        training_normalized_kl=(
            training_metrics.normalized_kl
        ),
        validation_samples=(
            validation_metrics.samples
        ),
        validation_batches=(
            validation_metrics.batches
        ),
        validation_loss=(
            validation_metrics.loss
        ),
        validation_reconstruction=(
            validation_metrics.reconstruction
        ),
        validation_inside=(
            validation_metrics.inside
        ),
        validation_outside=(
            validation_metrics.outside
        ),
        validation_kl=(
            validation_metrics.kl
        ),
        validation_normalized_kl=(
            validation_metrics.normalized_kl
        ),
        validation_sample_posterior=(
            validation_result.sample_posterior
        ),
    )


def advance_fit_state(
    *,
    previous_state: BRLoRAFitState,
    record: BRLoRAEpochRecord,
) -> tuple[
    BRLoRAFitState,
    bool,
]:
    """
    Advance BR-LoRA fit state after one completed train/validation epoch.

    Returns
    -------
    tuple
        ``(next_state, improved)``.

        ``improved`` is true when the current validation loss establishes
        a new best checkpoint.
    """

    if not isinstance(
        previous_state,
        BRLoRAFitState,
    ):
        raise TypeError(
            "`previous_state` must be a BRLoRAFitState."
        )

    if not isinstance(
        record,
        BRLoRAEpochRecord,
    ):
        raise TypeError(
            "`record` must be a BRLoRAEpochRecord."
        )

    expected_epoch = (
        previous_state.completed_epochs
        + 1
    )

    if (
        record.completed_epoch
        != expected_epoch
    ):
        raise BRLoRAFitError(
            "Epoch record does not immediately follow the previous state.\n"
            f"Expected completed epoch: {expected_epoch}\n"
            f"Observed completed epoch: {record.completed_epoch}"
        )

    if (
        record.global_step
        <= previous_state.global_step
    ):
        raise BRLoRAFitError(
            "Epoch record global step must exceed the previous "
            "global step.\n"
            f"Previous step: {previous_state.global_step}\n"
            f"Record step:   {record.global_step}"
        )

    improved = (
        previous_state.best_validation_loss
        is None
        or record.validation_loss
        < previous_state.best_validation_loss
    )

    if improved:

        best_validation_loss = (
            record.validation_loss
        )

        best_completed_epoch = (
            record.completed_epoch
        )

    else:

        best_validation_loss = (
            previous_state.best_validation_loss
        )

        best_completed_epoch = (
            previous_state.best_completed_epoch
        )

    next_state = BRLoRAFitState(
        completed_epochs=(
            record.completed_epoch
        ),
        global_step=(
            record.global_step
        ),
        best_validation_loss=(
            best_validation_loss
        ),
        best_completed_epoch=(
            best_completed_epoch
        ),
    )

    return (
        next_state,
        improved,
    )



def fit_br_lora(
    *,
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
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
    epoch_callback: (
        Callable[
            [
                BRLoRAEpochRecord,
                BRLoRAFitState,
                bool,
            ],
            None,
        ]
        | None
    ) = None,
) -> BRLoRAFitResult:
    """
    Run multi-epoch BR-LoRA training using already validated components.

    This function performs orchestration only. It delegates:

    - stochastic variational optimization to ``train_br_lora_epoch``;
    - validation to ``validate_br_lora_epoch``;
    - epoch record construction to ``build_epoch_record``;
    - best-state bookkeeping to ``advance_fit_state``;
    - serialization to the validated BR-LoRA checkpoint helpers.

    ``initial_state`` and ``initial_history`` support resumed execution after
    the caller has already restored model, optimizer, and RNG state from a
    checkpoint. This function does not independently load checkpoints.
    """

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
        initial_state=initial_state,
        initial_history=initial_history,
        epoch_callback=epoch_callback,
    )

    state = (
        BRLoRAFitState()
        if initial_state is None
        else initial_state
    )

    history = list(
        initial_history
    )

    output_path = Path(
        output_dir
    )

    latest_checkpoint_path = (
        output_path
        / "latest.pt"
    )

    best_checkpoint_path = (
        output_path
        / "best.pt"
    )

    while (
        state.completed_epochs
        < config.epochs
    ):
        completed_epoch = (
            state.completed_epochs
            + 1
        )

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
            sample_posterior=(
                config.training_sample_posterior
            ),
        )

        validation_result = validate_br_lora_epoch(
            model=model,
            batches=validation_loader,
            schedule=schedule,
            device=device,
            step=training_result.next_step,
            kl_weight=config.kl_weight,
            warmup_steps=config.kl_warmup_steps,
            outside_weight=config.outside_loss_weight,
            sample_posterior=(
                config.validation_sample_posterior
            ),
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

        history.append(
            record
        )

        history_tuple = tuple(
            history
        )

        _validate_history_sequence(
            history_tuple
        )

        if (
            len(history_tuple)
            != next_state.completed_epochs
        ):
            raise BRLoRAFitError(
                "History length and completed epoch count diverged."
            )

        checkpoint_payload = (
            build_br_lora_checkpoint_payload(
                model=model,
                optimizer=optimizer,
                completed_epochs=(
                    next_state.completed_epochs
                ),
                global_step=(
                    next_state.global_step
                ),
                best_validation_loss=(
                    next_state.best_validation_loss
                ),
                br_lora_config=(
                    br_lora_config
                ),
                training_config=(
                    config.to_dict()
                ),
                model_config=(
                    model_config
                ),
                data_config=(
                    data_config
                ),
                history=history_to_dicts(
                    history_tuple
                ),
                include_rng_state=True,
            )
        )

        save_br_lora_checkpoint(
            latest_checkpoint_path,
            checkpoint_payload,
        )

        if improved:
            save_br_lora_checkpoint(
                best_checkpoint_path,
                checkpoint_payload,
            )

        state = next_state

        if epoch_callback is not None:
            epoch_callback(
                record,
                state,
                improved,
            )

    #
    # Refresh run-level metadata in the retained best checkpoint.
    #
    # A resumed run may increase the total epoch target without producing
    # a new best validation loss. In that case best.pt still contains the
    # correct best-epoch scientific state, but its training_config["epochs"]
    # reflects the earlier invocation.
    #
    # Refresh only the run-level training configuration while preserving
    # the stored best model, optimizer, RNG state, history, epoch,
    # global step, and validation loss exactly.
    #

    if best_checkpoint_path.is_file():

        best_payload = (
            load_br_lora_checkpoint(
                best_checkpoint_path,
                map_location=device,
            )
        )

        best_payload[
            "training_config"
        ] = config.to_dict()

        save_br_lora_checkpoint(
            best_checkpoint_path,
            best_payload,
        )

    if not latest_checkpoint_path.is_file():
        raise BRLoRAFitError(
            "BR-LoRA fit completed without a latest checkpoint."
        )

    if (
        state.best_validation_loss is not None
        and not best_checkpoint_path.is_file()
    ):
        raise BRLoRAFitError(
            "BR-LoRA fit has a best validation state but no best checkpoint."
        )

    return BRLoRAFitResult(
        state=state,
        history=tuple(
            history
        ),
        latest_checkpoint_path=str(
            latest_checkpoint_path
        ),
        best_checkpoint_path=str(
            best_checkpoint_path
        ),
    )


def _validate_fit_inputs(
    *,
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    schedule: DiffusionSchedule,
    device: torch.device,
    config: BRLoRAFitConfig,
    output_dir: str | Path,
    br_lora_config: dict[str, Any],
    model_config: dict[str, Any],
    data_config: dict[str, Any],
    initial_state: BRLoRAFitState | None,
    initial_history: tuple[BRLoRAEpochRecord, ...],
    epoch_callback: object,
) -> None:
    """Validate static orchestration inputs before any training occurs."""

    if not isinstance(
        model,
        nn.Module,
    ):
        raise TypeError(
            "`model` must be a torch.nn.Module."
        )

    if not isinstance(
        train_loader,
        DataLoader,
    ):
        raise TypeError(
            "`train_loader` must be a DataLoader."
        )

    if not isinstance(
        validation_loader,
        DataLoader,
    ):
        raise TypeError(
            "`validation_loader` must be a DataLoader."
        )

    if not isinstance(
        optimizer,
        torch.optim.Optimizer,
    ):
        raise TypeError(
            "`optimizer` must be a torch.optim.Optimizer."
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

    if not isinstance(
        config,
        BRLoRAFitConfig,
    ):
        raise TypeError(
            "`config` must be a BRLoRAFitConfig."
        )

    if not isinstance(
        output_dir,
        (
            str,
            Path,
        ),
    ):
        raise TypeError(
            "`output_dir` must be a string or Path."
        )

    for name, value in (
        (
            "br_lora_config",
            br_lora_config,
        ),
        (
            "model_config",
            model_config,
        ),
        (
            "data_config",
            data_config,
        ),
    ):
        if not isinstance(
            value,
            dict,
        ):
            raise TypeError(
                f"`{name}` must be a dictionary."
            )

    if (
        initial_state is not None
        and not isinstance(
            initial_state,
            BRLoRAFitState,
        )
    ):
        raise TypeError(
            "`initial_state` must be a BRLoRAFitState or None."
        )

    if not isinstance(
        initial_history,
        tuple,
    ):
        raise TypeError(
            "`initial_history` must be a tuple."
        )

    if not all(
        isinstance(
            record,
            BRLoRAEpochRecord,
        )
        for record in initial_history
    ):
        raise TypeError(
            "Every initial history entry must be a BRLoRAEpochRecord."
        )

    _validate_history_sequence(
        initial_history
    )

    state = (
        BRLoRAFitState()
        if initial_state is None
        else initial_state
    )

    if (
        len(initial_history)
        != state.completed_epochs
    ):
        raise BRLoRAFitError(
            "Initial history length must equal the initial completed "
            "epoch count."
        )

    if initial_history:
        final_record = (
            initial_history[
                -1
            ]
        )

        if (
            final_record.completed_epoch
            != state.completed_epochs
        ):
            raise BRLoRAFitError(
                "Initial history and state disagree about the last "
                "completed epoch."
            )

        if (
            final_record.global_step
            != state.global_step
        ):
            raise BRLoRAFitError(
                "Initial history and state disagree about the global step."
            )

    if (
        state.completed_epochs
        > config.epochs
    ):
        raise BRLoRAFitError(
            "Initial completed epoch count exceeds configured epochs."
        )

    if (
        epoch_callback is not None
        and not callable(
            epoch_callback
        )
    ):
        raise TypeError(
            "`epoch_callback` must be callable or None."
        )

def _validate_history_sequence(
    history: tuple[
        BRLoRAEpochRecord,
        ...,
    ],
) -> None:
    """Require monotonically increasing epoch and global-step history."""

    previous_epoch = 0
    previous_step = -1

    for record in history:

        if (
            record.completed_epoch
            != previous_epoch + 1
        ):
            raise BRLoRAFitError(
                "History epochs must be contiguous starting at 1."
            )

        if (
            record.global_step
            <= previous_step
        ):
            raise BRLoRAFitError(
                "History global steps must increase strictly."
            )

        previous_epoch = (
            record.completed_epoch
        )

        previous_step = (
            record.global_step
        )


def _require_finite_float(
    value: float,
    *,
    name: str,
) -> float:
    """Validate a finite real scalar."""

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
        raise BRLoRAFitError(
            f"`{name}` must be finite."
        )

    return converted


def _require_nonnegative_finite_float(
    value: float,
    *,
    name: str,
) -> float:
    """Validate a finite non-negative real scalar."""

    converted = _require_finite_float(
        value,
        name=name,
    )

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
    """Validate a finite positive real scalar."""

    converted = _require_finite_float(
        value,
        name=name,
    )

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
        raise BRLoRAFitError(
            f"`{name}` must be non-negative."
        )

    return value


def _require_positive_integer(
    value: int,
    *,
    name: str,
) -> int:
    """Validate a positive integer."""

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
    "advance_fit_state",
    "build_epoch_record",
    "fit_br_lora",
    "history_from_dicts",
    "history_to_dicts",
]
