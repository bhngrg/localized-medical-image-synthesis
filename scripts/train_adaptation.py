#!/usr/bin/env python3

"""Train deterministic adaptation comparators on a validated baseline.

The initial supported method is Regional LoRA. The comparator uses the same
trained AppearanceX0UNet baseline, eligible BraTS slices, diffusion schedule,
and masked x0 reconstruction objective as the validated baseline workflow.

Production comparator training is intentionally restricted to ``full_train``.
Additional adaptation methods can be added incrementally without changing the
shared deterministic trainer.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import random
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )

# Keep NumPy before PyTorch for the current macOS development environment.
import numpy as np
import torch
import yaml

from src.data import BraTSH5PatchX0Dataset
from src.data.loaders import create_full_train_loader
from src.diffusion import DiffusionSchedule
from src.models.adapters import (
    configure_regional_lora,
    deterministic_lora_parameter_count,
    iter_lora_modules,
    make_adaptation_report,
)
from src.training.adaptation_setup import load_baseline_backbone
from src.training import (
    capture_rng_state,
    restore_rng_state,
)
from src.training.trainer import (
    build_checkpoint_payload,
    save_checkpoint,
    train_one_epoch,
)


DEFAULT_BASELINE_CONFIG = Path(
    "configs/baseline_patch_x0_full_train.yaml"
)

DEFAULT_ADAPTATION_CONFIG = Path(
    "configs/adaptation.yaml"
)

SUPPORTED_METHODS = (
    "regional_lora",
)

FULL_TRAIN_SPLIT_MODE = "full_train"

EXPECTED_REGIONAL_LORA_PARAMETER_COUNT = 18_052
EXPECTED_REGIONAL_LORA_TRAINABLE_TENSOR_COUNT = 14


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Train deterministic adaptation comparators on top of a trained "
            "patch-conditioned x0 diffusion backbone."
        )
    )

    parser.add_argument(
        "--method",
        choices=SUPPORTED_METHODS,
        default="regional_lora",
        help="Deterministic adaptation method to train.",
    )

    parser.add_argument(
        "--baseline-config",
        type=Path,
        default=DEFAULT_BASELINE_CONFIG,
        help=(
            "Baseline configuration supplying data, model, diffusion, "
            "and split settings."
        ),
    )

    parser.add_argument(
        "--adaptation-config",
        type=Path,
        default=DEFAULT_ADAPTATION_CONFIG,
        help="Deterministic adaptation and training configuration.",
    )

    parser.add_argument(
        "--base-checkpoint",
        type=Path,
        required=True,
        help="Trained baseline patch-conditioned x0 checkpoint.",
    )

    parser.add_argument(
        "--h5-root",
        type=Path,
        required=True,
        help="Root directory containing reconstructed BraTS H5 slices.",
    )

    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Manifest CSV corresponding to the H5 dataset.",
    )

    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help=(
            "Comparator checkpoint output directory. When omitted, defaults "
            "to checkpoints/peft/<method>/full_train."
        ),
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help=(
            "Optional override for training.epochs in the adaptation "
            "configuration. Intended for explicit audits or controlled runs."
        ),
    )

    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help=(
            "Optional audit-only cap on the number of training samples. "
            "When omitted, the complete eligible training set is used."
        ),
    )

    parser.add_argument(
        "--resume-checkpoint",
        type=Path,
        default=None,
        help=(
            "Optional latest.pt checkpoint from which model, optimizer, "
            "completed-epoch history, and RNG state are restored."
        ),
    )

    parser.add_argument(
        "--device",
        choices=[
            "auto",
            "cpu",
            "mps",
            "cuda",
        ],
        default="auto",
    )

    return parser.parse_args()


def load_yaml_mapping(
    path: Path,
    *,
    name: str,
) -> dict:
    """Load one YAML configuration mapping."""

    resolved = path.expanduser().resolve()

    if not resolved.is_file():
        raise FileNotFoundError(
            f"{name} not found:\n{resolved}"
        )

    with resolved.open(
        "r",
        encoding="utf-8",
    ) as file:
        payload = yaml.safe_load(
            file
        )

    if not isinstance(
        payload,
        dict,
    ):
        raise ValueError(
            f"{name} must contain a YAML mapping."
        )

    return payload


def resolve_existing_file(
    path: Path,
    *,
    name: str,
) -> Path:
    """Resolve and validate one existing file."""

    resolved = path.expanduser().resolve()

    if not resolved.exists():
        raise FileNotFoundError(
            f"{name} does not exist:\n{resolved}"
        )

    if not resolved.is_file():
        raise ValueError(
            f"{name} must be a file:\n{resolved}"
        )

    return resolved


def resolve_existing_directory(
    path: Path,
    *,
    name: str,
) -> Path:
    """Resolve and validate one existing directory."""

    resolved = path.expanduser().resolve()

    if not resolved.exists():
        raise FileNotFoundError(
            f"{name} does not exist:\n{resolved}"
        )

    if not resolved.is_dir():
        raise ValueError(
            f"{name} must be a directory:\n{resolved}"
        )

    return resolved


def resolve_device(
    requested: str,
) -> torch.device:
    """Resolve an explicit or automatic Torch device."""

    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but is not available."
            )

        return torch.device(
            "cuda"
        )

    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError(
                "MPS was requested but is not available."
            )

        return torch.device(
            "mps"
        )

    if requested == "cpu":
        return torch.device(
            "cpu"
        )

    if torch.cuda.is_available():
        return torch.device(
            "cuda"
        )

    if torch.backends.mps.is_available():
        return torch.device(
            "mps"
        )

    return torch.device(
        "cpu"
    )


def set_seed(
    seed: int,
) -> None:
    """Seed Python, NumPy, and Torch."""

    random.seed(
        seed
    )

    np.random.seed(
        seed
    )

    torch.manual_seed(
        seed
    )

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            seed
        )


def require_mapping_section(
    config: dict,
    key: str,
    *,
    config_name: str,
) -> dict:
    """Return one required mapping-valued configuration section."""

    if key not in config:
        raise KeyError(
            f"{config_name} is missing required section {key!r}."
        )

    value = config[
        key
    ]

    if not isinstance(
        value,
        dict,
    ):
        raise ValueError(
            f"{config_name}.{key} must be a mapping."
        )

    return value


def make_optional_capped_loader(
    *,
    dataset,
    max_samples: int | None,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
):
    """Optionally cap full-training data for an explicit audit run."""

    if max_samples is None:
        return (
            create_full_train_loader(
                dataset=dataset,
                batch_size=batch_size,
                num_workers=num_workers,
                pin_memory=pin_memory,
            ),
            dataset,
        )

    if (
        isinstance(
            max_samples,
            bool,
        )
        or not isinstance(
            max_samples,
            int,
        )
        or max_samples <= 0
    ):
        raise ValueError(
            "--max-train-samples must be a positive integer."
        )

    if max_samples > len(
        dataset
    ):
        raise ValueError(
            "--max-train-samples cannot exceed the eligible dataset size."
        )

    subset = torch.utils.data.Subset(
        dataset,
        range(
            max_samples
        ),
    )

    loader = create_full_train_loader(
        dataset=subset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return (
        loader,
        subset,
    )


def configure_method(
    *,
    model: torch.nn.Module,
    method: str,
    adaptation_config: dict,
) -> tuple[
    tuple[str, ...],
    dict,
]:
    """Configure one supported deterministic adaptation method."""

    if method != "regional_lora":
        raise ValueError(
            f"Unsupported adaptation method: {method!r}"
        )

    method_cfg = require_mapping_section(
        adaptation_config,
        "regional_lora",
        config_name="adaptation config",
    )

    target_layers_object = method_cfg.get(
        "target_layers"
    )

    if not isinstance(
        target_layers_object,
        list,
    ):
        raise ValueError(
            "adaptation config regional_lora.target_layers "
            "must be a YAML list."
        )

    target_layers = tuple(
        str(
            name
        )
        for name in target_layers_object
    )

    if not target_layers:
        raise ValueError(
            "regional_lora.target_layers must contain at least one layer."
        )

    rank = int(
        method_cfg.get(
            "rank",
            4,
        )
    )

    alpha = float(
        method_cfg.get(
            "alpha",
            8.0,
        )
    )

    dropout = float(
        method_cfg.get(
            "dropout",
            0.0,
        )
    )

    injected = configure_regional_lora(
        model,
        target_layers=target_layers,
        rank=rank,
        alpha=alpha,
        dropout=dropout,
    )

    method_metadata = {
        "target_layers": target_layers,
        "rank": rank,
        "alpha": alpha,
        "dropout": dropout,
    }

    return (
        injected,
        method_metadata,
    )


def validate_configured_method(
    *,
    model: torch.nn.Module,
    method: str,
    injected: tuple[str, ...],
    trainable_parameter_count: int,
    trainable_tensor_count: int,
) -> None:
    """Validate method-specific invariants after deterministic adaptation."""

    if method == "regional_lora":
        lora_parameter_count = deterministic_lora_parameter_count(
            model
        )

        if (
            lora_parameter_count
            != trainable_parameter_count
        ):
            raise RuntimeError(
                "Regional LoRA trainable parameter count does not equal "
                "the deterministic LoRA parameter count."
            )

        if (
            trainable_parameter_count
            != EXPECTED_REGIONAL_LORA_PARAMETER_COUNT
        ):
            raise RuntimeError(
                "Regional LoRA parameter count changed unexpectedly: "
                f"{trainable_parameter_count} != "
                f"{EXPECTED_REGIONAL_LORA_PARAMETER_COUNT}."
            )

        if (
            trainable_tensor_count
            != EXPECTED_REGIONAL_LORA_TRAINABLE_TENSOR_COUNT
        ):
            raise RuntimeError(
                "Regional LoRA trainable tensor count changed unexpectedly: "
                f"{trainable_tensor_count} != "
                f"{EXPECTED_REGIONAL_LORA_TRAINABLE_TENSOR_COUNT}."
            )

        if tuple(
            name
            for name, _
            in iter_lora_modules(
                model
            )
        ) != injected:
            raise RuntimeError(
                "Regional LoRA module inventory changed after configuration."
            )

        return

    raise ValueError(
        f"Unsupported adaptation method: {method!r}"
    )


def load_adaptation_checkpoint(
    path: Path,
    *,
    device: torch.device,
) -> dict:
    """Load and minimally validate one deterministic adaptation checkpoint."""

    resolved = resolve_existing_file(
        path,
        name="Resume checkpoint",
    )

    payload = torch.load(
        resolved,
        map_location=device,
        weights_only=False,
    )

    if not isinstance(payload, dict):
        raise ValueError(
            "Resume checkpoint must contain a dictionary payload."
        )

    required = {
        "model_state_dict",
        "optimizer_state_dict",
        "completed_epochs",
        "train_losses",
        "rng_state",
        "adaptation_method",
        "adaptation_config",
        "training_contract",
    }

    missing = sorted(
        required - set(payload)
    )

    if missing:
        raise ValueError(
            "Resume checkpoint is missing required key(s): "
            + ", ".join(missing)
        )

    return payload


def validate_resume_contract(
    *,
    payload: dict,
    current_contract: dict,
    requested_epochs: int,
) -> None:
    """Require a resume checkpoint to match the current training contract."""

    saved_contract = payload[
        "training_contract"
    ]

    if not isinstance(saved_contract, dict):
        raise ValueError(
            "Resume checkpoint training_contract must be a dictionary."
        )

    if saved_contract != current_contract:
        differing = sorted(
            key
            for key in (
                set(saved_contract)
                | set(current_contract)
            )
            if saved_contract.get(key)
            != current_contract.get(key)
        )

        details = "\n".join(
            f"  {key}: saved={saved_contract.get(key)!r}, "
            f"current={current_contract.get(key)!r}"
            for key in differing
        )

        raise ValueError(
            "Resume checkpoint does not match the current deterministic "
            "adaptation training contract.\n"
            + details
        )

    completed_epochs = payload[
        "completed_epochs"
    ]

    if (
        isinstance(completed_epochs, bool)
        or not isinstance(completed_epochs, int)
        or completed_epochs < 0
    ):
        raise ValueError(
            "Resume checkpoint completed_epochs must be a non-negative integer."
        )

    train_losses = payload[
        "train_losses"
    ]

    if not isinstance(train_losses, list):
        raise ValueError(
            "Resume checkpoint train_losses must be a list."
        )

    if len(train_losses) != completed_epochs:
        raise ValueError(
            "Resume checkpoint train_losses length does not equal "
            "completed_epochs."
        )

    if requested_epochs < completed_epochs:
        raise ValueError(
            "Requested epoch target cannot be less than the number of "
            "epochs already completed in the resume checkpoint."
        )


def main() -> None:
    """Construct, audit, and execute deterministic adaptation training."""

    args = parse_args()

    baseline_config = load_yaml_mapping(
        args.baseline_config,
        name="Baseline configuration",
    )

    adaptation_config = load_yaml_mapping(
        args.adaptation_config,
        name="Adaptation configuration",
    )

    data_cfg = require_mapping_section(
        baseline_config,
        "data",
        config_name="baseline config",
    )

    model_cfg = require_mapping_section(
        baseline_config,
        "model",
        config_name="baseline config",
    )

    diffusion_cfg = require_mapping_section(
        baseline_config,
        "diffusion",
        config_name="baseline config",
    )

    training_cfg = require_mapping_section(
        adaptation_config,
        "training",
        config_name="adaptation config",
    )

    split_mode = str(
        data_cfg.get(
            "split_mode",
            ""
        )
    )

    if split_mode != FULL_TRAIN_SPLIT_MODE:
        raise ValueError(
            "Deterministic adaptation comparator training currently requires "
            "data.split_mode='full_train'. Use "
            "configs/baseline_patch_x0_full_train.yaml for the production "
            "comparison protocol."
        )

    seed = int(
        baseline_config.get(
            "seed",
            42,
        )
    )

    h5_root = resolve_existing_directory(
        args.h5_root,
        name="H5 root",
    )

    manifest = resolve_existing_file(
        args.manifest,
        name="Manifest",
    )

    base_checkpoint = resolve_existing_file(
        args.base_checkpoint,
        name="Base checkpoint",
    )

    checkpoint_dir = (
        Path(
            "checkpoints"
        )
        / "peft"
        / args.method
        / FULL_TRAIN_SPLIT_MODE
        if args.checkpoint_dir is None
        else args.checkpoint_dir
    )

    checkpoint_dir = (
        checkpoint_dir
        .expanduser()
        .resolve()
    )

    device = resolve_device(
        args.device
    )

    set_seed(
        seed
    )

    dataset = BraTSH5PatchX0Dataset(
        root=h5_root,
        manifest_path=manifest,
        image_channel=int(
            data_cfg.get(
                "image_channel",
                0,
            )
        ),
        min_tumor_pixels=int(
            data_cfg.get(
                "min_tumor_pixels",
                300,
            )
        ),
        use_whole_tumor=bool(
            data_cfg.get(
                "use_whole_tumor",
                True,
            )
        ),
    )

    batch_size = int(
        data_cfg.get(
            "batch_size",
            8,
        )
    )

    num_workers = int(
        data_cfg.get(
            "num_workers",
            0,
        )
    )

    pin_memory = bool(
        data_cfg.get(
            "pin_memory",
            False,
        )
    )

    train_loader, train_dataset = make_optional_capped_loader(
        dataset=dataset,
        max_samples=args.max_train_samples,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    schedule = DiffusionSchedule(
        timesteps=int(
            diffusion_cfg.get(
                "timesteps",
                200,
            )
        ),
        beta_start=float(
            diffusion_cfg.get(
                "beta_start",
                1.0e-4,
            )
        ),
        beta_end=float(
            diffusion_cfg.get(
                "beta_end",
                0.02,
            )
        ),
        device=device,
    )

    model, baseline_checkpoint = load_baseline_backbone(
        checkpoint_path=base_checkpoint,
        model_cfg=model_cfg,
        device=device,
    )

    injected, method_metadata = configure_method(
        model=model,
        method=args.method,
        adaptation_config=adaptation_config,
    )

    report = make_adaptation_report(
        model,
        method=args.method,
        adapted_module_names=injected,
    )

    trainable = tuple(
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    trainable_tensor_count = len(
        trainable
    )

    if not trainable:
        raise RuntimeError(
            "Adaptation configuration produced no trainable parameters."
        )

    validate_configured_method(
        model=model,
        method=args.method,
        injected=injected,
        trainable_parameter_count=report.trainable_parameters,
        trainable_tensor_count=trainable_tensor_count,
    )

    learning_rate = float(
        training_cfg.get(
            "learning_rate",
            1.0e-4,
        )
    )

    weight_decay = float(
        training_cfg.get(
            "weight_decay",
            0.0,
        )
    )

    configured_epochs = int(
        training_cfg.get(
            "epochs",
            50,
        )
    )

    epochs = (
        configured_epochs
        if args.epochs is None
        else int(
            args.epochs
        )
    )

    if epochs <= 0:
        raise ValueError(
            "Training epochs must be positive."
        )

    outside_loss_weight = float(
        training_cfg.get(
            "outside_loss_weight",
            0.05,
        )
    )

    optimizer = torch.optim.AdamW(
        trainable,
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    print()
    print(
        "=" * 78
    )
    print(
        "DETERMINISTIC ADAPTATION TRAINING STACK"
    )
    print(
        "=" * 78
    )

    print()
    print(
        "Method"
    )
    print(
        "-" * 78
    )
    print(
        "Adaptation method         :",
        args.method,
    )
    print(
        "Adapted modules           :",
        len(
            injected
        ),
    )
    print(
        "Trainable tensors         :",
        trainable_tensor_count,
    )
    print(
        "Trainable parameters      :",
        f"{report.trainable_parameters:,}",
    )
    print(
        "Total parameters          :",
        f"{report.total_parameters:,}",
    )
    print(
        "Trainable percent         :",
        f"{report.trainable_percent:.6f}%",
    )

    print()
    print(
        "Paths"
    )
    print(
        "-" * 78
    )
    print(
        "Baseline config           :",
        args.baseline_config.expanduser().resolve(),
    )
    print(
        "Adaptation config         :",
        args.adaptation_config.expanduser().resolve(),
    )
    print(
        "Base checkpoint           :",
        base_checkpoint,
    )
    print(
        "H5 root                   :",
        h5_root,
    )
    print(
        "Manifest                  :",
        manifest,
    )
    print(
        "Checkpoint dir            :",
        checkpoint_dir,
    )

    print()
    print(
        "Protocol"
    )
    print(
        "-" * 78
    )
    print(
        "Device                    :",
        device,
    )
    print(
        "Seed                      :",
        seed,
    )
    print(
        "Split mode                :",
        split_mode,
    )
    print(
        "Training samples          :",
        f"{len(train_dataset):,}",
    )
    print(
        "Batch size                :",
        batch_size,
    )
    print(
        "Epochs                    :",
        epochs,
    )
    print(
        "Learning rate             :",
        learning_rate,
    )
    print(
        "Weight decay              :",
        weight_decay,
    )
    print(
        "Outside loss weight       :",
        outside_loss_weight,
    )
    print(
        "Diffusion timesteps       :",
        schedule.timesteps,
    )

    checkpoint_metadata = {
        "timesteps": schedule.timesteps,
        "base_channels": int(
            model_cfg.get(
                "base_channels",
                32,
            )
        ),
        "image_channel": int(
            data_cfg.get(
                "image_channel",
                0,
            )
        ),
        "min_tumor_pixels": int(
            data_cfg.get(
                "min_tumor_pixels",
                300,
            )
        ),
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "cond_dim": int(
            model_cfg.get(
                "cond_dim",
                4,
            )
        ),
        "training_mode": (
            "patch_conditioned_x0_diffusion_adaptation"
        ),
        "split_mode": FULL_TRAIN_SPLIT_MODE,
    }

    training_contract = {
        "adaptation_method": args.method,
        "adaptation_config": method_metadata,
        "base_checkpoint": str(base_checkpoint),
        "baseline_checkpoint_training_mode": baseline_checkpoint.get(
            "training_mode"
        ),
        "baseline_checkpoint_split_mode": baseline_checkpoint.get(
            "split_mode"
        ),
        "h5_root": str(h5_root),
        "manifest": str(manifest),
        "max_train_samples": args.max_train_samples,
        "training_samples": len(train_dataset),
        "seed": seed,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "outside_loss_weight": outside_loss_weight,
        "timesteps": schedule.timesteps,
        "base_channels": checkpoint_metadata["base_channels"],
        "image_channel": checkpoint_metadata["image_channel"],
        "min_tumor_pixels": checkpoint_metadata["min_tumor_pixels"],
        "cond_dim": checkpoint_metadata["cond_dim"],
        "split_mode": FULL_TRAIN_SPLIT_MODE,
    }

    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    latest_checkpoint_path = (
        checkpoint_dir
        / "latest.pt"
    )

    final_checkpoint_path = (
        checkpoint_dir
        / f"final_{args.method}_full_train.pt"
    )

    completed_epochs = 0
    train_losses: list[float] = []

    if args.resume_checkpoint is not None:
        resume_checkpoint = (
            args.resume_checkpoint
            .expanduser()
            .resolve()
        )

        resume_payload = load_adaptation_checkpoint(
            resume_checkpoint,
            device=device,
        )

        validate_resume_contract(
            payload=resume_payload,
            current_contract=training_contract,
            requested_epochs=epochs,
        )

        model.load_state_dict(
            resume_payload[
                "model_state_dict"
            ],
            strict=True,
        )

        optimizer.load_state_dict(
            resume_payload[
                "optimizer_state_dict"
            ]
        )

        completed_epochs = int(
            resume_payload[
                "completed_epochs"
            ]
        )

        train_losses = [
            float(loss)
            for loss in resume_payload[
                "train_losses"
            ]
        ]

        restore_rng_state(
            resume_payload[
                "rng_state"
            ]
        )

        print()
        print(
            "=" * 78
        )
        print(
            "DETERMINISTIC ADAPTATION RESUME"
        )
        print(
            "=" * 78
        )
        print(
            "Checkpoint               :",
            resume_checkpoint,
        )
        print(
            "Completed epochs         :",
            completed_epochs,
        )
        print(
            "Target epochs            :",
            epochs,
        )
        print(
            "Loss history length      :",
            len(train_losses),
        )
        print(
            "RNG restored             : True"
        )
        print(
            "=" * 78
        )

    for completed_epoch in range(
        completed_epochs + 1,
        epochs + 1,
    ):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            schedule=schedule,
            device=device,
            outside_loss_weight=outside_loss_weight,
            description=(
                f"Epoch {completed_epoch}/{epochs}"
            ),
        )

        train_losses.append(
            train_loss
        )

        print(
            f"Epoch {completed_epoch}: "
            f"train_loss={train_loss:.4f}"
        )

        latest_payload = build_checkpoint_payload(
            model=model,
            optimizer=optimizer,
            epoch=completed_epoch,
            epochs=epochs,
            final_train_loss=train_loss,
            final_val_loss=None,
            best_val_loss=None,
            **checkpoint_metadata,
        )

        latest_payload[
            "completed_epochs"
        ] = completed_epoch

        latest_payload[
            "train_losses"
        ] = list(
            train_losses
        )

        latest_payload[
            "rng_state"
        ] = capture_rng_state()

        latest_payload[
            "adaptation_method"
        ] = args.method

        latest_payload[
            "adaptation_config"
        ] = method_metadata

        latest_payload[
            "trainable_parameter_count"
        ] = report.trainable_parameters

        latest_payload[
            "trainable_tensor_count"
        ] = trainable_tensor_count

        latest_payload[
            "base_checkpoint"
        ] = str(
            base_checkpoint
        )

        latest_payload[
            "baseline_checkpoint_training_mode"
        ] = baseline_checkpoint.get(
            "training_mode"
        )

        latest_payload[
            "baseline_checkpoint_split_mode"
        ] = baseline_checkpoint.get(
            "split_mode"
        )

        latest_payload[
            "training_contract"
        ] = training_contract

        save_checkpoint(
            latest_checkpoint_path,
            latest_payload,
        )

        print(
            f"Saved latest checkpoint to {latest_checkpoint_path}"
        )

    if not train_losses:
        raise RuntimeError(
            "Adaptation training completed without any training-loss history."
        )

    final_payload = build_checkpoint_payload(
        model=model,
        optimizer=optimizer,
        epoch=None,
        epochs=epochs,
        final_train_loss=train_losses[-1],
        final_val_loss=None,
        best_val_loss=None,
        **checkpoint_metadata,
    )

    final_payload[
        "completed_epochs"
    ] = epochs

    final_payload[
        "train_losses"
    ] = list(
        train_losses
    )

    final_payload[
        "rng_state"
    ] = capture_rng_state()

    final_payload[
        "adaptation_method"
    ] = args.method

    final_payload[
        "adaptation_config"
    ] = method_metadata

    final_payload[
        "trainable_parameter_count"
    ] = report.trainable_parameters

    final_payload[
        "trainable_tensor_count"
    ] = trainable_tensor_count

    final_payload[
        "base_checkpoint"
    ] = str(
        base_checkpoint
    )

    final_payload[
        "baseline_checkpoint_training_mode"
    ] = baseline_checkpoint.get(
        "training_mode"
    )

    final_payload[
        "baseline_checkpoint_split_mode"
    ] = baseline_checkpoint.get(
        "split_mode"
    )

    final_payload[
        "training_contract"
    ] = training_contract

    save_checkpoint(
        final_checkpoint_path,
        final_payload,
    )

    print()
    print(
        f"Saved final model to {final_checkpoint_path}"
    )


if __name__ == "__main__":
    try:
        main()

    except (
        RuntimeError,
        ValueError,
        KeyError,
        FileNotFoundError,
    ) as error:
        raise SystemExit(
            str(
                error
            )
        ) from error
