#!/usr/bin/env python3

"""
Construct and train Bayesian Regional LoRA (BR-LoRA) on a trained baseline.

The training protocol is inherited from the baseline configuration:

internal
    Reproduce the validated internal 90/10 workflow. Training uses the internal
    optimization split, validation is performed every epoch, ``latest.pt`` is
    written every epoch, and ``best.pt`` is selected by validation loss.

full_train
    Use 100% of eligible training slices for a fixed number of epochs. No
    internal validation loss or best-checkpoint rule is computed.
    ``latest.pt`` is written every epoch for exact resume and ``final.pt`` is
    written when the configured epoch target is reached.

Both protocols support exact continuation from a validated BR-LoRA checkpoint.
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

from torch.utils.data import (
    DataLoader,
    Subset,
)

from src.data import (
    BraTSH5PatchX0Dataset,
    create_train_val_loaders,
)
from src.data.loaders import (
    create_full_train_loader,
)
from src.diffusion import DiffusionSchedule
from src.models import AppearanceX0UNet
from src.models.adapters import (
    convert_lora_to_variational,
    freeze_module,
    inject_lora,
    iter_lora_modules,
    iter_variational_lora_modules,
    variational_lora_parameter_count,
)
from src.training.br_lora_fit import (
    BRLoRAFitConfig,
    BRLoRAFitState,
    FULL_TRAIN_SPLIT_MODE,
    INTERNAL_SPLIT_MODE,
    fit_br_lora,
    history_from_dicts,
)
from src.training.br_lora_checkpoint import (
    load_br_lora_checkpoint,
    restore_br_lora_checkpoint,
)


DEFAULT_BASELINE_CONFIG = Path(
    "configs/baseline_patch_x0.yaml"
)

DEFAULT_BR_LORA_CONFIG = Path(
    "configs/br_lora.yaml"
)

EXPECTED_VARIATIONAL_PARAMETER_COUNT = 36_104
EXPECTED_TRAINABLE_TENSOR_COUNT = 28


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Train BR-LoRA on top of a trained patch-conditioned "
            "x0 diffusion backbone."
        )
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
        "--br-lora-config",
        type=Path,
        default=DEFAULT_BR_LORA_CONFIG,
        help="BR-LoRA adapter and variational-training configuration.",
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
        default=Path(
            "checkpoints/br_lora"
        ),
        help="BR-LoRA checkpoint output directory.",
    )

    parser.add_argument(
        "--resume-checkpoint",
        type=Path,
        default=None,
        help=(
            "Optional BR-LoRA latest checkpoint from which model, optimizer, "
            "history, fit state, and RNG state are restored before training."
        ),
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help=(
            "Optional override for training.epochs in the BR-LoRA "
            "configuration. Intended for explicit audits or controlled runs."
        ),
    )

    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help=(
            "Optional audit-only cap on the number of training samples. "
            "When omitted, the complete configured training set is used."
        ),
    )

    parser.add_argument(
        "--max-validation-samples",
        type=int,
        default=None,
        help=(
            "Optional audit-only cap on validation samples. This is valid "
            "only for data.split_mode='internal'."
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


def build_backbone(
    *,
    model_cfg: dict,
    device: torch.device,
) -> AppearanceX0UNet:
    """Construct the validated local AppearanceX0UNet."""

    return AppearanceX0UNet(
        in_ch=int(
            model_cfg.get(
                "in_channels",
                4,
            )
        ),
        out_ch=int(
            model_cfg.get(
                "out_channels",
                1,
            )
        ),
        base=int(
            model_cfg.get(
                "base_channels",
                32,
            )
        ),
        time_dim=int(
            model_cfg.get(
                "time_dim",
                128,
            )
        ),
        cond_dim=int(
            model_cfg.get(
                "cond_dim",
                4,
            )
        ),
    ).to(
        device
    )


def load_baseline_backbone(
    *,
    checkpoint_path: Path,
    model_cfg: dict,
    device: torch.device,
) -> tuple[
    AppearanceX0UNet,
    dict,
]:
    """Strictly restore the trained baseline backbone checkpoint."""

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    if not isinstance(
        checkpoint,
        dict,
    ):
        raise RuntimeError(
            "Baseline checkpoint must contain a dictionary."
        )

    if "model_state_dict" not in checkpoint:
        raise RuntimeError(
            "Baseline checkpoint is missing 'model_state_dict'."
        )

    model = build_backbone(
        model_cfg=model_cfg,
        device=device,
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
        raise RuntimeError(
            "Strict baseline checkpoint restoration unexpectedly "
            "reported state-dict mismatches."
        )

    return (
        model,
        checkpoint,
    )


def configure_br_lora(
    *,
    model: AppearanceX0UNet,
    br_lora_cfg: dict,
    training_cfg: dict,
) -> tuple[
    tuple[str, ...],
    tuple[str, ...],
]:
    """Freeze the backbone and construct fresh BR-LoRA adapters."""

    target_layers_object = br_lora_cfg.get(
        "target_layers"
    )

    if not isinstance(
        target_layers_object,
        list,
    ):
        raise ValueError(
            "br_lora.target_layers must be a YAML list."
        )

    target_layers = tuple(
        str(
            name
        )
        for name in target_layers_object
    )

    if not target_layers:
        raise ValueError(
            "br_lora.target_layers must contain at least one layer."
        )

    freeze_module(
        model
    )

    injected = inject_lora(
        model,
        rank=int(
            br_lora_cfg.get(
                "rank",
                4,
            )
        ),
        alpha=float(
            br_lora_cfg.get(
                "alpha",
                8.0,
            )
        ),
        dropout=float(
            br_lora_cfg.get(
                "dropout",
                0.0,
            )
        ),
        exact_names=target_layers,
    )

    if injected != target_layers:
        raise RuntimeError(
            "Fresh LoRA injection inventory does not match "
            "br_lora.target_layers.\n"
            f"Configured: {target_layers}\n"
            f"Injected:   {injected}"
        )

    deterministic_names = tuple(
        name
        for name, _
        in iter_lora_modules(
            model
        )
    )

    if deterministic_names != target_layers:
        raise RuntimeError(
            "Deterministic LoRA inventory does not match the "
            "configured target-layer sequence."
        )

    for name, module in iter_lora_modules(
        model
    ):
        nonzero_b = int(
            torch.count_nonzero(
                module.lora_B.weight
            ).item()
        )

        if nonzero_b != 0:
            raise RuntimeError(
                "Fresh deterministic LoRA B factors must be exactly zero; "
                f"{name!r} contains {nonzero_b} nonzero elements."
            )

    converted = convert_lora_to_variational(
        model,
        initial_std=float(
            br_lora_cfg.get(
                "initial_std",
                0.01,
            )
        ),
        prior_mean=float(
            br_lora_cfg.get(
                "prior_mean",
                0.0,
            )
        ),
        prior_std=float(
            br_lora_cfg.get(
                "prior_std",
                1.0,
            )
        ),
        minimum_std=float(
            br_lora_cfg.get(
                "minimum_std",
                1.0e-8,
            )
        ),
        target_names=target_layers,
        sample_posterior=bool(
            training_cfg.get(
                "sample_posterior",
                True,
            )
        ),
    )

    if converted != target_layers:
        raise RuntimeError(
            "BR-LoRA conversion inventory does not match "
            "br_lora.target_layers."
        )

    if iter_lora_modules(
        model
    ):
        raise RuntimeError(
            "Deterministic LoRA modules remain after BR-LoRA conversion."
        )

    variational_names = tuple(
        name
        for name, _
        in iter_variational_lora_modules(
            model
        )
    )

    if variational_names != target_layers:
        raise RuntimeError(
            "BR-LoRA adapter inventory does not match the "
            "configured target-layer sequence."
        )

    return (
        injected,
        converted,
    )


def posterior_parameters(
    model: AppearanceX0UNet,
) -> tuple[
    torch.nn.Parameter,
    ...,
]:
    """Return exactly the trainable BR-LoRA posterior parameters."""

    return tuple(
        parameter
        for parameter
        in model.parameters()
        if parameter.requires_grad
    )


def make_optional_capped_loader(
    *,
    loader: DataLoader,
    dataset,
    max_samples: int | None,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    name: str,
) -> tuple[
    DataLoader,
    object,
]:
    """
    Optionally cap one dataset for an explicit development/audit run.

    Production behavior is unchanged when ``max_samples`` is ``None``.
    Capped loaders use deterministic ordering and are intended only for
    small end-to-end validation runs.
    """

    if max_samples is None:
        return (
            loader,
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
    ):
        raise TypeError(
            f"`{name}` must be an integer or None."
        )

    if max_samples <= 0:
        raise ValueError(
            f"`{name}` must be positive when provided."
        )

    capped_count = min(
        max_samples,
        len(
            dataset
        ),
    )

    capped_dataset = Subset(
        dataset,
        range(
            capped_count
        ),
    )

    capped_loader = DataLoader(
        capped_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return (
        capped_loader,
        capped_dataset,
    )


def infer_best_completed_epoch(
    *,
    history,
    best_validation_loss: float | None,
) -> int | None:
    """Recover the epoch that established the stored running best loss."""

    if best_validation_loss is None:
        return None

    for record in history:
        if (
            record.validation_loss
            == best_validation_loss
        ):
            return record.completed_epoch

    raise RuntimeError(
        "Checkpoint best_validation_loss does not match any "
        "stored history record."
    )


def validate_resume_metadata(
    *,
    payload: dict,
    current_br_lora_config: dict,
    current_fit_config: BRLoRAFitConfig,
    current_model_config: dict,
    current_data_config: dict,
) -> None:
    """
    Require resume metadata to match the current scientific configuration.

    ``epochs`` is intentionally allowed to increase during resume. All other
    fit settings must match exactly.
    """

    checkpoint_br_lora = dict(
        payload[
            "br_lora_config"
        ]
    )

    if checkpoint_br_lora != current_br_lora_config:
        raise RuntimeError(
            "Resume checkpoint BR-LoRA configuration does not match "
            "the current BR-LoRA configuration."
        )

    checkpoint_training = dict(
        payload[
            "training_config"
        ]
    )

    current_training = (
        current_fit_config.to_dict()
    )

    checkpoint_epochs = int(
        checkpoint_training.pop(
            "epochs"
        )
    )

    current_epochs = int(
        current_training.pop(
            "epochs"
        )
    )

    if checkpoint_training != current_training:
        raise RuntimeError(
            "Resume checkpoint training configuration does not match "
            "the current training configuration apart from total epochs."
        )

    completed_epochs = int(
        payload[
            "completed_epochs"
        ]
    )

    if current_epochs < completed_epochs:
        raise RuntimeError(
            "Requested total epochs cannot be smaller than the number "
            "already completed in the resume checkpoint."
        )

    if current_epochs < checkpoint_epochs:
        raise RuntimeError(
            "A resumed run cannot reduce the total epoch target stored "
            "in the checkpoint."
        )

    checkpoint_model_config = payload[
        "model_config"
    ]

    if (
        checkpoint_model_config is None
        or dict(
            checkpoint_model_config
        )
        != current_model_config
    ):
        raise RuntimeError(
            "Resume checkpoint model configuration does not match "
            "the current baseline model configuration."
        )

    checkpoint_data_config = payload[
        "data_config"
    ]

    if (
        checkpoint_data_config is None
        or dict(
            checkpoint_data_config
        )
        != current_data_config
    ):
        raise RuntimeError(
            "Resume checkpoint data configuration does not match "
            "the current effective data configuration."
        )


def main() -> None:
    """Construct, audit, and execute fresh or resumed BR-LoRA training."""

    args = parse_args()

    baseline_config = load_yaml_mapping(
        args.baseline_config,
        name="Baseline configuration",
    )

    br_lora_config_file = load_yaml_mapping(
        args.br_lora_config,
        name="BR-LoRA configuration",
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

    br_lora_cfg = require_mapping_section(
        br_lora_config_file,
        "br_lora",
        config_name="BR-LoRA config",
    )

    br_training_cfg = require_mapping_section(
        br_lora_config_file,
        "training",
        config_name="BR-LoRA config",
    )

    seed = int(
        baseline_config.get(
            "seed",
            42,
        )
    )

    split_mode = str(
        data_cfg.get(
            "split_mode",
            INTERNAL_SPLIT_MODE,
        )
    )

    if split_mode not in {
        INTERNAL_SPLIT_MODE,
        FULL_TRAIN_SPLIT_MODE,
    }:
        raise ValueError(
            "data.split_mode must be 'internal' or 'full_train'."
        )

    if (
        split_mode == FULL_TRAIN_SPLIT_MODE
        and args.max_validation_samples is not None
    ):
        raise ValueError(
            "--max-validation-samples is not valid for full_train mode "
            "because no internal validation dataset is created."
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

    resume_checkpoint = (
        None
        if args.resume_checkpoint is None
        else resolve_existing_file(
            args.resume_checkpoint,
            name="Resume checkpoint",
        )
    )

    checkpoint_dir = (
        args.checkpoint_dir
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

    if split_mode == INTERNAL_SPLIT_MODE:

        (
            train_loader,
            validation_loader,
            train_dataset,
            validation_dataset,
        ) = create_train_val_loaders(
            dataset=dataset,
            batch_size=batch_size,
            train_fraction=float(
                data_cfg.get(
                    "train_fraction",
                    0.9,
                )
            ),
            seed=seed,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        (
            train_loader,
            train_dataset,
        ) = make_optional_capped_loader(
            loader=train_loader,
            dataset=train_dataset,
            max_samples=args.max_train_samples,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            name="max_train_samples",
        )

        (
            validation_loader,
            validation_dataset,
        ) = make_optional_capped_loader(
            loader=validation_loader,
            dataset=validation_dataset,
            max_samples=args.max_validation_samples,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            name="max_validation_samples",
        )

    else:

        train_loader = create_full_train_loader(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        train_dataset = dataset

        (
            train_loader,
            train_dataset,
        ) = make_optional_capped_loader(
            loader=train_loader,
            dataset=train_dataset,
            max_samples=args.max_train_samples,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            name="max_train_samples",
        )

        validation_loader = None
        validation_dataset = None

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

    injected, converted = configure_br_lora(
        model=model,
        br_lora_cfg=br_lora_cfg,
        training_cfg=br_training_cfg,
    )

    trainable = posterior_parameters(
        model
    )

    variational_parameter_count = (
        variational_lora_parameter_count(
            model,
            trainable_only=False,
        )
    )

    trainable_parameter_count = sum(
        parameter.numel()
        for parameter in trainable
    )

    trainable_tensor_count = len(
        trainable
    )

    if (
        variational_parameter_count
        != trainable_parameter_count
    ):
        raise RuntimeError(
            "Trainable parameter count does not equal the BR-LoRA "
            "variational posterior parameter count."
        )

    learning_rate = float(
        br_training_cfg.get(
            "learning_rate",
            1.0e-4,
        )
    )

    weight_decay = float(
        br_training_cfg.get(
            "weight_decay",
            0.0,
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
        "BR-LoRA TRAINING STACK CONSTRUCTION"
    )
    print(
        "=" * 78
    )

    print()
    print(
        "Paths"
    )
    print(
        "-" * 78
    )
    print(
        "Baseline config          :",
        args.baseline_config.expanduser().resolve(),
    )
    print(
        "BR-LoRA config           :",
        args.br_lora_config.expanduser().resolve(),
    )
    print(
        "Base checkpoint          :",
        base_checkpoint,
    )
    print(
        "Resume checkpoint        :",
        (
            "none"
            if resume_checkpoint is None
            else resume_checkpoint
        ),
    )
    print(
        "H5 root                  :",
        h5_root,
    )
    print(
        "Manifest                 :",
        manifest,
    )
    print(
        "Future checkpoint dir    :",
        checkpoint_dir,
    )

    print()
    print(
        "Data"
    )
    print(
        "-" * 78
    )
    print(
        "Split mode               :",
        split_mode,
    )
    print(
        "Eligible samples         :",
        f"{len(dataset):,}",
    )
    print(
        "Training samples         :",
        f"{len(train_dataset):,}",
    )

    if validation_dataset is None:
        print(
            "Validation samples       : none (full_train mode)"
        )
    else:
        print(
            "Validation samples       :",
            f"{len(validation_dataset):,}",
        )

    print(
        "Training batches         :",
        f"{len(train_loader):,}",
    )

    if validation_loader is None:
        print(
            "Validation batches       : none (full_train mode)"
        )
    else:
        print(
            "Validation batches       :",
            f"{len(validation_loader):,}",
        )

    print(
        "Batch size               :",
        batch_size,
    )

    print()
    print(
        "Backbone"
    )
    print(
        "-" * 78
    )
    print(
        "Device                   :",
        device,
    )
    print(
        "Strict checkpoint load   : True"
    )
    print(
        "Checkpoint epoch         :",
        baseline_checkpoint.get(
            "epoch",
            baseline_checkpoint.get(
                "epochs"
            ),
        ),
    )

    print()
    print(
        "BR-LoRA"
    )
    print(
        "-" * 78
    )
    print(
        "Injected adapters        :",
        len(
            injected
        ),
    )
    print(
        "Converted adapters       :",
        len(
            converted
        ),
    )
    print(
        "Target layers exact      :",
        injected == converted,
    )
    print(
        "Variational parameters   :",
        f"{variational_parameter_count:,}",
    )
    print(
        "Expected parameters      :",
        f"{EXPECTED_VARIATIONAL_PARAMETER_COUNT:,}",
    )
    print(
        "Parameter count exact    :",
        (
            variational_parameter_count
            == EXPECTED_VARIATIONAL_PARAMETER_COUNT
        ),
    )
    print(
        "Trainable tensors        :",
        trainable_tensor_count,
    )
    print(
        "Expected tensors         :",
        EXPECTED_TRAINABLE_TENSOR_COUNT,
    )
    print(
        "Trainable tensors exact  :",
        (
            trainable_tensor_count
            == EXPECTED_TRAINABLE_TENSOR_COUNT
        ),
    )

    print()
    print(
        "Optimizer"
    )
    print(
        "-" * 78
    )
    print(
        "Type                     :",
        optimizer.__class__.__name__,
    )
    print(
        "Learning rate            :",
        learning_rate,
    )
    print(
        "Weight decay             :",
        weight_decay,
    )

    print()
    print(
        "Diffusion"
    )
    print(
        "-" * 78
    )
    print(
        "Timesteps                :",
        schedule.timesteps,
    )
    print(
        "Beta start               :",
        schedule.beta_start,
    )
    print(
        "Beta end                 :",
        schedule.beta_end,
    )

    expected_ok = (
        variational_parameter_count
        == EXPECTED_VARIATIONAL_PARAMETER_COUNT
        and trainable_tensor_count
        == EXPECTED_TRAINABLE_TENSOR_COUNT
        and injected
        == converted
    )

    print()
    print(
        "=" * 78
    )

    if expected_ok:
        print(
            "CONSTRUCTION VERDICT: PASS"
        )
        print(
            "The trained baseline backbone was restored strictly and "
            "converted to the expected BR-LoRA training stack."
        )

    else:
        print(
            "CONSTRUCTION VERDICT: FAIL"
        )
        raise RuntimeError(
            "BR-LoRA construction did not satisfy the expected contract."
        )

    print(
        "=" * 78
    )

    configured_epochs = int(
        br_training_cfg.get(
            "epochs",
            30,
        )
    )

    epochs = (
        configured_epochs
        if args.epochs is None
        else args.epochs
    )

    if (
        isinstance(
            epochs,
            bool,
        )
        or not isinstance(
            epochs,
            int,
        )
        or epochs <= 0
    ):
        raise ValueError(
            "Training epochs must be a positive integer."
        )

    fit_config = BRLoRAFitConfig(
        epochs=epochs,
        kl_weight=float(
            br_training_cfg.get(
                "kl_weight",
                1.0e-6,
            )
        ),
        kl_warmup_steps=int(
            br_training_cfg.get(
                "kl_warmup_steps",
                1000,
            )
        ),
        outside_loss_weight=float(
            br_training_cfg.get(
                "outside_loss_weight",
                0.05,
            )
        ),
        max_grad_norm=(
            None
            if br_training_cfg.get(
                "max_grad_norm",
                1.0,
            ) is None
            else float(
                br_training_cfg.get(
                    "max_grad_norm",
                    1.0,
                )
            )
        ),
        training_sample_posterior=bool(
            br_training_cfg.get(
                "sample_posterior",
                True,
            )
        ),
        validation_sample_posterior=bool(
            br_training_cfg.get(
                "validation_sample_posterior",
                False,
            )
        ),
    )

    checkpoint_data_config = dict(
        data_cfg
    )

    checkpoint_data_config[
        "effective_training_samples"
    ] = len(
        train_dataset
    )

    checkpoint_data_config[
        "effective_validation_samples"
    ] = (
        None
        if validation_dataset is None
        else len(
            validation_dataset
        )
    )

    checkpoint_data_config[
        "max_train_samples"
    ] = args.max_train_samples

    checkpoint_data_config[
        "max_validation_samples"
    ] = args.max_validation_samples

    initial_state = None
    initial_history = ()

    if resume_checkpoint is not None:

        resume_payload = load_br_lora_checkpoint(
            resume_checkpoint,
            map_location=device,
        )

        validate_resume_metadata(
            payload=resume_payload,
            current_br_lora_config=dict(
                br_lora_cfg
            ),
            current_fit_config=fit_config,
            current_model_config=dict(
                model_cfg
            ),
            current_data_config=checkpoint_data_config,
        )

        restored = restore_br_lora_checkpoint(
            payload=resume_payload,
            model=model,
            optimizer=optimizer,
            strict=True,
            restore_rng=True,
        )

        initial_history = history_from_dicts(
            restored[
                "history"
            ]
        )

        best_completed_epoch = infer_best_completed_epoch(
            history=initial_history,
            best_validation_loss=restored[
                "best_validation_loss"
            ],
        )

        initial_state = BRLoRAFitState(
            completed_epochs=restored[
                "completed_epochs"
            ],
            global_step=restored[
                "global_step"
            ],
            best_validation_loss=restored[
                "best_validation_loss"
            ],
            best_completed_epoch=best_completed_epoch,
        )

        if (
            len(
                initial_history
            )
            != initial_state.completed_epochs
        ):
            raise RuntimeError(
                "Restored history length does not equal the checkpoint "
                "completed-epoch count."
            )

        print()
        print(
            "=" * 78
        )
        print(
            "BR-LoRA RESUME RESTORATION"
        )
        print(
            "=" * 78
        )
        print(
            "Resume mode              : True"
        )
        print(
            "Split mode               :",
            split_mode,
        )
        print(
            "Checkpoint               :",
            resume_checkpoint,
        )
        print(
            "Completed epochs         :",
            initial_state.completed_epochs,
        )
        print(
            "Global step              :",
            initial_state.global_step,
        )
        print(
            "History length           :",
            len(
                initial_history
            ),
        )

        if split_mode == INTERNAL_SPLIT_MODE:
            print(
                "Best completed epoch     :",
                initial_state.best_completed_epoch,
            )
            print(
                "Best validation loss     :",
                initial_state.best_validation_loss,
            )
        else:
            print(
                "Best completed epoch     : not applicable"
            )
            print(
                "Best validation loss     : not applicable"
            )

        print(
            "RNG restored             : True"
        )
        print(
            "=" * 78
        )

    else:

        print()
        print(
            "Resume mode              : False"
        )

    def epoch_callback(
        record,
        state,
        improved,
    ) -> None:
        print()
        print(
            f"Epoch {record.completed_epoch}/"
            f"{fit_config.epochs}"
        )
        print(
            f"Global step              : {state.global_step}"
        )
        print(
            f"Training loss            : "
            f"{record.training_loss:.8f}"
        )
        print(
            f"Training reconstruction  : "
            f"{record.training_reconstruction:.8f}"
        )

        if record.has_validation:
            print(
                f"Validation loss          : "
                f"{record.validation_loss:.8f}"
            )
            print(
                f"Validation reconstruction: "
                f"{record.validation_reconstruction:.8f}"
            )
            print(
                f"Best checkpoint updated  : {improved}"
            )
        else:
            print(
                "Validation               : not performed (full_train mode)"
            )

    print()
    print(
        "=" * 78
    )
    print(
        "BR-LoRA FIT"
    )
    print(
        "=" * 78
    )
    print(
        "Split mode               :",
        split_mode,
    )
    print(
        "Epoch target             :",
        fit_config.epochs,
    )
    print(
        "Starting completed epochs:",
        (
            0
            if initial_state is None
            else initial_state.completed_epochs
        ),
    )
    print(
        "Starting global step     :",
        (
            0
            if initial_state is None
            else initial_state.global_step
        ),
    )
    print(
        "Training samples         :",
        f"{len(train_dataset):,}",
    )

    if validation_dataset is None:
        print(
            "Validation samples       : none (full_train mode)"
        )
    else:
        print(
            "Validation samples       :",
            f"{len(validation_dataset):,}",
        )

    print(
        "Training posterior draws :",
        fit_config.training_sample_posterior,
    )

    if split_mode == INTERNAL_SPLIT_MODE:
        print(
            "Validation posterior draws:",
            fit_config.validation_sample_posterior,
        )
    else:
        print(
            "Validation posterior draws: not applicable"
        )

    result = fit_br_lora(
        model=model,
        train_loader=train_loader,
        validation_loader=validation_loader,
        optimizer=optimizer,
        schedule=schedule,
        device=device,
        config=fit_config,
        output_dir=checkpoint_dir,
        br_lora_config=dict(
            br_lora_cfg
        ),
        model_config=dict(
            model_cfg
        ),
        data_config=checkpoint_data_config,
        initial_state=initial_state,
        initial_history=initial_history,
        epoch_callback=epoch_callback,
    )

    print()
    print(
        "=" * 78
    )
    print(
        "BR-LoRA TRAINING COMPLETE"
    )
    print(
        "=" * 78
    )
    print(
        "Split mode               :",
        result.split_mode,
    )
    print(
        "Completed epochs         :",
        result.state.completed_epochs,
    )
    print(
        "Global step              :",
        result.state.global_step,
    )

    if result.split_mode == INTERNAL_SPLIT_MODE:
        print(
            "Best completed epoch     :",
            result.state.best_completed_epoch,
        )
        print(
            "Best validation loss     :",
            result.state.best_validation_loss,
        )
        print(
            "Latest checkpoint        :",
            result.latest_checkpoint_path,
        )
        print(
            "Best checkpoint          :",
            result.best_checkpoint_path,
        )
    else:
        print(
            "Validation               : not performed"
        )
        print(
            "Latest checkpoint        :",
            result.latest_checkpoint_path,
        )
        print(
            "Final checkpoint         :",
            result.final_checkpoint_path,
        )

    print(
        "=" * 78
    )


if __name__ == "__main__":
    try:
        main()

    except (
        RuntimeError,
        ValueError,
        FileNotFoundError,
        KeyError,
        TypeError,
    ) as exc:
        print(
            "\nBR-LoRA TRAINING FAILED",
            file=sys.stderr,
        )
        print(
            exc,
            file=sys.stderr,
        )
        sys.exit(
            1
        )
