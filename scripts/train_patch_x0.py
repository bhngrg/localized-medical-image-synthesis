#!/usr/bin/env python3

"""
Train the baseline patch-conditioned x0 diffusion model.

Two explicit split modes are supported:

internal
    Reproduces the notebook: 90/10 internal split by default, stochastic
    validation loss, and best-checkpoint selection.

full_train
    Uses 100% of eligible training slices for a fixed number of epochs. No
    masked validation loss is computed because no internal validation subset is
    created.
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

import numpy as np
import torch
import yaml

from src.data import (
    BraTSH5PatchX0Dataset,
    create_train_val_loaders,
)

from src.data.loaders import create_full_train_loader
from src.diffusion import DiffusionSchedule
from src.models import AppearanceX0UNet
from src.training import (
    build_checkpoint_payload,
    fit,
    save_checkpoint,
)
from src.training.trainer import fit_full_train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train the baseline patch-conditioned x0 diffusion model."
        )
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "configs/baseline_patch_x0.yaml"
        ),
    )

    parser.add_argument(
        "--h5-root",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path(
            "checkpoints"
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


def load_config(
    path: Path,
) -> dict:
    path = path.expanduser().resolve()

    if not path.is_file():
        raise ValueError(
            f"Configuration file not found:\n{path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(
            "Configuration must contain a YAML mapping."
        )

    return config


def resolve_device(
    requested: str,
) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but is not available."
            )
        return torch.device("cuda")

    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError(
                "MPS was requested but is not available."
            )
        return torch.device("mps")

    if requested == "cpu":
        return torch.device("cpu")

    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    seed = int(
        config.get(
            "seed",
            42,
        )
    )

    data_cfg = config["data"]
    model_cfg = config["model"]
    training_cfg = config["training"]
    diffusion_cfg = config["diffusion"]

    split_mode = str(
        data_cfg.get(
            "split_mode",
            "internal",
        )
    )

    if split_mode not in {
        "internal",
        "full_train",
    }:
        raise ValueError(
            "data.split_mode must be 'internal' or 'full_train'."
        )

    set_seed(seed)
    device = resolve_device(args.device)

    print("Using device:", device)
    print("Split mode:", split_mode)

    dataset = BraTSH5PatchX0Dataset(
        root=args.h5_root,
        manifest_path=args.manifest,
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

    if split_mode == "internal":
        (
            train_loader,
            val_loader,
            train_dataset,
            val_dataset,
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

        print(
            f"Training samples   : {len(train_dataset):,}"
        )

        print(
            f"Validation samples : {len(val_dataset):,}"
        )

    else:
        train_loader = create_full_train_loader(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        val_loader = None

        print(
            f"Training samples   : {len(dataset):,}"
        )

        print(
            "Validation samples : none "
            "(full_train mode)"
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

    model = AppearanceX0UNet(
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
    ).to(device)

    learning_rate = float(
        training_cfg.get(
            "learning_rate",
            1.0e-4,
        )
    )

    weight_decay = float(
        training_cfg.get(
            "weight_decay",
            0.01,
        )
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    epochs = int(
        training_cfg.get(
            "epochs",
            30,
        )
    )

    outside_loss_weight = float(
        training_cfg.get(
            "outside_loss_weight",
            0.05,
        )
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
            "patch_conditioned_x0_diffusion"
        ),
    }

    if split_mode == "internal":
        best_checkpoint_path = (
            args.checkpoint_dir
            / "best_patch_x0_diffusion.pt"
        )

        final_checkpoint_path = (
            args.checkpoint_dir
            / "final_patch_x0_diffusion.pt"
        )

        (
            train_losses,
            val_losses,
            best_val_loss,
        ) = fit(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            schedule=schedule,
            device=device,
            epochs=epochs,
            outside_loss_weight=outside_loss_weight,
            best_checkpoint_path=best_checkpoint_path,
            checkpoint_metadata=checkpoint_metadata,
        )

        final_payload = build_checkpoint_payload(
            model=model,
            optimizer=optimizer,
            epoch=None,
            epochs=epochs,
            final_train_loss=train_losses[-1],
            final_val_loss=val_losses[-1],
            best_val_loss=best_val_loss,
            **checkpoint_metadata,
        )

        save_checkpoint(
            final_checkpoint_path,
            final_payload,
        )

        print(
            f"Saved final model to {final_checkpoint_path}"
        )

    else:
        checkpoint_metadata[
            "split_mode"
        ] = "full_train"

        final_checkpoint_path = (
            args.checkpoint_dir
            / "final_patch_x0_diffusion_full_train.pt"
        )

        train_losses = fit_full_train(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            schedule=schedule,
            device=device,
            epochs=epochs,
            outside_loss_weight=outside_loss_weight,
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

        save_checkpoint(
            final_checkpoint_path,
            final_payload,
        )

        print(
            f"Saved final model to {final_checkpoint_path}"
        )


if __name__ == "__main__":
    try:
        main()

    except (
        RuntimeError,
        ValueError,
        FileNotFoundError,
        KeyError,
    ) as exc:
        print(
            "\nTRAINING FAILED",
            file=sys.stderr,
        )
        print(
            exc,
            file=sys.stderr,
        )
        sys.exit(1)
