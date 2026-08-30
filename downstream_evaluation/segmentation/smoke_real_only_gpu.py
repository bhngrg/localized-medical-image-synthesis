#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from downstream_evaluation.segmentation.dataset import (
    DownstreamBraTSSegmentationDataset,
)
from downstream_evaluation.segmentation.losses import (
    BCEDiceLoss,
)
from downstream_evaluation.segmentation.model import (
    VanillaUNet,
)
from downstream_evaluation.segmentation.train_real_only import (
    run_train_epoch,
    run_validation_epoch,
)
from downstream_evaluation.segmentation.transforms import (
    build_train_transform,
    build_validation_transform,
)


REPO_ROOT = Path(__file__).resolve().parents[2]

H5_ROOT = Path(
    "/scratch/bhanug/archive/BraTS2020_H5_Rebuilt"
)

TRAIN_MANIFEST = (
    REPO_ROOT
    / "downstream_evaluation"
    / "manifests"
    / "downstream_real_training_manifest.csv"
)

VALIDATION_MANIFEST = (
    REPO_ROOT
    / "downstream_evaluation"
    / "manifests"
    / "downstream_validation_manifest.csv"
)

SEED = 42
BATCH_SIZE = 26

TRAIN_POSITIVE_SLICES = 13
TRAIN_EMPTY_SLICES = 13

VALIDATION_POSITIVE_SLICES = 13
VALIDATION_EMPTY_SLICES = 13


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_mixed_indices(
    manifest: pd.DataFrame,
    positive_count: int,
    empty_count: int,
) -> list[int]:
    positive = manifest.index[
        manifest["has_tumor"].astype(bool)
    ].tolist()

    empty = manifest.index[
        ~manifest["has_tumor"].astype(bool)
    ].tolist()

    if len(positive) < positive_count:
        raise RuntimeError(
            "Not enough tumor-positive slices for smoke test."
        )

    if len(empty) < empty_count:
        raise RuntimeError(
            "Not enough tumor-free slices for smoke test."
        )

    return (
        positive[:positive_count]
        + empty[:empty_count]
    )


def main() -> None:
    set_seed(SEED)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. "
            "Run this smoke test inside a GPU Slurm allocation."
        )

    device = torch.device("cuda")

    print("Device:", device)
    print("CUDA device:", torch.cuda.get_device_name(0))
    print("CUDA device count:", torch.cuda.device_count())

    train_manifest = pd.read_csv(TRAIN_MANIFEST)
    validation_manifest = pd.read_csv(VALIDATION_MANIFEST)

    train_indices = choose_mixed_indices(
        train_manifest,
        positive_count=TRAIN_POSITIVE_SLICES,
        empty_count=TRAIN_EMPTY_SLICES,
    )

    validation_indices = choose_mixed_indices(
        validation_manifest,
        positive_count=VALIDATION_POSITIVE_SLICES,
        empty_count=VALIDATION_EMPTY_SLICES,
    )

    train_dataset = DownstreamBraTSSegmentationDataset(
        manifest_path=TRAIN_MANIFEST,
        h5_root=H5_ROOT,
        image_channel=0,
        transform=build_train_transform(),
    )

    validation_dataset = DownstreamBraTSSegmentationDataset(
        manifest_path=VALIDATION_MANIFEST,
        h5_root=H5_ROOT,
        image_channel=0,
        transform=build_validation_transform(),
    )

    train_subset = Subset(
        train_dataset,
        train_indices,
    )

    validation_subset = Subset(
        validation_dataset,
        validation_indices,
    )

    train_loader = DataLoader(
        train_subset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    validation_loader = DataLoader(
        validation_subset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    model = VanillaUNet(
        in_channels=1,
        out_channels=1,
    ).to(device)

    criterion = BCEDiceLoss()

    optimizer = torch.optim.Adamax(
        model.parameters(),
        lr=1e-3,
    )

    print()
    print("Smoke-test subset")
    print("Training slices:", len(train_subset))
    print("Validation slices:", len(validation_subset))
    print("Batch size:", BATCH_SIZE)

    print()
    print("Running training smoke epoch...")

    train_loss, train_dice = run_train_epoch(
        model=model,
        loader=train_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
    )

    print("Train loss:", train_loss)
    print("Train Dice:", train_dice)

    print()
    print("Running validation smoke epoch...")

    (
        validation_loss,
        validation_dice,
        validation_iou,
        validation_positive_dice,
        validation_positive_iou,
    ) = run_validation_epoch(
        model=model,
        loader=validation_loader,
        criterion=criterion,
        device=device,
    )

    print("Validation loss:", validation_loss)
    print("Validation Dice:", validation_dice)
    print("Validation IoU:", validation_iou)
    print(
        "Validation tumor-positive Dice:",
        validation_positive_dice,
    )
    print(
        "Validation tumor-positive IoU:",
        validation_positive_iou,
    )

    values = [
        train_loss,
        train_dice,
        validation_loss,
        validation_dice,
        validation_iou,
        validation_positive_dice,
        validation_positive_iou,
    ]

    if not all(np.isfinite(value) for value in values):
        raise RuntimeError(
            "Smoke test produced non-finite metrics."
        )

    print()
    print("GPU smoke test passed.")


if __name__ == "__main__":
    main()
