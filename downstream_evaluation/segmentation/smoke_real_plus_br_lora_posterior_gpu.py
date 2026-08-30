#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Subset

from downstream_evaluation.segmentation.dataset import (
    DownstreamBraTSSegmentationDataset,
)
from downstream_evaluation.segmentation.losses import (
    BCEDiceLoss,
)
from downstream_evaluation.segmentation.model import (
    VanillaUNet,
)
from downstream_evaluation.segmentation.posterior_sample_dataset import (
    BRLoRAPosteriorSampleSegmentationDataset,
)
from downstream_evaluation.segmentation.train_real_only import (
    run_train_epoch,
)
from downstream_evaluation.segmentation.train_real_plus_br_lora_posterior import (
    segmentation_collate,
)
from downstream_evaluation.segmentation.transforms import (
    build_train_transform,
)


REPO_ROOT = Path(__file__).resolve().parents[2]

H5_ROOT = Path(
    "/scratch/bhanug/archive/BraTS2020_H5_Rebuilt"
)

BR_LORA_LIBRARY_ROOT = Path(
    "/scratch/bhanug/br_lora_library/batches"
)

REAL_TRAIN_MANIFEST = (
    REPO_ROOT
    / "downstream_evaluation"
    / "manifests"
    / "downstream_real_training_manifest.csv"
)

SYNTHETIC_MANIFEST = (
    REPO_ROOT
    / "downstream_evaluation"
    / "manifests"
    / "br_lora_library_design_10000"
    / "br_lora_library_design_10000.csv"
)

SEED = 42
BATCH_SIZE = 26
REAL_SLICES = 13
SYNTHETIC_CASES = 13


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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

    real_dataset = DownstreamBraTSSegmentationDataset(
        manifest_path=REAL_TRAIN_MANIFEST,
        h5_root=H5_ROOT,
        image_channel=0,
        transform=build_train_transform(),
    )

    synthetic_dataset = BRLoRAPosteriorSampleSegmentationDataset(
        manifest_path=SYNTHETIC_MANIFEST,
        library_root=BR_LORA_LIBRARY_ROOT,
        h5_root=H5_ROOT,
        seed=SEED,
        transform=build_train_transform(),
    )

    combined_dataset = ConcatDataset(
        [
            real_dataset,
            synthetic_dataset,
        ]
    )

    real_indices = list(range(REAL_SLICES))

    synthetic_offset = len(real_dataset)

    synthetic_indices = [
        synthetic_offset + index
        for index in range(SYNTHETIC_CASES)
    ]

    subset = Subset(
        combined_dataset,
        real_indices + synthetic_indices,
    )

    loader = DataLoader(
        subset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        collate_fn=segmentation_collate,
    )

    print()
    print("Smoke-test subset")
    print("Real slices:", REAL_SLICES)
    print("Synthetic cases:", SYNTHETIC_CASES)
    print("Total samples:", len(subset))
    print("Batch size:", BATCH_SIZE)

    criterion = BCEDiceLoss()

    for epoch_index in [0, 1]:
        synthetic_dataset.set_epoch(epoch_index)

        selected = [
            int(
                synthetic_dataset.realization_schedule[
                    case_index,
                    epoch_index,
                ]
            )
            for case_index in range(SYNTHETIC_CASES)
        ]

        if len(set(selected)) == 0:
            raise RuntimeError(
                "No posterior realizations selected."
            )

        print()
        print(
            "Posterior schedule position:",
            epoch_index,
        )
        print(
            "First five selected realization indices:",
            selected[:5],
        )

        model = VanillaUNet(
            in_channels=1,
            out_channels=1,
        ).to(device)

        optimizer = torch.optim.Adamax(
            model.parameters(),
            lr=1e-3,
        )

        train_loss, train_dice = run_train_epoch(
            model=model,
            loader=loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        print("Train loss:", train_loss)
        print("Train Dice:", train_dice)

        if not np.isfinite(train_loss):
            raise RuntimeError(
                "Smoke test produced non-finite loss."
            )

        if not np.isfinite(train_dice):
            raise RuntimeError(
                "Smoke test produced non-finite Dice."
            )

    print()
    print("Experiment 3 GPU smoke test passed.")


if __name__ == "__main__":
    main()
