#!/usr/bin/env python3

from __future__ import annotations

import csv
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, get_worker_info

from downstream_evaluation.segmentation.dataset import (
    DownstreamBraTSSegmentationDataset,
)
from downstream_evaluation.segmentation.losses import (
    BCEDiceLoss,
)
from downstream_evaluation.segmentation.metrics import (
    dice_coefficient,
    iou_score,
)
from downstream_evaluation.segmentation.model import (
    VanillaUNet,
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

OUTPUT_ROOT = (
    REPO_ROOT
    / "outputs"
    / "downstream_segmentation"
    / "real_only_seed42_a30_normal_q"
)

CHECKPOINT_PATH = OUTPUT_ROOT / "best_model.pt"
HISTORY_PATH = OUTPUT_ROOT / "training_history.csv"


SEED = 42
IMAGE_CHANNEL = 0
BATCH_SIZE = 26
NUM_WORKERS = 4
NUM_EPOCHS = 20
LEARNING_RATE = 1e-3
THRESHOLD = 0.5


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)

    random.seed(worker_seed)
    np.random.seed(worker_seed)

    worker_info = get_worker_info()

    if worker_info is None:
        return

    transform = getattr(
        worker_info.dataset,
        "transform",
        None,
    )

    if transform is not None and hasattr(
        transform,
        "set_random_seed",
    ):
        transform.set_random_seed(worker_seed)


def run_train_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    model.train()

    total_loss = 0.0
    total_dice = 0.0
    total_samples = 0

    for batch in loader:
        images = batch["image"].to(
            device,
            non_blocking=True,
        )
        masks = batch["mask"].to(
            device,
            non_blocking=True,
        )

        optimizer.zero_grad(
            set_to_none=True,
        )

        logits = model(images)

        loss = criterion(
            logits,
            masks,
        )

        loss.backward()
        optimizer.step()

        batch_size = images.shape[0]

        with torch.no_grad():
            dice = dice_coefficient(
                logits,
                masks,
                threshold=THRESHOLD,
            )

        total_loss += float(loss.item()) * batch_size
        total_dice += float(dice.item()) * batch_size
        total_samples += batch_size

    return (
        total_loss / total_samples,
        total_dice / total_samples,
    )


@torch.no_grad()
def run_validation_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
) -> tuple[float, float, float, float, float]:
    model.eval()

    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    total_samples = 0

    positive_dice_sum = 0.0
    positive_iou_sum = 0.0
    positive_samples = 0

    for batch in loader:
        images = batch["image"].to(
            device,
            non_blocking=True,
        )
        masks = batch["mask"].to(
            device,
            non_blocking=True,
        )

        logits = model(images)

        loss = criterion(
            logits,
            masks,
        )

        dice = dice_coefficient(
            logits,
            masks,
            threshold=THRESHOLD,
        )

        iou = iou_score(
            logits,
            masks,
            threshold=THRESHOLD,
        )

        batch_size = images.shape[0]

        total_loss += float(loss.item()) * batch_size
        total_dice += float(dice.item()) * batch_size
        total_iou += float(iou.item()) * batch_size
        total_samples += batch_size

        positive_mask = (
            masks.sum(dim=(1, 2, 3)) > 0
        )

        n_positive = int(
            positive_mask.sum().item()
        )

        if n_positive > 0:
            positive_logits = logits[positive_mask]
            positive_targets = masks[positive_mask]

            positive_dice = dice_coefficient(
                positive_logits,
                positive_targets,
                threshold=THRESHOLD,
            )

            positive_iou = iou_score(
                positive_logits,
                positive_targets,
                threshold=THRESHOLD,
            )

            positive_dice_sum += (
                float(positive_dice.item())
                * n_positive
            )

            positive_iou_sum += (
                float(positive_iou.item())
                * n_positive
            )

            positive_samples += n_positive

    if positive_samples == 0:
        raise RuntimeError(
            "Validation set contains no tumor-positive slices."
        )

    return (
        total_loss / total_samples,
        total_dice / total_samples,
        total_iou / total_samples,
        positive_dice_sum / positive_samples,
        positive_iou_sum / positive_samples,
    )


def main() -> None:
    set_seed(SEED)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print("Device:", device)
    print("Seed:", SEED)

    train_transform = build_train_transform()
    validation_transform = build_validation_transform()

    if hasattr(
        train_transform,
        "set_random_seed",
    ):
        train_transform.set_random_seed(SEED)

    train_dataset = DownstreamBraTSSegmentationDataset(
        manifest_path=TRAIN_MANIFEST,
        h5_root=H5_ROOT,
        image_channel=IMAGE_CHANNEL,
        transform=train_transform,
    )

    validation_dataset = DownstreamBraTSSegmentationDataset(
        manifest_path=VALIDATION_MANIFEST,
        h5_root=H5_ROOT,
        image_channel=IMAGE_CHANNEL,
        transform=validation_transform,
    )

    print("Training slices:", len(train_dataset))
    print("Validation slices:", len(validation_dataset))

    generator = torch.Generator()
    generator.manual_seed(SEED)

    pin_memory = device.type == "cuda"

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker,
        generator=generator,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker,
        generator=generator,
    )

    model = VanillaUNet(
        in_channels=1,
        out_channels=1,
    ).to(device)

    criterion = BCEDiceLoss()

    optimizer = torch.optim.Adamax(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    best_validation_positive_dice = -1.0

    history = []

    print()
    print("Starting real-only downstream training.")
    print("Epochs:", NUM_EPOCHS)
    print("Batch size:", BATCH_SIZE)
    print("Learning rate:", LEARNING_RATE)
    print()

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_dice = run_train_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

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

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_dice": train_dice,
            "validation_loss": validation_loss,
            "validation_dice": validation_dice,
            "validation_iou": validation_iou,
            "validation_positive_dice": validation_positive_dice,
            "validation_positive_iou": validation_positive_iou,
        }

        history.append(row)

        print(
            f"Epoch {epoch:02d}/{NUM_EPOCHS} | "
            f"train_loss={train_loss:.6f} | "
            f"train_dice={train_dice:.6f} | "
            f"val_loss={validation_loss:.6f} | "
            f"val_dice={validation_dice:.6f} | "
            f"val_iou={validation_iou:.6f} | "
            f"val_positive_dice={validation_positive_dice:.6f} | "
            f"val_positive_iou={validation_positive_iou:.6f}"
        )

        if (
            validation_positive_dice
            > best_validation_positive_dice
        ):
            best_validation_positive_dice = (
                validation_positive_dice
            )

            # Create the output directory immediately before writing.
            OUTPUT_ROOT.mkdir(
                parents=True,
                exist_ok=True,
            )

            torch.save(
                {
                    "epoch": epoch,
                    "seed": SEED,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "validation_dice": validation_dice,
                    "validation_iou": validation_iou,
                    "validation_positive_dice": validation_positive_dice,
                    "validation_positive_iou": validation_positive_iou,
                    "train_manifest": str(
                        TRAIN_MANIFEST.relative_to(REPO_ROOT)
                    ),
                    "validation_manifest": str(
                        VALIDATION_MANIFEST.relative_to(REPO_ROOT)
                    ),
                    "image_channel": IMAGE_CHANNEL,
                    "batch_size": BATCH_SIZE,
                    "learning_rate": LEARNING_RATE,
                    "threshold": THRESHOLD,
                },
                CHECKPOINT_PATH,
            )

            print(
                "  Saved new best checkpoint:",
                CHECKPOINT_PATH,
            )

        # Create the output directory immediately before writing.
        OUTPUT_ROOT.mkdir(
            parents=True,
            exist_ok=True,
        )

        with HISTORY_PATH.open(
            "w",
            newline="",
        ) as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "epoch",
                    "train_loss",
                    "train_dice",
                    "validation_loss",
                    "validation_dice",
                    "validation_iou",
                    "validation_positive_dice",
                    "validation_positive_iou",
                ],
            )

            writer.writeheader()
            writer.writerows(history)

    print()
    print("Training complete.")
    print(
        "Best tumor-positive validation Dice:",
        best_validation_positive_dice,
    )
    print("Best checkpoint:", CHECKPOINT_PATH)
    print("Training history:", HISTORY_PATH)


if __name__ == "__main__":
    main()
