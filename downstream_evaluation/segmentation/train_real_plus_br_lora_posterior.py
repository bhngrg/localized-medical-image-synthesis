#!/usr/bin/env python3

from __future__ import annotations

import csv
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import (
    ConcatDataset,
    DataLoader,
    get_worker_info,
)

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
    / "real_plus_br_lora_posterior_seed42_a30_normal_q"
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


def segmentation_collate(
    batch: list[dict[str, object]],
) -> dict[str, torch.Tensor]:
    return {
        "image": torch.stack(
            [sample["image"] for sample in batch]
        ),
        "mask": torch.stack(
            [sample["mask"] for sample in batch]
        ),
    }


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _seed_dataset_transform(
    dataset,
    seed: int,
) -> None:
    if isinstance(dataset, ConcatDataset):
        for child_dataset in dataset.datasets:
            _seed_dataset_transform(
                child_dataset,
                seed,
            )
        return

    transform = getattr(
        dataset,
        "transform",
        None,
    )

    if transform is not None and hasattr(
        transform,
        "set_random_seed",
    ):
        transform.set_random_seed(seed)


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)

    random.seed(worker_seed)
    np.random.seed(worker_seed)

    worker_info = get_worker_info()

    if worker_info is None:
        return

    _seed_dataset_transform(
        worker_info.dataset,
        worker_seed,
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

    real_train_transform = build_train_transform()
    synthetic_train_transform = build_train_transform()
    validation_transform = build_validation_transform()

    if hasattr(
        real_train_transform,
        "set_random_seed",
    ):
        real_train_transform.set_random_seed(SEED)

    if hasattr(
        synthetic_train_transform,
        "set_random_seed",
    ):
        synthetic_train_transform.set_random_seed(SEED)

    real_train_dataset = DownstreamBraTSSegmentationDataset(
        manifest_path=REAL_TRAIN_MANIFEST,
        h5_root=H5_ROOT,
        image_channel=IMAGE_CHANNEL,
        transform=real_train_transform,
    )

    synthetic_train_dataset = (
        BRLoRAPosteriorSampleSegmentationDataset(
            manifest_path=SYNTHETIC_MANIFEST,
            library_root=BR_LORA_LIBRARY_ROOT,
            h5_root=H5_ROOT,
            seed=SEED,
            transform=synthetic_train_transform,
        )
    )

    train_dataset = ConcatDataset(
        [
            real_train_dataset,
            synthetic_train_dataset,
        ]
    )

    validation_dataset = DownstreamBraTSSegmentationDataset(
        manifest_path=VALIDATION_MANIFEST,
        h5_root=H5_ROOT,
        image_channel=IMAGE_CHANNEL,
        transform=validation_transform,
    )

    print("Real training slices:", len(real_train_dataset))
    print(
        "Synthetic training cases:",
        len(synthetic_train_dataset),
    )
    print("Combined training slices:", len(train_dataset))
    print("Validation slices:", len(validation_dataset))

    if len(real_train_dataset) != 41460:
        raise RuntimeError(
            "Expected exactly 41,460 real training slices."
        )

    if len(synthetic_train_dataset) != 10000:
        raise RuntimeError(
            "Expected exactly 10,000 synthetic training cases."
        )

    if len(train_dataset) != 51460:
        raise RuntimeError(
            "Expected exactly 51,460 combined training slices."
        )

    if len(validation_dataset) != 5735:
        raise RuntimeError(
            "Expected exactly 5,735 validation slices."
        )

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
        collate_fn=segmentation_collate,
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
    print(
        "Starting real + BR-LoRA posterior-sampling "
        "downstream training."
    )
    print("Epochs:", NUM_EPOCHS)
    print("Batch size:", BATCH_SIZE)
    print("Learning rate:", LEARNING_RATE)
    print(
        "Posterior realizations available per case:",
        synthetic_train_dataset.POSTERIOR_SAMPLES,
    )
    print(
        "Distinct posterior realizations used per case:",
        NUM_EPOCHS,
    )
    print()

    for epoch in range(1, NUM_EPOCHS + 1):
        synthetic_train_dataset.set_epoch(
            epoch - 1
        )

        print(
            f"Epoch {epoch:02d}: "
            f"posterior schedule position {epoch - 1}"
        )

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
            "posterior_schedule_position": (
                epoch - 1
            ),
            "train_loss": train_loss,
            "train_dice": train_dice,
            "validation_loss": validation_loss,
            "validation_dice": validation_dice,
            "validation_iou": validation_iou,
            "validation_positive_dice": (
                validation_positive_dice
            ),
            "validation_positive_iou": (
                validation_positive_iou
            ),
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
                    "validation_positive_dice": (
                        validation_positive_dice
                    ),
                    "validation_positive_iou": (
                        validation_positive_iou
                    ),
                    "real_train_manifest": str(
                        REAL_TRAIN_MANIFEST.relative_to(
                            REPO_ROOT
                        )
                    ),
                    "synthetic_manifest": str(
                        SYNTHETIC_MANIFEST.relative_to(
                            REPO_ROOT
                        )
                    ),
                    "validation_manifest": str(
                        VALIDATION_MANIFEST.relative_to(
                            REPO_ROOT
                        )
                    ),
                    "real_training_slices": len(
                        real_train_dataset
                    ),
                    "synthetic_training_cases": len(
                        synthetic_train_dataset
                    ),
                    "combined_training_slices": len(
                        train_dataset
                    ),
                    "posterior_samples_available": (
                        synthetic_train_dataset.POSTERIOR_SAMPLES
                    ),
                    "distinct_posterior_samples_per_case": (
                        NUM_EPOCHS
                    ),
                    "posterior_schedule_position": (
                        epoch - 1
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
                    "posterior_schedule_position",
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
