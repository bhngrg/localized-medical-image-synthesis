#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import random

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
from downstream_evaluation.segmentation.losses import BCEDiceLoss
from downstream_evaluation.segmentation.model import VanillaUNet
from downstream_evaluation.segmentation.synthetic_dataset import (
    BRLoRAPosteriorMeanSegmentationDataset,
)
from downstream_evaluation.segmentation.transforms import (
    build_train_transform,
)
from downstream_evaluation.segmentation.train_real_plus_br_lora import (
    BATCH_SIZE,
    BR_LORA_LIBRARY_ROOT,
    H5_ROOT,
    LEARNING_RATE,
    NUM_WORKERS,
    REAL_TRAIN_MANIFEST,
    SEED,
    SYNTHETIC_MANIFEST,
)


N_STEPS = 10


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    torch.use_deterministic_algorithms(True)


def seed_dataset_transform(dataset, seed: int) -> None:
    if isinstance(dataset, ConcatDataset):
        for child in dataset.datasets:
            seed_dataset_transform(child, seed)
        return

    transform = getattr(dataset, "transform", None)

    if transform is not None and hasattr(
        transform,
        "set_random_seed",
    ):
        transform.set_random_seed(seed)


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)

    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)

    info = get_worker_info()

    if info is not None:
        seed_dataset_transform(
            info.dataset,
            worker_seed,
        )


def collate(batch):
    return {
        "image": torch.stack(
            [sample["image"] for sample in batch]
        ),
        "mask": torch.stack(
            [sample["mask"] for sample in batch]
        ),
    }


def build_loader() -> DataLoader:
    real_transform = build_train_transform()
    synthetic_transform = build_train_transform()

    if hasattr(real_transform, "set_random_seed"):
        real_transform.set_random_seed(SEED)

    if hasattr(
        synthetic_transform,
        "set_random_seed",
    ):
        synthetic_transform.set_random_seed(SEED)

    real = DownstreamBraTSSegmentationDataset(
        manifest_path=REAL_TRAIN_MANIFEST,
        h5_root=H5_ROOT,
        image_channel=0,
        transform=real_transform,
    )

    synthetic = BRLoRAPosteriorMeanSegmentationDataset(
        manifest_path=SYNTHETIC_MANIFEST,
        library_root=BR_LORA_LIBRARY_ROOT,
        h5_root=H5_ROOT,
        transform=synthetic_transform,
    )

    dataset = ConcatDataset(
        [real, synthetic]
    )

    generator = torch.Generator()
    generator.manual_seed(SEED)

    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=generator,
        collate_fn=collate,
    )


def state_hash(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()

    for name, tensor in model.state_dict().items():
        digest.update(name.encode())
        digest.update(
            tensor.detach()
            .cpu()
            .contiguous()
            .numpy()
            .tobytes()
        )

    return digest.hexdigest()


def run_once(run_label: str) -> dict[str, object]:
    set_seed(SEED)

    loader = build_loader()

    device = torch.device("cuda")

    model = VanillaUNet(
        in_channels=1,
        out_channels=1,
    ).to(device)

    criterion = BCEDiceLoss()

    optimizer = torch.optim.Adamax(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    initial_hash = state_hash(model)

    losses = []

    model.train()

    for step, batch in enumerate(loader, start=1):
        if step > N_STEPS:
            break

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

        losses.append(float(loss.item()))

    torch.cuda.synchronize()

    final_hash = state_hash(model)

    print()
    print(run_label)
    print("Initial model hash:", initial_hash)

    for index, value in enumerate(
        losses,
        start=1,
    ):
        print(
            f"Step {index:02d} loss: {value:.12f}"
        )

    print("Final model hash:", final_hash)

    return {
        "initial_hash": initial_hash,
        "losses": losses,
        "final_hash": final_hash,
    }


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for this diagnostic."
        )

    print("GPU:", torch.cuda.get_device_name(0))
    print("PyTorch:", torch.__version__)
    print("Seed:", SEED)
    print("Steps:", N_STEPS)
    print(
        "cudnn deterministic:",
        torch.backends.cudnn.deterministic,
    )
    print(
        "cudnn benchmark:",
        torch.backends.cudnn.benchmark,
    )

    run_a = run_once("RUN A")
    run_b = run_once("RUN B")

    same_initial = (
        run_a["initial_hash"]
        == run_b["initial_hash"]
    )

    same_losses = (
        np.asarray(run_a["losses"]).tobytes()
        == np.asarray(run_b["losses"]).tobytes()
    )

    same_final = (
        run_a["final_hash"]
        == run_b["final_hash"]
    )

    print()
    print("=" * 80)
    print("Initial weights identical:", same_initial)
    print("Loss sequence identical   :", same_losses)
    print("Final weights identical   :", same_final)

    if (
        same_initial
        and same_losses
        and same_final
    ):
        print(
            "CURRENT CUDA TRAINING PATH: "
            "BITWISE REPRODUCIBLE FOR THIS TEST"
        )
    else:
        print(
            "CURRENT CUDA TRAINING PATH: DIVERGED"
        )


if __name__ == "__main__":
    main()
