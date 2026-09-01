#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import random

import numpy as np
import torch
import yaml
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
from src.config import (
    load_folders_config,
    resolve_path,
    save_folders_config,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_CONFIG = (
    PROJECT_ROOT
    / "downstream_evaluation"
    / "configs"
    / "segmentation.yaml"
)

DEFAULT_REAL_TRAIN_MANIFEST = (
    PROJECT_ROOT
    / "downstream_evaluation"
    / "manifests"
    / "downstream_real_training_manifest.csv"
)

DEFAULT_SYNTHETIC_MANIFEST = (
    PROJECT_ROOT
    / "downstream_evaluation"
    / "manifests"
    / "br_lora_library_design_10000"
    / "br_lora_library_design_10000.csv"
)

N_STEPS = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose bitwise CUDA reproducibility for the downstream "
            "real + BR-LoRA posterior-mean training path."
        )
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Tracked downstream segmentation configuration YAML.",
    )

    parser.add_argument(
        "--folders-file",
        type=Path,
        default=Path("data/folders.yaml"),
        help="Machine-specific path configuration YAML.",
    )

    parser.add_argument(
        "--h5-root",
        type=Path,
        default=None,
        help="BraTS H5 root. Overrides h5_root in --folders-file.",
    )

    parser.add_argument(
        "--br-lora-library-root",
        type=Path,
        default=None,
        help=(
            "BR-LoRA library root. Overrides br_lora_library_root "
            "in --folders-file."
        ),
    )

    parser.add_argument(
        "--real-train-manifest",
        type=Path,
        default=None,
        help=(
            "Frozen real-training manifest. CLI overrides folders YAML; "
            "otherwise the tracked repository manifest is used."
        ),
    )

    parser.add_argument(
        "--synthetic-manifest",
        type=Path,
        default=None,
        help=(
            "Frozen BR-LoRA synthetic-library manifest. CLI overrides "
            "folders YAML; otherwise the tracked repository manifest "
            "is used."
        ),
    )

    return parser.parse_args()


def load_config(path: Path) -> dict:
    path = path.expanduser().resolve()

    if not path.is_file():
        raise ValueError(
            f"Configuration file not found:\n{path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        loaded = yaml.safe_load(file)

    if not isinstance(loaded, dict):
        raise ValueError(
            "Configuration must contain a YAML mapping."
        )

    return loaded


def resolve_optional_repo_path(
    *,
    key: str,
    cli_value: Path | None,
    folders_config: dict[str, str],
    default: Path,
) -> Path:
    """
    Resolve CLI > folders YAML > tracked repository default.
    """
    if cli_value is not None:
        path = Path(cli_value)
        folders_config[key] = str(path)
        return path

    configured = folders_config.get(key)

    if configured:
        return Path(configured)

    return default


def resolve_paths(
    args: argparse.Namespace,
) -> dict[str, Path]:
    folders_config = load_folders_config(
        args.folders_file
    )

    h5_root = resolve_path(
        key="h5_root",
        cli_value=args.h5_root,
        config=folders_config,
        selector=None,
    )

    library_root = resolve_path(
        key="br_lora_library_root",
        cli_value=args.br_lora_library_root,
        config=folders_config,
        selector=None,
    )

    real_train_manifest = resolve_optional_repo_path(
        key="downstream_real_training_manifest",
        cli_value=args.real_train_manifest,
        folders_config=folders_config,
        default=DEFAULT_REAL_TRAIN_MANIFEST,
    )

    synthetic_manifest = resolve_optional_repo_path(
        key="downstream_synthetic_manifest",
        cli_value=args.synthetic_manifest,
        folders_config=folders_config,
        default=DEFAULT_SYNTHETIC_MANIFEST,
    )

    save_folders_config(
        args.folders_file,
        folders_config,
    )

    return {
        "h5_root": Path(h5_root),
        "library_root": Path(library_root),
        "real_train_manifest": Path(real_train_manifest),
        "synthetic_manifest": Path(synthetic_manifest),
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


def build_loader(
    *,
    seed: int,
    batch_size: int,
    num_workers: int,
    h5_root: Path,
    library_root: Path,
    real_train_manifest: Path,
    synthetic_manifest: Path,
) -> DataLoader:
    real_transform = build_train_transform()
    synthetic_transform = build_train_transform()

    if hasattr(real_transform, "set_random_seed"):
        real_transform.set_random_seed(seed)

    if hasattr(
        synthetic_transform,
        "set_random_seed",
    ):
        synthetic_transform.set_random_seed(seed)

    real = DownstreamBraTSSegmentationDataset(
        manifest_path=real_train_manifest,
        h5_root=h5_root,
        image_channel=0,
        transform=real_transform,
    )

    synthetic = BRLoRAPosteriorMeanSegmentationDataset(
        manifest_path=synthetic_manifest,
        library_root=library_root,
        h5_root=h5_root,
        transform=synthetic_transform,
    )

    dataset = ConcatDataset(
        [real, synthetic]
    )

    generator = torch.Generator()
    generator.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
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


def run_once(
    run_label: str,
    *,
    seed: int,
    batch_size: int,
    num_workers: int,
    learning_rate: float,
    paths: dict[str, Path],
) -> dict[str, object]:
    set_seed(seed)

    loader = build_loader(
        seed=seed,
        batch_size=batch_size,
        num_workers=num_workers,
        h5_root=paths["h5_root"],
        library_root=paths["library_root"],
        real_train_manifest=paths["real_train_manifest"],
        synthetic_manifest=paths["synthetic_manifest"],
    )

    device = torch.device("cuda")

    model = VanillaUNet(
        in_channels=1,
        out_channels=1,
    ).to(device)

    criterion = BCEDiceLoss()

    optimizer = torch.optim.Adamax(
        model.parameters(),
        lr=learning_rate,
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

    args = parse_args()
    config = load_config(args.config)
    paths = resolve_paths(args)

    seed = int(config["seed"])
    batch_size = int(config["data"]["batch_size"])
    num_workers = int(config["data"]["num_workers"])
    learning_rate = float(
        config["training"]["learning_rate"]
    )

    print("GPU:", torch.cuda.get_device_name(0))
    print("PyTorch:", torch.__version__)
    print("Seed:", seed)
    print("Batch size:", batch_size)
    print("Workers:", num_workers)
    print("Learning rate:", learning_rate)
    print("Steps:", N_STEPS)
    print("H5 root:", paths["h5_root"])
    print(
        "BR-LoRA library root:",
        paths["library_root"],
    )
    print(
        "Real-training manifest:",
        paths["real_train_manifest"],
    )
    print(
        "Synthetic manifest:",
        paths["synthetic_manifest"],
    )
    print(
        "cudnn deterministic:",
        torch.backends.cudnn.deterministic,
    )
    print(
        "cudnn benchmark:",
        torch.backends.cudnn.benchmark,
    )

    run_a = run_once(
        "RUN A",
        seed=seed,
        batch_size=batch_size,
        num_workers=num_workers,
        learning_rate=learning_rate,
        paths=paths,
    )

    run_b = run_once(
        "RUN B",
        seed=seed,
        batch_size=batch_size,
        num_workers=num_workers,
        learning_rate=learning_rate,
        paths=paths,
    )

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
