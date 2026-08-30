#!/usr/bin/env python3

"""
Train the downstream tumor-segmentation model.

Three regimes are supported:

real_only
    Train on the frozen real BraTS training manifest.

real_plus_br_lora_mean
    Train on the same real data plus the fixed BR-LoRA posterior-mean
    synthetic library.

real_plus_br_lora_posterior
    Train on the same real data plus one deterministic posterior realization
    per synthetic case per epoch.

Scientific defaults reproduce the preliminary downstream experiments. The
hardened implementation additionally enables the strict CUDA reproducibility
controls validated by the repository's GPU reproducibility diagnostic.

Provenance
----------
The downstream segmentation idea and vanilla U-Net structure were adapted in
part from:

    https://github.com/edaaydinea/Low-Grade-Glioma-Segmentation

The present implementation was rewritten for the BraTS/UCSF-PDGM setting and
uses this repository's preprocessing, frozen manifests, BR-LoRA synthetic
libraries, loss/metric implementation, and reproducibility controls.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )

import torch
from torch.utils.data import ConcatDataset, DataLoader
import yaml

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
from downstream_evaluation.segmentation.reproducibility import (
    DEFAULT_CUBLAS_WORKSPACE_CONFIG,
    configure_reproducibility,
    make_generator,
    seed_dataset_transform,
    seed_worker,
    validate_cuda_reproducibility_environment,
)
from downstream_evaluation.segmentation.synthetic_dataset import (
    BRLoRAPosteriorMeanSegmentationDataset,
)
from downstream_evaluation.segmentation.training import (
    run_train_epoch,
    run_validation_epoch,
    segmentation_collate,
)
from downstream_evaluation.segmentation.transforms import (
    build_train_transform,
    build_validation_transform,
)
from src.config import (
    load_folders_config,
    resolve_path,
    save_folders_config,
)


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

DEFAULT_VALIDATION_MANIFEST = (
    PROJECT_ROOT
    / "downstream_evaluation"
    / "manifests"
    / "downstream_validation_manifest.csv"
)

DEFAULT_SYNTHETIC_MANIFEST = (
    PROJECT_ROOT
    / "downstream_evaluation"
    / "manifests"
    / "br_lora_library_design_10000"
    / "br_lora_library_design_10000.csv"
)

REGIMES = (
    "real_only",
    "real_plus_br_lora_mean",
    "real_plus_br_lora_posterior",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train the downstream BraTS tumor-segmentation model."
        )
    )

    parser.add_argument(
        "--regime",
        choices=REGIMES,
        required=True,
        help="Downstream training-data regime.",
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Tracked downstream experiment configuration YAML.",
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
        help=(
            "BraTS H5 root. Overrides h5_root in --folders-file."
        ),
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
        "--validation-manifest",
        type=Path,
        default=None,
        help=(
            "Frozen validation manifest. CLI overrides folders YAML; "
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

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Run output directory. If omitted, a non-overwriting path "
            "under outputs/downstream_segmentation/runs is generated."
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

    parser.add_argument(
        "--validate-only",
        action="store_true",
        help=(
            "Validate configuration, paths, dataset contracts, and "
            "reproducibility prerequisites without starting training "
            "or creating run outputs."
        ),
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
        loaded = yaml.safe_load(file)

    if not isinstance(loaded, dict):
        raise ValueError(
            "Configuration must contain a YAML mapping."
        )

    return loaded


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


def resolve_optional_repo_path(
    *,
    key: str,
    cli_value: Path | None,
    folders_config: dict[str, str],
    default: Path,
) -> Path:
    """
    Resolve CLI > folders YAML > tracked repository default.

    CLI overrides are persisted to the machine-specific folders mapping.
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
) -> dict[str, Path | None]:
    folders_config = load_folders_config(
        args.folders_file
    )

    h5_root = resolve_path(
        key="h5_root",
        cli_value=args.h5_root,
        config=folders_config,
        selector=None,
    )

    real_train_manifest = resolve_optional_repo_path(
        key="downstream_real_training_manifest",
        cli_value=args.real_train_manifest,
        folders_config=folders_config,
        default=DEFAULT_REAL_TRAIN_MANIFEST,
    )

    validation_manifest = resolve_optional_repo_path(
        key="downstream_validation_manifest",
        cli_value=args.validation_manifest,
        folders_config=folders_config,
        default=DEFAULT_VALIDATION_MANIFEST,
    )

    synthetic_manifest = None
    library_root = None

    if args.regime != "real_only":
        synthetic_manifest = resolve_optional_repo_path(
            key="downstream_synthetic_manifest",
            cli_value=args.synthetic_manifest,
            folders_config=folders_config,
            default=DEFAULT_SYNTHETIC_MANIFEST,
        )

        library_root = resolve_path(
            key="br_lora_library_root",
            cli_value=args.br_lora_library_root,
            config=folders_config,
            selector=None,
        )

    save_folders_config(
        args.folders_file,
        folders_config,
    )

    return {
        "h5_root": Path(h5_root),
        "real_train_manifest": Path(real_train_manifest),
        "validation_manifest": Path(validation_manifest),
        "synthetic_manifest": (
            None
            if synthetic_manifest is None
            else Path(synthetic_manifest)
        ),
        "library_root": (
            None
            if library_root is None
            else Path(library_root)
        ),
    }


def sha256_file(
    path: Path,
) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def git_commit() -> str | None:
    try:
        result = subprocess.run(
            [
                "git",
                "rev-parse",
                "HEAD",
            ],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (
        OSError,
        subprocess.CalledProcessError,
    ):
        return None

    return result.stdout.strip()


def package_version(
    name: str,
) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def default_output_dir(
    regime: str,
    seed: int,
) -> Path:
    partition = os.environ.get(
        "SLURM_JOB_PARTITION",
        "local",
    )

    job_id = os.environ.get(
        "SLURM_JOB_ID"
    )

    if job_id:
        run_id = f"job_{job_id}"
    else:
        timestamp = datetime.now(
            timezone.utc
        ).strftime("%Y%m%dT%H%M%SZ")
        run_id = f"run_{timestamp}"

    return (
        PROJECT_ROOT
        / "outputs"
        / "downstream_segmentation"
        / "runs"
        / regime
        / f"seed_{seed}"
        / f"{partition}_{run_id}"
    )


def prepare_output_directory(
    path: Path,
) -> Path:
    path = path.expanduser().resolve()

    if path.exists():
        if not path.is_dir():
            raise ValueError(
                f"Output path exists and is not a directory: {path}"
            )

        if any(path.iterdir()):
            raise RuntimeError(
                "Refusing to overwrite a non-empty output directory:\n"
                f"{path}"
            )

    path.mkdir(
        parents=True,
        exist_ok=True,
    )

    return path


def validate_file(
    path: Path,
    description: str,
) -> None:
    if not path.is_file():
        raise FileNotFoundError(
            f"{description} not found: {path}"
        )


def build_run_metadata(
    *,
    args: argparse.Namespace,
    config_path: Path,
    output_dir: Path,
    paths: dict[str, Path | None],
    seed: int,
    device: torch.device,
    reproducibility_cfg: dict,
) -> dict:
    manifest_hashes = {}

    for label in (
        "real_train_manifest",
        "validation_manifest",
        "synthetic_manifest",
    ):
        path = paths[label]

        if path is not None:
            manifest_hashes[label] = {
                "path": str(path),
                "sha256": sha256_file(path),
            }

    metadata = {
        "created_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "regime": args.regime,
        "seed": seed,
        "config": {
            "path": str(config_path),
            "sha256": sha256_file(config_path),
        },
        "manifests": manifest_hashes,
        "paths": {
            key: (
                None
                if value is None
                else str(value)
            )
            for key, value in paths.items()
        },
        "output_dir": str(output_dir),
        "git_commit": git_commit(),
        "runtime": {
            "python": sys.version,
            "torch": torch.__version__,
            "numpy": package_version("numpy"),
            "pandas": package_version("pandas"),
            "albumentations": package_version(
                "albumentations"
            ),
            "cuda_available": torch.cuda.is_available(),
            "cuda_build": torch.version.cuda,
            "device": str(device),
            "gpu_name": (
                torch.cuda.get_device_name(0)
                if device.type == "cuda"
                else None
            ),
            "slurm_job_id": os.environ.get(
                "SLURM_JOB_ID"
            ),
            "slurm_partition": os.environ.get(
                "SLURM_JOB_PARTITION"
            ),
            "host": os.environ.get(
                "HOSTNAME"
            ),
        },
        "reproducibility": {
            "deterministic_algorithms": bool(
                reproducibility_cfg[
                    "deterministic_algorithms"
                ]
            ),
            "cudnn_deterministic": bool(
                reproducibility_cfg[
                    "cudnn_deterministic"
                ]
            ),
            "cudnn_benchmark": bool(
                reproducibility_cfg[
                    "cudnn_benchmark"
                ]
            ),
            "cublas_workspace_config": os.environ.get(
                "CUBLAS_WORKSPACE_CONFIG"
            ),
            "separate_train_validation_generators": True,
            "train_generator_seed": seed,
            "validation_generator_seed": seed + 1,
        },
        "provenance": {
            "reference_repository": (
                "https://github.com/edaaydinea/"
                "Low-Grade-Glioma-Segmentation"
            ),
            "note": (
                "Downstream segmentation idea and vanilla U-Net "
                "structure adapted in part from the reference "
                "repository; current implementation was rewritten "
                "for the BraTS/UCSF-PDGM and BR-LoRA workflow."
            ),
        },
    }

    return metadata


def write_json(
    path: Path,
    payload: dict,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            payload,
            file,
            indent=2,
            sort_keys=True,
        )
        file.write("\n")


def write_history(
    path: Path,
    history: list[dict],
    *,
    posterior_sampling: bool,
) -> None:
    fieldnames = [
        "epoch",
    ]

    if posterior_sampling:
        fieldnames.append(
            "posterior_schedule_position"
        )

    fieldnames.extend(
        [
            "train_loss",
            "train_dice",
            "validation_loss",
            "validation_dice",
            "validation_iou",
            "validation_positive_dice",
            "validation_positive_iou",
        ]
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(history)


def main() -> None:
    args = parse_args()

    config_path = args.config.expanduser().resolve()
    config = load_config(config_path)

    if args.regime not in config["regimes"]:
        raise ValueError(
            f"Regime {args.regime!r} is not defined in {config_path}."
        )

    paths = resolve_paths(args)

    validate_file(
        paths["real_train_manifest"],
        "Real-training manifest",
    )

    validate_file(
        paths["validation_manifest"],
        "Validation manifest",
    )

    if args.regime != "real_only":
        validate_file(
            paths["synthetic_manifest"],
            "Synthetic manifest",
        )

    seed = int(
        config.get(
            "seed",
            42,
        )
    )

    data_cfg = config["data"]
    training_cfg = config["training"]
    reproducibility_cfg = config["reproducibility"]
    expected_cfg = config["expected_counts"]
    regime_cfg = config["regimes"][args.regime]

    image_channel = int(
        data_cfg.get(
            "image_channel",
            0,
        )
    )

    batch_size = int(
        data_cfg.get(
            "batch_size",
            26,
        )
    )

    num_workers = int(
        data_cfg.get(
            "num_workers",
            4,
        )
    )

    configured_pin_memory = data_cfg.get(
        "pin_memory",
        None,
    )

    if (
        configured_pin_memory is not None
        and not isinstance(
            configured_pin_memory,
            bool,
        )
    ):
        raise ValueError(
            "data.pin_memory must be true, false, or null."
        )

    epochs = int(
        training_cfg.get(
            "epochs",
            20,
        )
    )

    learning_rate = float(
        training_cfg.get(
            "learning_rate",
            0.001,
        )
    )

    threshold = float(
        training_cfg.get(
            "threshold",
            0.5,
        )
    )

    deterministic_algorithms = bool(
        reproducibility_cfg.get(
            "deterministic_algorithms",
            True,
        )
    )

    cudnn_deterministic = bool(
        reproducibility_cfg.get(
            "cudnn_deterministic",
            True,
        )
    )

    cudnn_benchmark = bool(
        reproducibility_cfg.get(
            "cudnn_benchmark",
            False,
        )
    )

    configured_cublas = str(
        reproducibility_cfg.get(
            "cublas_workspace_config",
            DEFAULT_CUBLAS_WORKSPACE_CONFIG,
        )
    )

    if configured_cublas != DEFAULT_CUBLAS_WORKSPACE_CONFIG:
        raise ValueError(
            "The hardened downstream configuration currently supports "
            "CUBLAS_WORKSPACE_CONFIG=:4096:8, matching the validated "
            "GPU reproducibility audit."
        )

    device = resolve_device(args.device)

    pin_memory = (
        device.type == "cuda"
        if configured_pin_memory is None
        else configured_pin_memory
    )

    if device.type == "cuda" and deterministic_algorithms:
        validate_cuda_reproducibility_environment()

    configure_reproducibility(
        seed,
        deterministic_algorithms=deterministic_algorithms,
        cudnn_deterministic=cudnn_deterministic,
        cudnn_benchmark=cudnn_benchmark,
    )

    output_dir = (
        default_output_dir(
            args.regime,
            seed,
        )
        if args.output_dir is None
        else args.output_dir
    )

    print("Downstream segmentation training")
    print("=" * 78)
    print("Regime:", args.regime)
    print("Device:", device)
    print("Seed:", seed)
    print("Output directory:", output_dir)
    print(
        "Strict deterministic algorithms:",
        deterministic_algorithms,
    )

    if device.type == "cuda":
        print(
            "CUBLAS_WORKSPACE_CONFIG:",
            os.environ.get(
                "CUBLAS_WORKSPACE_CONFIG"
            ),
        )
        print(
            "cuDNN deterministic:",
            torch.backends.cudnn.deterministic,
        )
        print(
            "cuDNN benchmark:",
            torch.backends.cudnn.benchmark,
        )

    real_train_transform = build_train_transform()
    validation_transform = build_validation_transform()

    if hasattr(
        real_train_transform,
        "set_random_seed",
    ):
        real_train_transform.set_random_seed(seed)

    real_train_dataset = DownstreamBraTSSegmentationDataset(
        manifest_path=paths[
            "real_train_manifest"
        ],
        h5_root=paths["h5_root"],
        image_channel=image_channel,
        transform=real_train_transform,
    )

    validation_dataset = DownstreamBraTSSegmentationDataset(
        manifest_path=paths[
            "validation_manifest"
        ],
        h5_root=paths["h5_root"],
        image_channel=image_channel,
        transform=validation_transform,
    )

    synthetic_dataset = None
    synthetic_mode = str(
        regime_cfg.get(
            "synthetic_mode",
            "none",
        )
    )

    if synthetic_mode == "none":
        train_dataset = real_train_dataset
        collate_fn = None

    elif synthetic_mode == "posterior_mean":
        synthetic_transform = build_train_transform()

        if hasattr(
            synthetic_transform,
            "set_random_seed",
        ):
            synthetic_transform.set_random_seed(seed)

        synthetic_dataset = (
            BRLoRAPosteriorMeanSegmentationDataset(
                manifest_path=paths[
                    "synthetic_manifest"
                ],
                library_root=paths[
                    "library_root"
                ],
                h5_root=paths["h5_root"],
                transform=synthetic_transform,
            )
        )

        train_dataset = ConcatDataset(
            [
                real_train_dataset,
                synthetic_dataset,
            ]
        )

        collate_fn = segmentation_collate

    elif synthetic_mode == "posterior_sample":
        synthetic_transform = build_train_transform()

        if hasattr(
            synthetic_transform,
            "set_random_seed",
        ):
            synthetic_transform.set_random_seed(seed)

        synthetic_dataset = (
            BRLoRAPosteriorSampleSegmentationDataset(
                manifest_path=paths[
                    "synthetic_manifest"
                ],
                library_root=paths[
                    "library_root"
                ],
                h5_root=paths["h5_root"],
                seed=seed,
                transform=synthetic_transform,
            )
        )

        expected_posterior_samples = int(
            regime_cfg.get(
                "posterior_samples_available",
                100,
            )
        )

        if (
            synthetic_dataset.POSTERIOR_SAMPLES
            != expected_posterior_samples
        ):
            raise RuntimeError(
                "Posterior-sample count does not match the "
                "configured data contract."
            )

        if epochs > synthetic_dataset.POSTERIOR_SAMPLES:
            raise ValueError(
                "Training epochs exceed the number of distinct "
                "posterior realizations available per synthetic case."
            )

        train_dataset = ConcatDataset(
            [
                real_train_dataset,
                synthetic_dataset,
            ]
        )

        collate_fn = segmentation_collate

    else:
        raise ValueError(
            f"Unsupported synthetic_mode: {synthetic_mode!r}"
        )

    expected_real = int(
        expected_cfg[
            "real_training_slices"
        ]
    )

    expected_validation = int(
        expected_cfg[
            "validation_slices"
        ]
    )

    if len(real_train_dataset) != expected_real:
        raise RuntimeError(
            "Unexpected real-training sample count: "
            f"expected {expected_real:,}, "
            f"observed {len(real_train_dataset):,}."
        )

    if len(validation_dataset) != expected_validation:
        raise RuntimeError(
            "Unexpected validation sample count: "
            f"expected {expected_validation:,}, "
            f"observed {len(validation_dataset):,}."
        )

    if synthetic_dataset is not None:
        expected_synthetic = int(
            expected_cfg[
                "synthetic_training_cases"
            ]
        )

        expected_combined = int(
            expected_cfg[
                "combined_training_slices"
            ]
        )

        if len(synthetic_dataset) != expected_synthetic:
            raise RuntimeError(
                "Unexpected synthetic sample count: "
                f"expected {expected_synthetic:,}, "
                f"observed {len(synthetic_dataset):,}."
            )

        if len(train_dataset) != expected_combined:
            raise RuntimeError(
                "Unexpected combined training sample count: "
                f"expected {expected_combined:,}, "
                f"observed {len(train_dataset):,}."
            )

    print(
        f"Real training slices: {len(real_train_dataset):,}"
    )

    if synthetic_dataset is not None:
        print(
            f"Synthetic training cases: {len(synthetic_dataset):,}"
        )

    print(
        f"Total training samples: {len(train_dataset):,}"
    )
    print(
        f"Validation slices: {len(validation_dataset):,}"
    )

    if args.validate_only:
        print()
        print("=" * 78)
        print("VALIDATION PASSED")
        print("No training was started.")
        print("No run output directory was created.")
        return

    output_dir = prepare_output_directory(
        output_dir
    )

    run_metadata = build_run_metadata(
        args=args,
        config_path=config_path,
        output_dir=output_dir,
        paths=paths,
        seed=seed,
        device=device,
        reproducibility_cfg=reproducibility_cfg,
    )

    write_json(
        output_dir / "run_metadata.json",
        run_metadata,
    )

    with (
        output_dir / "resolved_config.yaml"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        yaml.safe_dump(
            config,
            file,
            sort_keys=False,
        )

    train_generator = make_generator(seed)
    validation_generator = make_generator(
        seed + 1
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker,
        generator=train_generator,
        collate_fn=collate_fn,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker,
        generator=validation_generator,
    )

    model = VanillaUNet(
        in_channels=1,
        out_channels=1,
    ).to(device)

    criterion = BCEDiceLoss()

    optimizer = torch.optim.Adamax(
        model.parameters(),
        lr=learning_rate,
    )

    history = []
    best_positive_dice = float("-inf")
    best_checkpoint_path = (
        output_dir / "best_model.pt"
    )

    print("Epochs:", epochs)
    print("Batch size:", batch_size)
    print("Learning rate:", learning_rate)
    print("Threshold:", threshold)
    print()

    for epoch in range(1, epochs + 1):
        posterior_position = None

        if isinstance(
            synthetic_dataset,
            BRLoRAPosteriorSampleSegmentationDataset,
        ):
            posterior_position = epoch - 1
            synthetic_dataset.set_epoch(
                posterior_position
            )

            print(
                f"Epoch {epoch:02d}: posterior schedule "
                f"position {posterior_position}"
            )

        train_loss, train_dice = run_train_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            threshold=threshold,
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
            threshold=threshold,
        )

        row = {
            "epoch": epoch,
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

        if posterior_position is not None:
            row[
                "posterior_schedule_position"
            ] = posterior_position

        history.append(row)

        print(
            f"Epoch {epoch:02d}/{epochs:02d} | "
            f"train loss {train_loss:.6f} | "
            f"train Dice {train_dice:.6f} | "
            f"val loss {validation_loss:.6f} | "
            f"val Dice {validation_dice:.6f} | "
            f"val IoU {validation_iou:.6f} | "
            f"val+ Dice {validation_positive_dice:.6f} | "
            f"val+ IoU {validation_positive_iou:.6f}"
        )

        if validation_positive_dice > best_positive_dice:
            best_positive_dice = (
                validation_positive_dice
            )

            checkpoint = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "regime": args.regime,
                "seed": seed,
                "validation_dice": validation_dice,
                "validation_iou": validation_iou,
                "validation_positive_dice": (
                    validation_positive_dice
                ),
                "validation_positive_iou": (
                    validation_positive_iou
                ),
                "image_channel": image_channel,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "threshold": threshold,
                "epochs": epochs,
                "real_training_slices": len(
                    real_train_dataset
                ),
                "validation_slices": len(
                    validation_dataset
                ),
                "real_train_manifest": str(
                    paths["real_train_manifest"]
                ),
                "validation_manifest": str(
                    paths["validation_manifest"]
                ),
                "strict_deterministic_algorithms": (
                    deterministic_algorithms
                ),
                "cudnn_deterministic": (
                    cudnn_deterministic
                ),
                "cudnn_benchmark": cudnn_benchmark,
                "cublas_workspace_config": (
                    os.environ.get(
                        "CUBLAS_WORKSPACE_CONFIG"
                    )
                ),
                "git_commit": git_commit(),
            }

            if synthetic_dataset is not None:
                checkpoint.update(
                    {
                        "synthetic_manifest": str(
                            paths[
                                "synthetic_manifest"
                            ]
                        ),
                        "synthetic_training_cases": len(
                            synthetic_dataset
                        ),
                        "combined_training_slices": len(
                            train_dataset
                        ),
                    }
                )

            if isinstance(
                synthetic_dataset,
                BRLoRAPosteriorSampleSegmentationDataset,
            ):
                checkpoint.update(
                    {
                        "posterior_samples_available": (
                            synthetic_dataset.POSTERIOR_SAMPLES
                        ),
                        "distinct_posterior_samples_per_case": (
                            epochs
                        ),
                        "posterior_schedule_position": (
                            posterior_position
                        ),
                    }
                )

            best_checkpoint_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            torch.save(
                checkpoint,
                best_checkpoint_path,
            )

        write_history(
            output_dir / "training_history.csv",
            history,
            posterior_sampling=isinstance(
                synthetic_dataset,
                BRLoRAPosteriorSampleSegmentationDataset,
            ),
        )

    completed_metadata = dict(
        run_metadata
    )

    completed_metadata["completed_utc"] = (
        datetime.now(
            timezone.utc
        ).isoformat()
    )
    completed_metadata["best_validation_positive_dice"] = (
        best_positive_dice
    )
    completed_metadata["best_checkpoint"] = {
        "path": str(best_checkpoint_path),
        "sha256": sha256_file(
            best_checkpoint_path
        ),
    }

    write_json(
        output_dir / "run_metadata.json",
        completed_metadata,
    )

    print()
    print("=" * 78)
    print("Training complete.")
    print(
        "Best validation positive-slice Dice:",
        f"{best_positive_dice:.12f}",
    )
    print(
        "Best checkpoint:",
        best_checkpoint_path,
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
            "\nDOWNSTREAM TRAINING FAILED",
            file=sys.stderr,
        )
        print(
            exc,
            file=sys.stderr,
        )
        sys.exit(1)
