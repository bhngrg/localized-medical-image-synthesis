#!/usr/bin/env python3

"""
Run localized synthesis with a fitted deterministic adaptation comparator.

The fitted adaptation method and its construction parameters are recovered
directly from checkpoint metadata. Supported methods are Regional LoRA, DoRA,
LoKr, and BitFit.

Pair discovery and selection reuse the validated baseline regional-composition
utilities. Diffusion preparation is identical to the validated BR-LoRA
preparation when supplied the same batch, timestep fraction, and noise tensor.
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
import matplotlib.pyplot as plt
import torch
import yaml

from src.diffusion import DiffusionSchedule
from src.inference import (
    adaptation_inference,
    discover_composition_candidates,
    load_fitted_adaptation,
    prepare_adaptation_batch,
    prepare_selected_pairs,
    select_clean_insertion_pairs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run localized synthesis using a fitted deterministic "
            "adaptation comparator."
        )
    )

    parser.add_argument(
        "--baseline-config",
        type=Path,
        default=Path(
            "configs/baseline_patch_x0.yaml"
        ),
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
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
        "--output-dir",
        type=Path,
        default=Path(
            "outputs/adaptation"
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Optional inference seed override. When omitted, uses the "
            "baseline configuration seed."
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
    *,
    name: str,
) -> dict:
    resolved = (
        path
        .expanduser()
        .resolve()
    )

    if not resolved.is_file():
        raise FileNotFoundError(
            f"{name} not found:\n{resolved}"
        )

    with resolved.open(
        "r",
        encoding="utf-8",
    ) as handle:
        config = yaml.safe_load(
            handle
        )

    if not isinstance(
        config,
        dict,
    ):
        raise ValueError(
            f"{name} must contain a YAML mapping."
        )

    return config


def resolve_device(
    requested: str,
) -> torch.device:
    if requested == "cpu":
        return torch.device(
            "cpu"
        )

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


def cpu_tensor(
    value: torch.Tensor,
) -> torch.Tensor:
    return (
        value
        .detach()
        .cpu()
    )


def save_adaptation_figure(
    *,
    path: Path,
    retained: dict[str, torch.Tensor],
    prediction: torch.Tensor,
    composite: torch.Tensor,
    method: str,
) -> None:
    """Save the baseline-style six-column deterministic synthesis figure."""

    base_images = retained[
        "base_images"
    ]

    masks = retained[
        "transferred_masks"
    ]

    known = retained[
        "known"
    ]

    donor_patches = retained[
        "donor_patches"
    ]

    prediction_cpu = cpu_tensor(
        prediction
    )

    composite_cpu = cpu_tensor(
        composite
    )

    n = base_images.shape[
        0
    ]

    fig, axes = plt.subplots(
        n,
        6,
        figsize=(
            17,
            3 * n,
        ),
    )

    if n == 1:
        axes = axes[
            None,
            :,
        ]

    titles = [
        "Tumor-free base MRI",
        "Transferred mask",
        "Known input",
        "Donor tumor patch",
        f"{method} prediction",
        "Synthetic composite",
    ]

    for i in range(
        n
    ):
        images = [
            base_images[
                i,
                0,
            ].detach().cpu().numpy(),
            masks[
                i,
                0,
            ].detach().cpu().numpy(),
            known[
                i,
                0,
            ].detach().cpu().numpy(),
            donor_patches[
                i,
                0,
            ].detach().cpu().numpy(),
            prediction_cpu[
                i,
                0,
            ].numpy(),
            composite_cpu[
                i,
                0,
            ].numpy(),
        ]

        for j, image in enumerate(
            images
        ):
            axes[
                i,
                j,
            ].imshow(
                image,
                cmap="gray",
            )

            axes[
                i,
                j,
            ].set_title(
                titles[
                    j
                ]
            )

            axes[
                i,
                j,
            ].axis(
                "off"
            )

    plt.tight_layout()

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.savefig(
        path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


def main() -> None:
    args = parse_args()

    baseline_config = load_config(
        args.baseline_config,
        name="Baseline configuration",
    )

    data_cfg = baseline_config[
        "data"
    ]

    diffusion_cfg = baseline_config[
        "diffusion"
    ]

    inference_cfg = baseline_config.get(
        "inference",
        {},
    )

    helper_cfg = baseline_config.get(
        "inference_helpers",
        {},
    )

    if not isinstance(
        data_cfg,
        dict,
    ):
        raise ValueError(
            "Baseline data configuration must be a mapping."
        )

    if not isinstance(
        diffusion_cfg,
        dict,
    ):
        raise ValueError(
            "Baseline diffusion configuration must be a mapping."
        )

    if not isinstance(
        inference_cfg,
        dict,
    ):
        raise ValueError(
            "Baseline inference configuration must be a mapping."
        )

    if not isinstance(
        helper_cfg,
        dict,
    ):
        raise ValueError(
            "Baseline inference_helpers configuration must be a mapping."
        )

    seed = (
        int(
            baseline_config.get(
                "seed",
                42,
            )
        )
        if args.seed is None
        else int(
            args.seed
        )
    )

    device = resolve_device(
        args.device
    )

    set_seed(
        seed
    )

    checkpoint_path = (
        args.checkpoint
        .expanduser()
        .resolve()
    )

    h5_root = (
        args.h5_root
        .expanduser()
        .resolve()
    )

    manifest = (
        args.manifest
        .expanduser()
        .resolve()
    )

    output_dir = (
        args.output_dir
        .expanduser()
        .resolve()
    )

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Adaptation checkpoint not found:\n{checkpoint_path}"
        )

    if not h5_root.is_dir():
        raise FileNotFoundError(
            f"H5 root not found:\n{h5_root}"
        )

    if not manifest.is_file():
        raise FileNotFoundError(
            f"Manifest not found:\n{manifest}"
        )

    loaded = load_fitted_adaptation(
        checkpoint_path,
        device=device,
    )

    model = loaded.model

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

    tumor_free_files, tumor_mask_files = (
        discover_composition_candidates(
            h5_root=h5_root,
            manifest_path=manifest,
            min_tumor_pixels=int(
                data_cfg.get(
                    "min_tumor_pixels",
                    300,
                )
            ),
        )
    )

    pairs = select_clean_insertion_pairs(
        tumor_free_files=tumor_free_files,
        tumor_mask_files=tumor_mask_files,
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
        max_base_candidates=int(
            inference_cfg.get(
                "max_base_candidates",
                300,
            )
        ),
        max_mask_candidates=int(
            inference_cfg.get(
                "max_mask_candidates",
                500,
            )
        ),
        max_pairs=int(
            inference_cfg.get(
                "max_pairs",
                4,
            )
        ),
        min_overlap=float(
            inference_cfg.get(
                "min_overlap",
                0.80,
            )
        ),
        margin=int(
            helper_cfg.get(
                "mask_margin",
                10,
            )
        ),
        brain_threshold=float(
            helper_cfg.get(
                "brain_threshold",
                0.05,
            )
        ),
        seed=seed,
    )

    batch, retained = prepare_selected_pairs(
        selected_pairs=pairs,
        image_channel=int(
            data_cfg.get(
                "image_channel",
                0,
            )
        ),
    )

    prepared = prepare_adaptation_batch(
        batch,
        schedule=schedule,
        device=device,
        timestep_fraction=float(
            inference_cfg.get(
                "timestep_fraction",
                0.75,
            )
        ),
    )

    result = adaptation_inference(
        model=model,
        prepared=prepared,
    )

    method = loaded.adaptation_method

    method_output_dir = (
        output_dir
        / method
    )

    method_output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    tensor_path = (
        method_output_dir
        / "synthesis.pt"
    )

    figure_path = (
        method_output_dir
        / "synthesis.png"
    )

    payload = {
        "adaptation_method": method,
        "adaptation_config": dict(
            loaded.checkpoint[
                "adaptation_config"
            ]
        ),
        "model_config": dict(
            loaded.checkpoint[
                "model_config"
            ]
        ),
        "checkpoint": str(
            checkpoint_path
        ),
        "seed": seed,
        "selected_pairs": pairs,
        "timestep": cpu_tensor(
            prepared.timestep
        ),
        "diffusion_noise": cpu_tensor(
            prepared.diffusion_noise
        ),
        "base_images": cpu_tensor(
            retained[
                "base_images"
            ]
        ),
        "transferred_masks": cpu_tensor(
            retained[
                "transferred_masks"
            ]
        ),
        "known": cpu_tensor(
            retained[
                "known"
            ]
        ),
        "donor_patches": cpu_tensor(
            retained[
                "donor_patches"
            ]
        ),
        "donor_conditions": cpu_tensor(
            retained[
                "donor_conditions"
            ]
        ),
        "x_t": cpu_tensor(
            prepared.x_t
        ),
        "prediction": cpu_tensor(
            result.prediction
        ),
        "composite": cpu_tensor(
            result.composite
        ),
        "trainable_parameter_count": (
            loaded.trainable_parameter_count
        ),
        "trainable_tensor_count": (
            loaded.trainable_tensor_count
        ),
    }

    torch.save(
        payload,
        tensor_path,
    )

    save_adaptation_figure(
        path=figure_path,
        retained=retained,
        prediction=result.prediction,
        composite=result.composite,
        method=method,
    )

    print()
    print(
        "=" * 78
    )
    print(
        "DETERMINISTIC ADAPTATION LOCALIZED SYNTHESIS"
    )
    print(
        "=" * 78
    )

    print(
        "Method                   :",
        method,
    )

    print(
        "Checkpoint               :",
        checkpoint_path,
    )

    print(
        "Device                   :",
        device,
    )

    print(
        "Seed                     :",
        seed,
    )

    print(
        "Tumor-free candidates    :",
        len(
            tumor_free_files
        ),
    )

    print(
        "Donor-mask candidates    :",
        len(
            tumor_mask_files
        ),
    )

    print(
        "Selected pairs           :",
        len(
            pairs
        ),
    )

    print(
        "Adapted modules          :",
        len(
            loaded.adapted_module_names
        ),
    )

    print(
        "Trainable tensors        :",
        loaded.trainable_tensor_count,
    )

    print(
        "Trainable parameters     :",
        f"{loaded.trainable_parameter_count:,}",
    )

    print(
        "Timestep                 :",
        int(
            prepared.timestep[
                0
            ].item()
        ),
    )

    print(
        "Tensor output            :",
        tensor_path,
    )

    print(
        "Figure output            :",
        figure_path,
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
        TypeError,
    ) as error:
        raise SystemExit(
            str(
                error
            )
        ) from error
