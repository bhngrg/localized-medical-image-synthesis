#!/usr/bin/env python3

"""
Run Bayesian Regional LoRA (BR-LoRA) localized synthesis.

Two explicit inference modes are supported:

posterior_mean
    Use the fitted posterior means of all variational LoRA factors.

posterior_samples
    Draw independent BR-LoRA posterior realizations while holding the prepared
    diffusion input fixed. Raw sample stacks and direct Monte Carlo
    mean/variance/std summaries are saved without assigning any reliability
    interpretation.

Pair discovery and selection intentionally reuse the validated baseline
regional-composition utilities.
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

from src.data import load_h5_full
from src.diffusion import DiffusionSchedule
from src.inference import (
    discover_composition_candidates,
    load_fitted_br_lora,
    posterior_mean_inference,
    posterior_sample_inference,
    prepare_br_lora_batch,
    select_clean_insertion_pairs,
)


INFERENCE_MODES = (
    "posterior_mean",
    "posterior_samples",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run BR-LoRA posterior-mean or posterior-sampled localized "
            "medical image synthesis."
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
        "--br-lora-config",
        type=Path,
        default=Path(
            "configs/br_lora.yaml"
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
        "--mode",
        choices=INFERENCE_MODES,
        default="posterior_mean",
    )

    parser.add_argument(
        "--posterior-samples",
        type=int,
        default=None,
        help=(
            "Optional override for inference.posterior_samples in the "
            "BR-LoRA configuration."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Optional inference seed override. When omitted, the baseline "
            "configuration seed is used."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "outputs/br_lora"
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
    path = path.expanduser().resolve()

    if not path.is_file():
        raise FileNotFoundError(
            f"{name} not found:\n{path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        config = yaml.safe_load(
            file
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
                "CUDA was requested but is unavailable."
            )

        return torch.device(
            "cuda"
        )

    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError(
                "MPS was requested but is unavailable."
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


def prepare_selected_pairs(
    *,
    selected_pairs: list[dict],
    image_channel: int,
) -> tuple[
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
]:
    """
    Convert selected base/donor pairs into the validated model-input contract.

    Returns both the dataset-style batch used by ``prepare_br_lora_batch`` and
    the stacked pair tensors retained for output/visualization.
    """

    if not selected_pairs:
        raise ValueError(
            "selected_pairs is empty."
        )

    base_images = []
    transferred_masks = []
    donor_patches = []
    donor_conditions = []

    for pair in selected_pairs:
        (
            base_image,
            _,
            _,
        ) = load_h5_full(
            pair[
                "base_path"
            ],
            image_channel=image_channel,
        )

        (
            donor_image,
            donor_mask,
            donor_condition,
        ) = load_h5_full(
            pair[
                "mask_path"
            ],
            image_channel=image_channel,
        )

        donor_patch = (
            donor_image
            * donor_mask
        )

        base_images.append(
            base_image
        )

        transferred_masks.append(
            donor_mask
        )

        donor_patches.append(
            donor_patch
        )

        donor_conditions.append(
            donor_condition
        )

    base_images = torch.stack(
        base_images,
        dim=0,
    )

    transferred_masks = torch.stack(
        transferred_masks,
        dim=0,
    )

    donor_patches = torch.stack(
        donor_patches,
        dim=0,
    )

    donor_conditions = torch.stack(
        donor_conditions,
        dim=0,
    )

    known = (
        base_images
        * (
            1.0
            - transferred_masks
        )
    )

    batch = {
        "x0": base_images,
        "known": known,
        "mask": transferred_masks,
        "donor_patch": donor_patches,
        "cond": donor_conditions,
    }

    retained = {
        "base_images": base_images,
        "transferred_masks": transferred_masks,
        "known": known,
        "donor_patches": donor_patches,
        "donor_conditions": donor_conditions,
    }

    return (
        batch,
        retained,
    )


def save_posterior_mean_figure(
    *,
    path: Path,
    retained: dict[str, torch.Tensor],
    prediction: torch.Tensor,
    composite: torch.Tensor,
) -> None:
    """Save the baseline-style six-column posterior-mean visualization."""

    n = retained[
        "base_images"
    ].shape[
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
        "BR-LoRA posterior mean",
        "Synthetic composite",
    ]

    for i in range(
        n
    ):
        images = [
            retained[
                "base_images"
            ][
                i,
                0,
            ].detach().cpu().numpy(),

            retained[
                "transferred_masks"
            ][
                i,
                0,
            ].detach().cpu().numpy(),

            retained[
                "known"
            ][
                i,
                0,
            ].detach().cpu().numpy(),

            retained[
                "donor_patches"
            ][
                i,
                0,
            ].detach().cpu().numpy(),

            prediction[
                i,
                0,
            ].detach().cpu().numpy(),

            composite[
                i,
                0,
            ].detach().cpu().numpy(),
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


def save_posterior_samples_figure(
    *,
    path: Path,
    retained: dict[str, torch.Tensor],
    prediction_mean: torch.Tensor,
    composite_mean: torch.Tensor,
    composite_std: torch.Tensor,
) -> None:
    """
    Save direct posterior-sampling summaries.

    The final column is a raw posterior standard-deviation map. It is not
    labeled or interpreted as a reliability score.
    """

    n = retained[
        "base_images"
    ].shape[
        0
    ]

    fig, axes = plt.subplots(
        n,
        7,
        figsize=(
            20,
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
        "Posterior-sample mean",
        "Composite mean",
        "Composite posterior std",
    ]

    for i in range(
        n
    ):
        images = [
            retained[
                "base_images"
            ][
                i,
                0,
            ].detach().cpu().numpy(),

            retained[
                "transferred_masks"
            ][
                i,
                0,
            ].detach().cpu().numpy(),

            retained[
                "known"
            ][
                i,
                0,
            ].detach().cpu().numpy(),

            retained[
                "donor_patches"
            ][
                i,
                0,
            ].detach().cpu().numpy(),

            prediction_mean[
                i,
                0,
            ].detach().cpu().numpy(),

            composite_mean[
                i,
                0,
            ].detach().cpu().numpy(),

            composite_std[
                i,
                0,
            ].detach().cpu().numpy(),
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


def cpu_tensor(
    tensor: torch.Tensor,
) -> torch.Tensor:
    """Detach one tensor and move it to CPU for portable serialization."""

    return (
        tensor
        .detach()
        .cpu()
    )


def main() -> None:
    args = parse_args()

    baseline_config = load_config(
        args.baseline_config,
        name="Baseline configuration",
    )

    br_lora_config = load_config(
        args.br_lora_config,
        name="BR-LoRA configuration",
    )

    data_cfg = baseline_config[
        "data"
    ]

    diffusion_cfg = baseline_config[
        "diffusion"
    ]

    baseline_inference_cfg = baseline_config.get(
        "inference",
        {},
    )

    helper_cfg = baseline_config.get(
        "inference_helpers",
        {},
    )

    br_inference_cfg = br_lora_config.get(
        "inference",
        {},
    )

    if not isinstance(
        br_inference_cfg,
        dict,
    ):
        raise ValueError(
            "BR-LoRA inference configuration must be a mapping."
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

    posterior_samples = (
        int(
            br_inference_cfg.get(
                "posterior_samples",
                50,
            )
        )
        if args.posterior_samples is None
        else int(
            args.posterior_samples
        )
    )

    if posterior_samples <= 0:
        raise ValueError(
            "posterior_samples must be positive."
        )

    resample_diffusion_noise = bool(
        br_inference_cfg.get(
            "resample_diffusion_noise",
            False,
        )
    )

    if (
        args.mode
        == "posterior_samples"
        and resample_diffusion_noise
    ):
        raise ValueError(
            "This validated BR-LoRA posterior-sampling CLI currently "
            "requires inference.resample_diffusion_noise=false so posterior "
            "variation is isolated from diffusion-noise variation."
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
            f"BR-LoRA checkpoint not found:\n{checkpoint_path}"
        )

    if not h5_root.is_dir():
        raise FileNotFoundError(
            f"H5 root not found:\n{h5_root}"
        )

    if not manifest.is_file():
        raise FileNotFoundError(
            f"Manifest not found:\n{manifest}"
        )

    loaded = load_fitted_br_lora(
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
            baseline_inference_cfg.get(
                "max_base_candidates",
                300,
            )
        ),
        max_mask_candidates=int(
            baseline_inference_cfg.get(
                "max_mask_candidates",
                500,
            )
        ),
        max_pairs=int(
            baseline_inference_cfg.get(
                "max_pairs",
                4,
            )
        ),
        min_overlap=float(
            baseline_inference_cfg.get(
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

    prepared = prepare_br_lora_batch(
        batch,
        schedule=schedule,
        device=device,
        timestep_fraction=float(
            baseline_inference_cfg.get(
                "timestep_fraction",
                0.75,
            )
        ),
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print(
        "=" * 78
    )
    print(
        "BR-LoRA LOCALIZED SYNTHESIS"
    )
    print(
        "=" * 78
    )

    print(
        "Mode                     :",
        args.mode,
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
        "Variational parameters   :",
        f"{loaded.variational_parameter_count:,}",
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
        "Fixed diffusion noise    :",
        True,
    )

    if args.mode == "posterior_mean":

        result = posterior_mean_inference(
            model=model,
            prepared=prepared,
        )

        tensor_path = (
            output_dir
            / "posterior_mean.pt"
        )

        figure_path = (
            output_dir
            / "posterior_mean.png"
        )

        payload = {
            "mode": args.mode,
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
        }

        torch.save(
            payload,
            tensor_path,
        )

        save_posterior_mean_figure(
            path=figure_path,
            retained=retained,
            prediction=result.prediction,
            composite=result.composite,
        )

        print(
            "Posterior samples        :",
            "not applicable",
        )

    else:

        result = posterior_sample_inference(
            model=model,
            prepared=prepared,
            posterior_samples=posterior_samples,
        )

        tensor_path = (
            output_dir
            / "posterior_samples.pt"
        )

        figure_path = (
            output_dir
            / "posterior_samples.png"
        )

        payload = {
            "mode": args.mode,
            "checkpoint": str(
                checkpoint_path
            ),
            "seed": seed,
            "posterior_samples": (
                posterior_samples
            ),
            "resample_diffusion_noise": (
                False
            ),
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
            "prediction_samples": cpu_tensor(
                result.prediction_samples
            ),
            "composite_samples": cpu_tensor(
                result.composite_samples
            ),
            "prediction_mean": cpu_tensor(
                result.prediction_mean
            ),
            "prediction_variance": cpu_tensor(
                result.prediction_variance
            ),
            "prediction_std": cpu_tensor(
                result.prediction_std
            ),
            "composite_mean": cpu_tensor(
                result.composite_mean
            ),
            "composite_variance": cpu_tensor(
                result.composite_variance
            ),
            "composite_std": cpu_tensor(
                result.composite_std
            ),
        }

        torch.save(
            payload,
            tensor_path,
        )

        save_posterior_samples_figure(
            path=figure_path,
            retained=retained,
            prediction_mean=(
                result.prediction_mean
            ),
            composite_mean=(
                result.composite_mean
            ),
            composite_std=(
                result.composite_std
            ),
        )

        print(
            "Posterior samples        :",
            posterior_samples,
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
        FileNotFoundError,
        KeyError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        print(
            "\nBR-LoRA SYNTHESIS FAILED",
            file=sys.stderr,
        )

        print(
            exc,
            file=sys.stderr,
        )

        sys.exit(
            1
        )
