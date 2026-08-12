#!/usr/bin/env python3

"""
Audit Bayesian Regional LoRA (BR-LoRA) posterior sampling.

This script validates posterior-sampling behavior of a fitted BR-LoRA model.
It intentionally reuses the validated BR-LoRA inference pipeline and does not
implement any new inference logic.

The audit currently focuses on generating raw posterior realizations from a
fixed diffusion input. No uncertainty interpretation, reliability score, or
benchmarking metric is computed here.

Posterior variation is isolated by reusing the prepared diffusion input across
all realizations so that differences arise only from sampled BR-LoRA adapter
parameters.
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
import torch
import yaml

from src.diffusion import DiffusionSchedule
from src.inference import (
    discover_composition_candidates,
    load_fitted_br_lora,
    posterior_sample_inference,
    prepare_br_lora_batch,
    prepare_selected_pairs,
    select_clean_insertion_pairs,
)

AUDIT_NAME = "br_lora_posterior_sampling"


# Pair manifests belong to the future dataset-level evaluation workflow;
# this audit intentionally retains deterministic seeded pair selection.
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit Bayesian Regional LoRA posterior sampling using the "
            "validated BR-LoRA inference pipeline."
        )
    )

    parser.add_argument(
        "--baseline-config",
        type=Path,
        default=Path(
            "configs/baseline_patch_x0.yaml"
        ),
        help=(
            "Baseline configuration used to reconstruct the validated "
            "diffusion and data pipeline."
        ),
    )

    parser.add_argument(
        "--br-lora-config",
        type=Path,
        default=Path(
            "configs/br_lora.yaml"
        ),
        help=(
            "Bayesian Regional LoRA configuration."
        ),
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help=(
            "Path to a fitted BR-LoRA checkpoint."
        ),
    )

    parser.add_argument(
        "--h5-root",
        type=Path,
        required=True,
        help=(
            "Root directory containing the reconstructed BraTS H5 files."
        ),
    )

    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help=(
            "Dataset manifest used for candidate discovery."
        ),
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
        "--num-pairs",
        type=int,
        default=4,
        help=(
            "Number of localized synthesis pairs to audit."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Optional audit seed override. When omitted, the baseline "
            "configuration seed is used."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "outputs/br_lora_posterior"
        ),
        help=(
            "Directory for serialized audit outputs."
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
        help=(
            "Execution device."
        ),
    )

    return parser.parse_args()


def load_config(
    path: Path,
    *,
    name: str,
) -> dict:
    """Load one YAML configuration mapping."""

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
    """Resolve an explicit or automatic Torch device."""

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
    """Seed Python, NumPy, and Torch."""

    random.seed(
        seed
    )

    np.random.seed(
        seed
    )

    torch.manual_seed(
        seed
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
    """Construct the fixed BR-LoRA posterior-audit inference state."""

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
        baseline_inference_cfg,
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

    if (
        isinstance(
            args.num_pairs,
            bool,
        )
        or not isinstance(
            args.num_pairs,
            int,
        )
        or args.num_pairs <= 0
    ):
        raise ValueError(
            "num_pairs must be a positive integer."
        )

    resample_diffusion_noise = bool(
        br_inference_cfg.get(
            "resample_diffusion_noise",
            False,
        )
    )

    if resample_diffusion_noise:
        raise ValueError(
            "The BR-LoRA posterior audit requires "
            "inference.resample_diffusion_noise=false so posterior variation "
            "is isolated from diffusion-noise variation."
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
        max_pairs=args.num_pairs,
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

    if len(
        pairs
    ) != args.num_pairs:
        raise RuntimeError(
            "Pair selection did not return the requested number of audit "
            "pairs.\n"
            f"Requested: {args.num_pairs}\n"
            f"Selected:  {len(pairs)}"
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

    expected_batch_size = len(
        pairs
    )

    if (
        prepared.target.shape[
            0
        ]
        != expected_batch_size
    ):
        raise RuntimeError(
            "Prepared BR-LoRA batch size does not match the selected pair "
            "count."
        )

    if not torch.equal(
        prepared.target.detach().cpu(),
        retained[
            "base_images"
        ].detach().cpu(),
    ):
        raise RuntimeError(
            "Prepared target tensors do not exactly match retained base "
            "images."
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
        "BR-LoRA POSTERIOR AUDIT CONSTRUCTION"
    )
    print(
        "=" * 78
    )

    print()
    print(
        "Audit"
    )
    print(
        "-" * 78
    )
    print(
        "Audit name               :",
        AUDIT_NAME,
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
        "Posterior samples        :",
        posterior_samples,
    )
    print(
        "Requested pairs          :",
        args.num_pairs,
    )
    print(
        "Selected pairs           :",
        len(
            pairs
        ),
    )

    print()
    print(
        "Candidate discovery"
    )
    print(
        "-" * 78
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

    print()
    print(
        "BR-LoRA"
    )
    print(
        "-" * 78
    )
    print(
        "Variational modules      :",
        len(
            loaded.variational_module_names
        ),
    )
    print(
        "Variational parameters   :",
        f"{loaded.variational_parameter_count:,}",
    )

    print()
    print(
        "Prepared inference state"
    )
    print(
        "-" * 78
    )
    print(
        "Batch size               :",
        prepared.target.shape[
            0
        ],
    )
    print(
        "Tensor shape             :",
        tuple(
            prepared.target.shape
        ),
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
    print(
        "Diffusion noise finite   :",
        bool(
            torch.isfinite(
                prepared.diffusion_noise
            ).all().item()
        ),
    )
    print(
        "x_t finite               :",
        bool(
            torch.isfinite(
                prepared.x_t
            ).all().item()
        ),
    )

    print()
    print(
        "Output"
    )
    print(
        "-" * 78
    )
    print(
        "Future output directory  :",
        output_dir,
    )

    print()

    print(
        "=" * 78
    )

    print(
        "GENERATING POSTERIOR REALIZATIONS"
    )

    print(
        "=" * 78
    )

    result = posterior_sample_inference(
        model=model,
        prepared=prepared,
        posterior_samples=posterior_samples,
    )

    if (
        result.posterior_samples
        != posterior_samples
    ):
        raise RuntimeError(
            "Returned posterior sample count does not match "
            "the requested value."
        )

    if (
        result.prediction_samples.shape[0]
        != posterior_samples
    ):
        raise RuntimeError(
            "Prediction sample stack has an unexpected "
            "leading dimension."
        )

    if (
        result.composite_samples.shape[0]
        != posterior_samples
    ):
        raise RuntimeError(
            "Composite sample stack has an unexpected "
            "leading dimension."
        )

    if not torch.isfinite(
        result.prediction_samples
    ).all():
        raise RuntimeError(
            "Prediction samples contain non-finite values."
        )

    if not torch.isfinite(
        result.composite_samples
    ).all():
        raise RuntimeError(
            "Composite samples contain non-finite values."
        )

    for tensor, name in (
        (
            result.prediction_mean,
            "prediction_mean",
        ),
        (
            result.prediction_variance,
            "prediction_variance",
        ),
        (
            result.prediction_std,
            "prediction_std",
        ),
        (
            result.composite_mean,
            "composite_mean",
        ),
        (
            result.composite_variance,
            "composite_variance",
        ),
        (
            result.composite_std,
            "composite_std",
        ),
    ):
        if not torch.isfinite(
            tensor
        ).all():
            raise RuntimeError(
                f"{name} contains non-finite values."
            )

    if (
        result.prediction_variance
        < 0
    ).any():
        raise RuntimeError(
            "Prediction variance contains negative values."
        )

    if (
        result.composite_variance
        < 0
    ).any():
        raise RuntimeError(
            "Composite variance contains negative values."
        )

    tensor_path = (
        output_dir
        / "posterior_realizations.pt"
    )

    payload = {
        "audit_name": AUDIT_NAME,
        "checkpoint": str(
            checkpoint_path
        ),
        "seed": seed,
        "posterior_samples": posterior_samples,
        "selected_pairs": pairs,

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

        "timestep": cpu_tensor(
            prepared.timestep
        ),

        "diffusion_noise": cpu_tensor(
            prepared.diffusion_noise
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

    print()

    print(
        "Posterior realizations   :",
        posterior_samples,
    )

    print(
        "Prediction stack shape   :",
        tuple(
            result.prediction_samples.shape
        ),
    )

    print(
        "Composite stack shape    :",
        tuple(
            result.composite_samples.shape
        ),
    )

    print(
        "Serialized audit         :",
        tensor_path,
    )

    print()

    print(
        "=" * 78
    )

    print(
        "POSTERIOR SAMPLING AUDIT: PASS"
    )

    print(
        "Raw BR-LoRA posterior realizations were generated "
        "successfully from a fixed diffusion input."
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
            "\nBR-LoRA POSTERIOR AUDIT FAILED",
            file=sys.stderr,
        )

        print(
            exc,
            file=sys.stderr,
        )

        sys.exit(
            1
        )