#!/usr/bin/env python3

"""
Run baseline patch-conditioned x0 inference and regional composition.

This script loads a trained checkpoint, selects tumor-free base / donor-mask
pairs using the registered manifest, synthesizes localized insertions, and saves
a visualization matching the reference notebook's Cell 23 layout.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Keep current macOS environment import order during development.
import numpy as np
import matplotlib.pyplot as plt
import torch
import yaml

from src.diffusion import DiffusionSchedule
from src.inference import (
    discover_composition_candidates,
    select_clean_insertion_pairs,
    synthesize_insertion_pairs,
)
from src.models import AppearanceX0UNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run baseline patch-conditioned x0 tumor-free insertion."
        )
    )

    parser.add_argument(
        "--config",
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
        "--output",
        type=Path,
        default=Path(
            "outputs/patch_x0_tumor_free_insertion_test.png"
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
) -> dict:
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
            "Configuration must contain a YAML mapping."
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


def main() -> None:
    args = parse_args()
    config = load_config(
        args.config
    )

    data_cfg = config[
        "data"
    ]

    model_cfg = config[
        "model"
    ]

    diffusion_cfg = config[
        "diffusion"
    ]

    inference_cfg = config.get(
        "inference",
        {},
    )

    helper_cfg = config.get(
        "inference_helpers",
        {},
    )

    device = resolve_device(
        args.device
    )

    checkpoint = torch.load(
        args.checkpoint,
        map_location=device,
        weights_only=False,
    )

    model = AppearanceX0UNet(
        in_ch=int(
            model_cfg.get(
                "in_channels",
                4,
            )
        ),
        out_ch=int(
            model_cfg.get(
                "out_channels",
                1,
            )
        ),
        base=int(
            model_cfg.get(
                "base_channels",
                32,
            )
        ),
        time_dim=int(
            model_cfg.get(
                "time_dim",
                128,
            )
        ),
        cond_dim=int(
            model_cfg.get(
                "cond_dim",
                4,
            )
        ),
    ).to(
        device
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model.eval()

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
                1e-4,
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
            h5_root=args.h5_root,
            manifest_path=args.manifest,
            min_tumor_pixels=int(
                data_cfg.get(
                    "min_tumor_pixels",
                    300,
                )
            ),
        )
    )

    print(
        "Tumor-free base slices:",
        len(
            tumor_free_files
        ),
    )

    print(
        "Tumor mask donor slices:",
        len(
            tumor_mask_files
        ),
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
        seed=int(
            config.get(
                "seed",
                42,
            )
        ),
    )

    print(
        "Selected pairs:",
        len(
            pairs
        ),
    )

    for index, pair in enumerate(
        pairs,
        start=1,
    ):
        print(
            f"\nPair {index}"
        )
        print(
            "Base:",
            pair[
                "base_path"
            ],
        )
        print(
            "Mask:",
            pair[
                "mask_path"
            ],
        )
        print(
            "Overlap:",
            pair[
                "overlap_frac"
            ],
        )
        print(
            "Pixels:",
            pair[
                "mask_pixels"
            ],
        )

    result = synthesize_insertion_pairs(
        model=model,
        selected_pairs=pairs,
        schedule=schedule,
        device=device,
        image_channel=int(
            data_cfg.get(
                "image_channel",
                0,
            )
        ),
        timestep_fraction=float(
            inference_cfg.get(
                "timestep_fraction",
                0.75,
            )
        ),
    )

    n = result[
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
        "Patch-cond x0",
        "Synthetic composite",
    ]

    for i in range(
        n
    ):
        images = [
            result[
                "base_images"
            ][
                i,
                0,
            ].detach().cpu().numpy(),

            result[
                "transferred_masks"
            ][
                i,
                0,
            ].detach().cpu().numpy(),

            result[
                "known"
            ][
                i,
                0,
            ].detach().cpu().numpy(),

            result[
                "donor_patches"
            ][
                i,
                0,
            ].detach().cpu().numpy(),

            result[
                "pred_x0"
            ][
                i,
                0,
            ].detach().cpu().numpy(),

            result[
                "synthetic_composite"
            ][
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

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.savefig(
        args.output,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    print(
        f"\nSaved {args.output}"
    )


if __name__ == "__main__":
    try:
        main()

    except (
        RuntimeError,
        ValueError,
        FileNotFoundError,
        KeyError,
    ) as exc:
        print(
            "\nINFERENCE FAILED",
            file=sys.stderr,
        )

        print(
            exc,
            file=sys.stderr,
        )

        sys.exit(
            1
        )
