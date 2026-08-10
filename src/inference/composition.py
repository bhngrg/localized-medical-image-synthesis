"""
Tumor-free regional composition utilities extracted from notebook Cells 21–23.

The notebook discovers tumor-free and donor-mask slices by scanning every H5
file. This modular implementation uses the already-generated manifest.csv,
which preserves the same slice-selection criteria without repeating that scan.
"""

from __future__ import annotations

from pathlib import Path
import random

import pandas as pd
import torch
from tqdm import tqdm

from src.data.preprocessing import (
    get_brain_mask,
    load_h5_full,
    mask_has_margin,
    mask_inside_brain_fraction,
)
from src.diffusion import DiffusionSchedule


REQUIRED_MANIFEST_COLUMNS = {
    "slice_path",
    "target",
    "label0_pxl_cnt",
    "label1_pxl_cnt",
    "label2_pxl_cnt",
}


def discover_composition_candidates(
    *,
    h5_root: str | Path,
    manifest_path: str | Path,
    min_tumor_pixels: int = 300,
) -> tuple[
    list[Path],
    list[Path],
]:
    """
    Reproduce Cell 21's tumor-free and donor-mask candidate discovery.

    Returns
    -------
    tuple
        ``(tumor_free_files, tumor_mask_files)``.
    """
    h5_root = Path(
        h5_root
    ).expanduser().resolve()

    manifest_path = Path(
        manifest_path
    ).expanduser().resolve()

    if not h5_root.is_dir():
        raise ValueError(
            f"H5 root does not exist:\n{h5_root}"
        )

    if not manifest_path.is_file():
        raise ValueError(
            f"Manifest does not exist:\n{manifest_path}"
        )

    manifest = pd.read_csv(
        manifest_path
    )

    missing = sorted(
        REQUIRED_MANIFEST_COLUMNS
        - set(
            manifest.columns
        )
    )

    if missing:
        raise ValueError(
            "Manifest is missing required column(s): "
            + ", ".join(
                missing
            )
        )

    tumor_pixels = (
        manifest[
            "label0_pxl_cnt"
        ]
        + manifest[
            "label1_pxl_cnt"
        ]
        + manifest[
            "label2_pxl_cnt"
        ]
    )

    tumor_free_rows = manifest.loc[
        tumor_pixels == 0
    ]

    tumor_mask_rows = manifest.loc[
        tumor_pixels
        >= min_tumor_pixels
    ]

    tumor_free_files = sorted(
        [
            h5_root
            / Path(
                str(
                    value
                )
            ).name
            for value in tumor_free_rows[
                "slice_path"
            ].tolist()
        ]
    )

    tumor_mask_files = sorted(
        [
            h5_root
            / Path(
                str(
                    value
                )
            ).name
            for value in tumor_mask_rows[
                "slice_path"
            ].tolist()
        ]
    )

    return (
        tumor_free_files,
        tumor_mask_files,
    )


def select_clean_insertion_pairs(
    *,
    tumor_free_files: list[Path],
    tumor_mask_files: list[Path],
    image_channel: int = 0,
    min_tumor_pixels: int = 300,
    max_base_candidates: int = 300,
    max_mask_candidates: int = 500,
    max_pairs: int = 4,
    min_overlap: float = 0.80,
    margin: int = 10,
    brain_threshold: float = 0.05,
    seed: int = 42,
) -> list[dict]:
    """
    Reproduce Cell 22's candidate subsampling and pair-selection logic.

    The first qualifying donor mask is selected for each candidate base, exactly
    as in the reference notebook.
    """
    if max_pairs <= 0:
        raise ValueError(
            "max_pairs must be positive."
        )

    rng = random.Random(
        seed
    )

    base_candidates = rng.sample(
        tumor_free_files,
        k=min(
            max_base_candidates,
            len(
                tumor_free_files
            ),
        ),
    )

    mask_candidates = rng.sample(
        tumor_mask_files,
        k=min(
            max_mask_candidates,
            len(
                tumor_mask_files
            ),
        ),
    )

    selected_pairs = []

    for base_path in tqdm(
        base_candidates,
        desc="Finding clean insertion pairs",
    ):
        (
            base_img,
            base_mask,
            _,
        ) = load_h5_full(
            base_path,
            image_channel=image_channel,
        )

        if (
            base_mask.sum().item()
            != 0
        ):
            continue

        brain_mask = get_brain_mask(
            base_img,
            threshold=brain_threshold,
        )

        for mask_path in mask_candidates:
            (
                _donor_img,
                donor_mask,
                _donor_cond,
            ) = load_h5_full(
                mask_path,
                image_channel=image_channel,
            )

            mask_pixels = donor_mask.sum().item()

            if (
                mask_pixels
                < min_tumor_pixels
            ):
                continue

            overlap_frac = mask_inside_brain_fraction(
                donor_mask,
                brain_mask,
            )

            has_margin = mask_has_margin(
                donor_mask,
                margin=margin,
            )

            if (
                overlap_frac
                >= min_overlap
                and has_margin
            ):
                selected_pairs.append(
                    {
                        "base_path":
                            base_path,

                        "mask_path":
                            mask_path,

                        "overlap_frac":
                            overlap_frac,

                        "mask_pixels":
                            mask_pixels,
                    }
                )

                break

        if (
            len(
                selected_pairs
            )
            >= max_pairs
        ):
            break

    return selected_pairs


@torch.no_grad()
def synthesize_insertion_pairs(
    *,
    model: torch.nn.Module,
    selected_pairs: list[dict],
    schedule: DiffusionSchedule,
    device: torch.device,
    image_channel: int = 0,
    timestep_fraction: float = 0.75,
) -> dict[str, torch.Tensor]:
    """
    Reproduce Cell 23's tumor-free insertion synthesis.

    Fresh Gaussian noise is sampled for each call, matching the notebook.
    """
    if not selected_pairs:
        raise ValueError(
            "selected_pairs is empty."
        )

    if not (
        0.0 <= timestep_fraction < 1.0
    ):
        raise ValueError(
            "timestep_fraction must satisfy 0 <= value < 1."
        )

    model.eval()

    base_images = []
    transferred_masks = []
    donor_patches = []
    donor_conditions = []

    for pair in selected_pairs:
        (
            base_img,
            _,
            _,
        ) = load_h5_full(
            pair[
                "base_path"
            ],
            image_channel=image_channel,
        )

        (
            donor_img,
            donor_mask,
            donor_cond,
        ) = load_h5_full(
            pair[
                "mask_path"
            ],
            image_channel=image_channel,
        )

        donor_patch = (
            donor_img
            * donor_mask
        )

        base_images.append(
            base_img
        )

        transferred_masks.append(
            donor_mask
        )

        donor_patches.append(
            donor_patch
        )

        donor_conditions.append(
            donor_cond
        )

    base_images = torch.stack(
        base_images,
        dim=0,
    ).to(
        device
    )

    transferred_masks = torch.stack(
        transferred_masks,
        dim=0,
    ).to(
        device
    )

    donor_patches = torch.stack(
        donor_patches,
        dim=0,
    ).to(
        device
    )

    donor_conditions = torch.stack(
        donor_conditions,
        dim=0,
    ).to(
        device
    )

    known = (
        base_images
        * (
            1.0
            - transferred_masks
        )
    )

    t_value = int(
        timestep_fraction
        * schedule.timesteps
    )

    t = torch.full(
        (
            base_images.shape[0],
        ),
        t_value,
        device=device,
        dtype=torch.long,
    )

    noise = torch.randn_like(
        base_images
    )

    x_t_full = schedule.q_sample(
        x0=base_images,
        t=t,
        noise=noise,
    )

    x_t = (
        base_images
        * (
            1.0
            - transferred_masks
        )
        + x_t_full
        * transferred_masks
    )

    model_input = torch.cat(
        [
            x_t,
            known,
            transferred_masks,
            donor_patches,
        ],
        dim=1,
    )

    pred_x0 = model(
        model_input,
        t,
        donor_conditions,
    )

    synthetic_composite = (
        base_images
        * (
            1.0
            - transferred_masks
        )
        + pred_x0
        * transferred_masks
    )

    return {
        "base_images":
            base_images,

        "transferred_masks":
            transferred_masks,

        "known":
            known,

        "donor_patches":
            donor_patches,

        "donor_conditions":
            donor_conditions,

        "t":
            t,

        "x_t":
            x_t,

        "pred_x0":
            pred_x0,

        "synthetic_composite":
            synthetic_composite,
    }
