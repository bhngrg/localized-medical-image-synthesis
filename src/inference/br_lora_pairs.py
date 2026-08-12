"""
Shared pair preparation for BR-LoRA localized inference.

This module contains the validated conversion from selected tumor-free
base / donor-mask pairs into the tensor contract consumed by
``prepare_br_lora_batch``.

Pair discovery and pair selection remain separate concerns.
"""

from __future__ import annotations

import torch

from src.data import load_h5_full


def prepare_selected_pairs(
    *,
    selected_pairs: list[dict],
    image_channel: int,
) -> tuple[
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
]:
    """
    Convert selected base/donor pairs into the validated inference contract.

    Returns both the dataset-style batch used by ``prepare_br_lora_batch`` and
    the stacked tensors retained for serialization or visualization.
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


__all__ = [
    "prepare_selected_pairs",
]
