#!/usr/bin/env python3

from __future__ import annotations

import albumentations as A
from albumentations.pytorch import ToTensorV2


def build_train_transform():
    """
    Geometric augmentation for downstream BraTS segmentation training.

    Intensity normalization is intentionally omitted because H5 loading
    already applies the repository's established FLAIR normalization.
    """
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Transpose(p=0.5),
            ToTensorV2(),
        ]
    )


def build_validation_transform():
    """
    Deterministic validation transform.

    No resizing, augmentation, or additional normalization is applied.
    """
    return A.Compose(
        [
            ToTensorV2(),
        ]
    )
