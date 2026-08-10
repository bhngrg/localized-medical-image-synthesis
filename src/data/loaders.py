"""
Dataset splitting and DataLoader construction.

The default ``internal`` mode reproduces notebook Cell 4 exactly.
``full_train`` is an explicit extension that uses the complete eligible
training dataset without creating an internal validation split.
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split


def split_dataset(
    dataset: Dataset,
    train_fraction: float = 0.9,
    seed: int = 42,
) -> tuple[Subset, Subset]:
    """
    Split a Dataset exactly as in notebook Cell 4.
    """
    if not (
        0.0 < train_fraction < 1.0
    ):
        raise ValueError(
            "train_fraction must be strictly between 0 and 1."
        )

    train_size = int(
        train_fraction
        * len(
            dataset
        )
    )

    val_size = len(
        dataset
    ) - train_size

    generator = torch.Generator().manual_seed(
        seed
    )

    return random_split(
        dataset,
        [
            train_size,
            val_size,
        ],
        generator=generator,
    )


def create_train_val_loaders(
    dataset: Dataset,
    batch_size: int = 8,
    train_fraction: float = 0.9,
    seed: int = 42,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> tuple[
    DataLoader,
    DataLoader,
    Subset,
    Subset,
]:
    """
    Create notebook-equivalent internal train/validation loaders.
    """
    if batch_size <= 0:
        raise ValueError(
            "batch_size must be positive."
        )

    if num_workers < 0:
        raise ValueError(
            "num_workers must be non-negative."
        )

    train_dataset, val_dataset = split_dataset(
        dataset=dataset,
        train_fraction=train_fraction,
        seed=seed,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return (
        train_loader,
        val_loader,
        train_dataset,
        val_dataset,
    )


def create_full_train_loader(
    dataset: Dataset,
    batch_size: int = 8,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DataLoader:
    """
    Create a shuffled loader over 100% of the eligible training dataset.

    This is an explicit extension of the notebook baseline. No internal
    validation dataset is created.
    """
    if batch_size <= 0:
        raise ValueError(
            "batch_size must be positive."
        )

    if num_workers < 0:
        raise ValueError(
            "num_workers must be non-negative."
        )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
