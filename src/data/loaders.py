"""
Dataset splitting and DataLoader construction extracted from notebook Cell 4.
"""

from __future__ import annotations

from torch.utils.data import DataLoader, Dataset, Subset, random_split
import torch


def split_dataset(
    dataset: Dataset,
    train_fraction: float = 0.9,
    seed: int = 42,
) -> tuple[Subset, Subset]:
    """
    Split a Dataset exactly as in notebook Cell 4.

    Notebook defaults
    -----------------
    train_fraction = 0.9
    seed = 42

    The training size uses ``int(train_fraction * len(dataset))`` and the
    remainder is assigned to validation, matching the notebook.
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
    Create training and validation DataLoaders.

    Defaults reproduce notebook Cell 4:
    - batch_size = 8
    - train_fraction = 0.9
    - seed = 42
    - num_workers = 0
    - training shuffle = True
    - validation shuffle = False

    ``pin_memory`` is exposed for users but defaults to False so the notebook
    behavior is unchanged.

    Returns
    -------
    tuple
        ``(train_loader, val_loader, train_dataset, val_dataset)``
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
