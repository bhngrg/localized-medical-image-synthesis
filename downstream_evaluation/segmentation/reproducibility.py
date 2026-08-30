"""
Reproducibility helpers for downstream segmentation experiments.

The strict CUDA configuration implemented here was validated with a dedicated
GPU reproducibility diagnostic on NVIDIA A30 hardware. In that diagnostic,
two independently reinitialized runs within the same Slurm job and GPU
allocation, using the same seed, produced identical loss sequences and
identical final model-state hashes.

Cross-job reproducibility is verified separately by rerunning the hardened
training workflow after the repository refactor.

For CUDA jobs, launchers should also set:

    CUBLAS_WORKSPACE_CONFIG=:4096:8

before Python starts.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch
from torch.utils.data import ConcatDataset, get_worker_info


DEFAULT_CUBLAS_WORKSPACE_CONFIG = ":4096:8"


def configure_reproducibility(
    seed: int,
    *,
    deterministic_algorithms: bool = True,
    cudnn_deterministic: bool = True,
    cudnn_benchmark: bool = False,
) -> None:
    """
    Configure Python, NumPy, and PyTorch reproducibility controls.

    Strict deterministic algorithms are enabled by default because this is the
    configuration that passed the downstream CUDA bitwise-reproducibility
    diagnostic.
    """
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = cudnn_deterministic
        torch.backends.cudnn.benchmark = cudnn_benchmark

    torch.use_deterministic_algorithms(
        deterministic_algorithms
    )


def validate_cuda_reproducibility_environment() -> None:
    """
    Validate the cuBLAS workspace setting required for strict CUDA runs.

    This check is intentionally enforced only when CUDA is available.
    """
    if not torch.cuda.is_available():
        return

    value = os.environ.get(
        "CUBLAS_WORKSPACE_CONFIG"
    )

    if value != DEFAULT_CUBLAS_WORKSPACE_CONFIG:
        raise RuntimeError(
            "Strict downstream CUDA reproducibility requires "
            "CUBLAS_WORKSPACE_CONFIG=:4096:8 to be set before Python "
            "starts. Current value: "
            f"{value!r}"
        )


def seed_dataset_transform(
    dataset,
    seed: int,
) -> None:
    """
    Seed Albumentations-style transforms attached to a dataset.

    ConcatDataset children are handled recursively so the real and synthetic
    training datasets receive deterministic worker-specific transform seeds.
    """
    if isinstance(dataset, ConcatDataset):
        for child_dataset in dataset.datasets:
            seed_dataset_transform(
                child_dataset,
                seed,
            )
        return

    transform = getattr(
        dataset,
        "transform",
        None,
    )

    if hasattr(
        transform,
        "set_random_seed",
    ):
        transform.set_random_seed(seed)


def seed_worker(worker_id: int) -> None:
    """
    Deterministically seed one DataLoader worker and its dataset transform.
    """
    del worker_id

    worker_seed = torch.initial_seed() % (2**32)

    random.seed(worker_seed)
    np.random.seed(worker_seed)

    worker_info = get_worker_info()

    if worker_info is None:
        return

    seed_dataset_transform(
        worker_info.dataset,
        worker_seed,
    )


def make_generator(
    seed: int,
) -> torch.Generator:
    """
    Create an explicitly seeded PyTorch generator for a DataLoader.
    """
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator
