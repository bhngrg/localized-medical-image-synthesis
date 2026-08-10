"""
Image and mask preprocessing utilities extracted from notebook Cell 2.

The implementations in this module intentionally preserve the behavior of the
reference notebook.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import torch


def normalize_image_channel(x: np.ndarray) -> np.ndarray:
    """
    Normalize one 2-D MRI channel exactly as in the reference notebook.

    Pixels with values greater than zero are treated as brain pixels for
    percentile estimation. If more than 10 such pixels are present, the 1st
    and 99th percentiles are used as clipping bounds; otherwise the full-image
    minimum and maximum are used.

    Parameters
    ----------
    x
        Two-dimensional image array.

    Returns
    -------
    numpy.ndarray
        Float32 image scaled approximately to [0, 1].
    """
    brain = x > 0

    if brain.sum() > 10:
        lo, hi = np.percentile(
            x[brain],
            [1, 99],
        )
    else:
        lo, hi = x.min(), x.max()

    x = np.clip(
        x,
        lo,
        hi,
    )

    x = (
        x - lo
    ) / (
        hi - lo + 1e-8
    )

    return x.astype(
        np.float32
    )


def compute_tumor_appearance_stats(
    x: np.ndarray,
    m: np.ndarray,
) -> np.ndarray:
    """
    Compute the four-element donor/tumor conditioning vector.

    Parameters
    ----------
    x
        Normalized 2-D image.
    m
        Binary 2-D tumor mask.

    Returns
    -------
    numpy.ndarray
        ``[mean, std, max, area_fraction]`` as float32.
    """
    tumor_pixels = x[
        m > 0
    ]

    if tumor_pixels.size == 0:
        return np.array(
            [0, 0, 0, 0],
            dtype=np.float32,
        )

    mean = tumor_pixels.mean()
    std = tumor_pixels.std()
    max_val = tumor_pixels.max()
    area_fraction = m.mean()

    return np.array(
        [
            mean,
            std,
            max_val,
            area_fraction,
        ],
        dtype=np.float32,
    )


def load_h5_full(
    path: str | Path,
    image_channel: int = 0,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """
    Load one H5 slice and reproduce the notebook preprocessing.

    Notes
    -----
    The H5 mask is collapsed across all three segmentation channels using
    ``max(axis=-1)``. This is the whole-tumor mask used by the reference
    notebook regardless of the slice-selection mode used by the Dataset.

    Parameters
    ----------
    path
        H5 file path.
    image_channel
        MRI image channel to use. Notebook default: 0 (FLAIR).

    Returns
    -------
    tuple
        ``(x, mask, cond)`` where ``x`` and ``mask`` have shape
        ``[1, H, W]`` and ``cond`` has shape ``[4]``.
    """
    path = Path(
        path
    )

    with h5py.File(
        path,
        "r",
    ) as h5_file:
        image = h5_file[
            "image"
        ][:].astype(
            np.float32
        )

        mask = h5_file[
            "mask"
        ][:].astype(
            np.float32
        )

    if image_channel < 0 or image_channel >= image.shape[-1]:
        raise ValueError(
            f"image_channel={image_channel} is outside the valid "
            f"range 0-{image.shape[-1] - 1} for {path}."
        )

    x = image[
        :,
        :,
        image_channel,
    ]

    x = normalize_image_channel(
        x
    )

    mask = mask.max(
        axis=-1
    )

    mask = (
        mask > 0
    ).astype(
        np.float32
    )

    cond = compute_tumor_appearance_stats(
        x,
        mask,
    )

    x_tensor = torch.from_numpy(
        x[
            None,
            :,
            :,
        ]
    ).float()

    mask_tensor = torch.from_numpy(
        mask[
            None,
            :,
            :,
        ]
    ).float()

    cond_tensor = torch.from_numpy(
        cond
    ).float()

    return (
        x_tensor,
        mask_tensor,
        cond_tensor,
    )


def get_brain_mask(
    image: torch.Tensor,
    threshold: float = 0.05,
) -> torch.Tensor:
    """
    Construct the notebook's threshold-based brain mask.

    Notebook default threshold: 0.05.
    """
    return (
        image > threshold
    ).float()


def mask_inside_brain_fraction(
    mask: torch.Tensor,
    brain_mask: torch.Tensor,
) -> float:
    """
    Return the fraction of mask pixels that lie inside a brain mask.
    """
    mask_area = mask.sum()

    if mask_area.item() == 0:
        return 0.0

    overlap = (
        mask * brain_mask
    ).sum()

    return (
        overlap
        / (
            mask_area + 1e-8
        )
    ).item()


def mask_has_margin(
    mask: torch.Tensor,
    margin: int = 10,
) -> bool:
    """
    Check whether a binary mask remains at least ``margin`` pixels from edges.

    Notebook default margin: 10.
    """
    m = mask[
        0
    ] > 0

    if m.sum().item() == 0:
        return False

    ys, xs = torch.where(
        m
    )

    y_min = ys.min().item()
    y_max = ys.max().item()
    x_min = xs.min().item()
    x_max = xs.max().item()

    height, width = m.shape

    return (
        y_min >= margin
        and x_min >= margin
        and y_max < height - margin
        and x_max < width - margin
    )
