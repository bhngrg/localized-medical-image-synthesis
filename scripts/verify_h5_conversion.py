#!/usr/bin/env python3

"""
Temporarily verify how the BraTS NIfTI volumes were converted to H5 slices.

This script compares:
    - the four image channels in one H5 slice
    - the four corresponding BraTS MRI modalities
    - the three H5 mask channels
    - BraTS segmentation labels 1, 2, and 4
    - several candidate intensity-standardization strategies

It is intended only for reverse engineering the historical H5 conversion.
Once the conversion convention is confirmed and documented, this script can
be removed.

Example
-------
python scripts/verify_h5_conversion.py \
    --h5 /path/to/volume_1_slice_70.h5 \
    --flair /path/to/BraTS20_Training_001_flair.nii \
    --t1 /path/to/BraTS20_Training_001_t1.nii \
    --t1ce /path/to/BraTS20_Training_001_t1ce.nii \
    --t2 /path/to/BraTS20_Training_001_t2.nii \
    --seg /path/to/BraTS20_Training_001_seg.nii \
    --slice 70
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import nibabel as nib
import numpy as np


MODALITY_NAMES = ("FLAIR", "T1", "T1ce", "T2")
SEGMENTATION_LABELS = (1, 2, 4)

# Simple 2D transforms are tested explicitly so that orientation differences
# are reported rather than silently assumed.
TRANSFORMS = {
    "identity": lambda x: x,
    "flip_axis0": lambda x: np.flip(x, axis=0),
    "flip_axis1": lambda x: np.flip(x, axis=1),
    "flip_both": lambda x: np.flip(np.flip(x, axis=0), axis=1),
    "transpose": lambda x: x.T,
    "transpose_flip_axis0": lambda x: np.flip(x.T, axis=0),
    "transpose_flip_axis1": lambda x: np.flip(x.T, axis=1),
    "transpose_flip_both": lambda x: np.flip(
        np.flip(x.T, axis=0),
        axis=1,
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare one historical BraTS H5 slice with its original "
            "BraTS NIfTI volumes."
        )
    )

    parser.add_argument(
        "--h5",
        type=Path,
        required=True,
        help="Path to the historical H5 slice.",
    )

    parser.add_argument(
        "--flair",
        type=Path,
        required=True,
        help="Path to the corresponding FLAIR NIfTI volume.",
    )

    parser.add_argument(
        "--t1",
        type=Path,
        required=True,
        help="Path to the corresponding T1 NIfTI volume.",
    )

    parser.add_argument(
        "--t1ce",
        type=Path,
        required=True,
        help="Path to the corresponding T1ce NIfTI volume.",
    )

    parser.add_argument(
        "--t2",
        type=Path,
        required=True,
        help="Path to the corresponding T2 NIfTI volume.",
    )

    parser.add_argument(
        "--seg",
        type=Path,
        required=True,
        help="Path to the corresponding segmentation NIfTI volume.",
    )

    parser.add_argument(
        "--slice",
        type=int,
        required=True,
        dest="slice_index",
        help="Explicit zero-based NIfTI slice index to compare.",
    )

    return parser.parse_args()


def validate_input_file(
    path: Path,
    description: str,
) -> Path:
    path = path.expanduser().resolve()

    if not path.exists():
        raise FileNotFoundError(
            f"{description} does not exist:\n{path}"
        )

    if not path.is_file():
        raise ValueError(
            f"{description} is not a file:\n{path}"
        )

    return path


def load_h5_slice(
    path: Path,
) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as h5_file:
        if "image" not in h5_file:
            raise ValueError(
                f"H5 file does not contain 'image':\n{path}"
            )

        if "mask" not in h5_file:
            raise ValueError(
                f"H5 file does not contain 'mask':\n{path}"
            )

        image = np.asarray(h5_file["image"])
        mask = np.asarray(h5_file["mask"])

    if image.ndim != 3:
        raise ValueError(
            f"Expected H5 image to be 3-D; observed {image.shape}."
        )

    if mask.ndim != 3:
        raise ValueError(
            f"Expected H5 mask to be 3-D; observed {mask.shape}."
        )

    return image, mask


def load_nifti_volume(
    path: Path,
) -> np.ndarray:
    nii = nib.load(str(path))
    return np.asarray(nii.get_fdata())


def extract_slice(
    volume: np.ndarray,
    slice_index: int,
    description: str,
) -> np.ndarray:
    if volume.ndim != 3:
        raise ValueError(
            f"{description} must be a 3-D volume; "
            f"observed shape {volume.shape}."
        )

    if not 0 <= slice_index < volume.shape[2]:
        raise ValueError(
            f"Requested slice {slice_index} is outside the valid range "
            f"0-{volume.shape[2] - 1} for {description}."
        )

    return np.asarray(
        volume[:, :, slice_index]
    )


def pearson_correlation(
    a: np.ndarray,
    b: np.ndarray,
) -> float:
    if a.shape != b.shape:
        return float("nan")

    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()

    finite = np.isfinite(a) & np.isfinite(b)

    a = a[finite]
    b = b[finite]

    if a.size == 0:
        return float("nan")

    a_std = np.std(a)
    b_std = np.std(b)

    if a_std == 0 or b_std == 0:
        return float("nan")

    return float(
        np.corrcoef(a, b)[0, 1]
    )


def dice_coefficient(
    a: np.ndarray,
    b: np.ndarray,
) -> float:
    if a.shape != b.shape:
        return float("nan")

    a = a.astype(bool)
    b = b.astype(bool)

    denominator = a.sum() + b.sum()

    if denominator == 0:
        return 1.0

    intersection = np.logical_and(
        a,
        b,
    ).sum()

    return float(
        2.0 * intersection / denominator
    )


def best_image_match(
    h5_channel: np.ndarray,
    modality_slice: np.ndarray,
) -> tuple[float, str]:
    best_corr = -np.inf
    best_transform = ""

    for transform_name, transform_fn in TRANSFORMS.items():
        transformed = transform_fn(
            modality_slice
        )

        corr = pearson_correlation(
            h5_channel,
            transformed,
        )

        if np.isnan(corr):
            continue

        if corr > best_corr:
            best_corr = corr
            best_transform = transform_name

    if best_corr == -np.inf:
        return (
            float("nan"),
            "no valid comparison",
        )

    return (
        best_corr,
        best_transform,
    )


def best_mask_match(
    h5_mask: np.ndarray,
    segmentation_mask: np.ndarray,
) -> tuple[float, str]:
    best_dice = -np.inf
    best_transform = ""

    for transform_name, transform_fn in TRANSFORMS.items():
        transformed = transform_fn(
            segmentation_mask
        )

        dice = dice_coefficient(
            h5_mask,
            transformed,
        )

        if np.isnan(dice):
            continue

        if dice > best_dice:
            best_dice = dice
            best_transform = transform_name

    if best_dice == -np.inf:
        return (
            float("nan"),
            "no valid comparison",
        )

    return (
        best_dice,
        best_transform,
    )


def print_array_summary(
    name: str,
    array: np.ndarray,
) -> None:
    finite = array[
        np.isfinite(array)
    ]

    print(name)
    print(f"  shape : {array.shape}")
    print(f"  dtype : {array.dtype}")

    if finite.size == 0:
        print("  min   : NA")
        print("  max   : NA")
        print("  mean  : NA")
        print("  std   : NA")
        return

    print(
        f"  min   : {finite.min():.6g}"
    )
    print(
        f"  max   : {finite.max():.6g}"
    )
    print(
        f"  mean  : {finite.mean():.6g}"
    )
    print(
        f"  std   : {finite.std():.6g}"
    )


def verify_standardization(
    h5_image: np.ndarray,
    modality_slices: dict[str, np.ndarray],
) -> None:
    """
    Compare historical H5 intensities against candidate z-score strategies.
    """

    print(
        "\nZ-score standardization verification"
    )
    print("-" * 72)

    for channel_index, modality_name in enumerate(
        MODALITY_NAMES
    ):
        raw = modality_slices[
            modality_name
        ].astype(np.float64)

        stored = h5_image[
            :, :, channel_index
        ].astype(np.float64)

        # ----------------------------------------------------
        # Candidate 1:
        # Mean/std computed from all pixels.
        # ----------------------------------------------------
        mean_all = raw.mean()
        std_all = raw.std()

        if std_all == 0:
            raise ValueError(
                f"{modality_name} slice has zero "
                "standard deviation."
            )

        z_all = (
            raw - mean_all
        ) / std_all

        # ----------------------------------------------------
        # Candidate 2:
        # Mean/std computed from nonzero pixels, but applied
        # to all pixels.
        # ----------------------------------------------------
        nonzero = raw != 0

        if not np.any(nonzero):
            raise ValueError(
                f"{modality_name} slice contains "
                "no nonzero pixels."
            )

        mean_nonzero = raw[
            nonzero
        ].mean()

        std_nonzero = raw[
            nonzero
        ].std()

        if std_nonzero == 0:
            raise ValueError(
                f"{modality_name} nonzero pixels "
                "have zero standard deviation."
            )

        z_nonzero_full = (
            raw - mean_nonzero
        ) / std_nonzero

        # ----------------------------------------------------
        # Candidate 3:
        # Normalize only nonzero pixels and keep background 0.
        # ----------------------------------------------------
        z_nonzero_background_zero = np.zeros_like(
            raw,
            dtype=np.float64,
        )

        z_nonzero_background_zero[
            nonzero
        ] = (
            raw[nonzero] - mean_nonzero
        ) / std_nonzero

        candidates = {
            "all pixels":
                z_all,

            "nonzero statistics applied to all pixels":
                z_nonzero_full,

            "nonzero-only with zero background":
                z_nonzero_background_zero,
        }

        print(f"\n{modality_name}")

        best_name = None
        best_mae = float("inf")
        best_max_error = float("inf")

        for candidate_name, candidate in candidates.items():
            difference = (
                stored - candidate
            )

            mae = float(
                np.mean(
                    np.abs(difference)
                )
            )

            max_error = float(
                np.max(
                    np.abs(difference)
                )
            )

            print(
                f"  {candidate_name}:"
            )
            print(
                f"    MAE       = {mae:.12g}"
            )
            print(
                f"    max error = {max_error:.12g}"
            )

            if mae < best_mae:
                best_mae = mae
                best_max_error = max_error
                best_name = candidate_name

        print(
            f"  BEST MATCH: {best_name}"
        )
        print(
            f"    MAE       = {best_mae:.12g}"
        )
        print(
            f"    max error = {best_max_error:.12g}"
        )


def main() -> None:
    args = parse_args()

    h5_path = validate_input_file(
        args.h5,
        "H5 file",
    )

    modality_paths = {
        "FLAIR": validate_input_file(
            args.flair,
            "FLAIR file",
        ),
        "T1": validate_input_file(
            args.t1,
            "T1 file",
        ),
        "T1ce": validate_input_file(
            args.t1ce,
            "T1ce file",
        ),
        "T2": validate_input_file(
            args.t2,
            "T2 file",
        ),
    }

    seg_path = validate_input_file(
        args.seg,
        "Segmentation file",
    )

    print("=" * 72)
    print(
        "BraTS H5 Conversion Verification"
    )
    print("=" * 72)
    print(
        f"H5 file     : {h5_path}"
    )
    print(
        f"Slice index : {args.slice_index}"
    )

    h5_image, h5_mask = load_h5_slice(
        h5_path
    )

    print(
        "\nHistorical H5 structure"
    )
    print("-" * 72)

    print(
        f"image shape : {h5_image.shape}"
    )
    print(
        f"image dtype : {h5_image.dtype}"
    )
    print(
        f"mask shape  : {h5_mask.shape}"
    )
    print(
        f"mask dtype  : {h5_mask.dtype}"
    )

    if h5_image.shape[-1] != 4:
        raise ValueError(
            "Expected four H5 image channels for this "
            "verification, but observed "
            f"{h5_image.shape[-1]}."
        )

    if h5_mask.shape[-1] != 3:
        raise ValueError(
            "Expected three H5 mask channels for this "
            "verification, but observed "
            f"{h5_mask.shape[-1]}."
        )

    modality_slices = {}

    for modality_name, path in modality_paths.items():
        volume = load_nifti_volume(
            path
        )

        modality_slices[
            modality_name
        ] = extract_slice(
            volume,
            args.slice_index,
            modality_name,
        )

    seg_volume = load_nifti_volume(
        seg_path
    )

    seg_slice = extract_slice(
        seg_volume,
        args.slice_index,
        "segmentation",
    )

    print(
        "\nNIfTI volume information"
    )
    print("-" * 72)

    for modality_name, path in modality_paths.items():
        volume = nib.load(
            str(path)
        )

        print(
            f"{modality_name:<6}: "
            f"shape={volume.shape}, "
            f"dtype={volume.get_data_dtype()}"
        )

    seg_nii = nib.load(
        str(seg_path)
    )

    print(
        f"{'SEG':<6}: "
        f"shape={seg_nii.shape}, "
        f"dtype={seg_nii.get_data_dtype()}"
    )

    print(
        "\nH5 image-channel statistics"
    )
    print("-" * 72)

    for channel_index in range(
        h5_image.shape[-1]
    ):
        print_array_summary(
            f"H5 image channel {channel_index}",
            h5_image[
                :, :, channel_index
            ],
        )

    print(
        "\nNIfTI slice statistics"
    )
    print("-" * 72)

    for modality_name in MODALITY_NAMES:
        print_array_summary(
            modality_name,
            modality_slices[
                modality_name
            ],
        )

    # ========================================================
    # Verify historical z-score normalization
    # ========================================================
    verify_standardization(
        h5_image=h5_image,
        modality_slices=modality_slices,
    )

    # ========================================================
    # MRI channel mapping
    # ========================================================
    print(
        "\nMRI channel comparison"
    )
    print("-" * 72)

    print(
        "Each cell reports the highest Pearson correlation "
        "obtained across the explicitly tested 2-D orientation "
        "transforms."
    )

    correlation_results = {}

    header = (
        f"{'H5 channel':<14}"
        + "".join(
            f"{name:>18}"
            for name in MODALITY_NAMES
        )
    )

    print()
    print(header)

    for channel_index in range(4):
        row = f"{channel_index:<14}"

        correlation_results[
            channel_index
        ] = {}

        for modality_name in MODALITY_NAMES:
            corr, transform = best_image_match(
                h5_image[
                    :, :, channel_index
                ],
                modality_slices[
                    modality_name
                ],
            )

            correlation_results[
                channel_index
            ][
                modality_name
            ] = (
                corr,
                transform,
            )

            row += f"{corr:>18.6f}"

        print(row)

    print(
        "\nBest MRI matches"
    )
    print("-" * 72)

    inferred_image_mapping = {}

    for channel_index in range(4):
        candidates = correlation_results[
            channel_index
        ]

        best_modality = max(
            candidates,
            key=lambda name: (
                -np.inf
                if np.isnan(
                    candidates[name][0]
                )
                else candidates[name][0]
            ),
        )

        best_corr, best_transform = candidates[
            best_modality
        ]

        inferred_image_mapping[
            channel_index
        ] = (
            best_modality,
            best_corr,
            best_transform,
        )

        print(
            f"H5 image channel {channel_index} -> "
            f"{best_modality:<6} "
            f"(correlation={best_corr:.6f}, "
            f"transform={best_transform})"
        )

    # ========================================================
    # Segmentation mapping
    # ========================================================
    print(
        "\nSegmentation values in selected NIfTI slice"
    )
    print("-" * 72)

    unique_values, counts = np.unique(
        seg_slice.astype(np.int64),
        return_counts=True,
    )

    for value, count in zip(
        unique_values,
        counts,
    ):
        print(
            f"label {value:<2}: "
            f"{count:,} pixels"
        )

    print(
        "\nH5 mask pixel counts"
    )
    print("-" * 72)

    for channel_index in range(3):
        positive_pixels = np.count_nonzero(
            h5_mask[
                :, :, channel_index
            ]
        )

        print(
            f"H5 mask channel {channel_index}: "
            f"{positive_pixels:,} pixels"
        )

    print(
        "\nMask-channel comparison"
    )
    print("-" * 72)

    mask_results = {}

    header = (
        f"{'H5 mask':<14}"
        + "".join(
            f"{('label ' + str(label)):>18}"
            for label in SEGMENTATION_LABELS
        )
    )

    print()
    print(header)

    for channel_index in range(3):
        row = f"{channel_index:<14}"

        mask_results[
            channel_index
        ] = {}

        for label in SEGMENTATION_LABELS:
            nifti_mask = (
                seg_slice == label
            )

            dice, transform = best_mask_match(
                h5_mask[
                    :, :, channel_index
                ] > 0,
                nifti_mask,
            )

            mask_results[
                channel_index
            ][label] = (
                dice,
                transform,
            )

            row += f"{dice:>18.6f}"

        print(row)

    print(
        "\nBest mask matches"
    )
    print("-" * 72)

    inferred_mask_mapping = {}

    for channel_index in range(3):
        candidates = mask_results[
            channel_index
        ]

        best_label = max(
            candidates,
            key=lambda label: (
                -np.inf
                if np.isnan(
                    candidates[label][0]
                )
                else candidates[label][0]
            ),
        )

        best_dice, best_transform = candidates[
            best_label
        ]

        inferred_mask_mapping[
            channel_index
        ] = (
            best_label,
            best_dice,
            best_transform,
        )

        print(
            f"H5 mask channel {channel_index} -> "
            f"BraTS label {best_label} "
            f"(Dice={best_dice:.6f}, "
            f"transform={best_transform})"
        )

    # ========================================================
    # Final inferred conversion
    # ========================================================
    print(
        "\n" + "=" * 72
    )
    print(
        "INFERRED H5 CONVERSION"
    )
    print("=" * 72)

    print(
        "\nMRI channel order"
    )

    for channel_index in range(4):
        modality, corr, transform = (
            inferred_image_mapping[
                channel_index
            ]
        )

        print(
            f"  H5 image channel {channel_index} -> "
            f"{modality} "
            f"[r={corr:.6f}; {transform}]"
        )

    print(
        "\nMask channel order"
    )

    for channel_index in range(3):
        label, dice, transform = (
            inferred_mask_mapping[
                channel_index
            ]
        )

        print(
            f"  H5 mask channel {channel_index} -> "
            f"BraTS label {label} "
            f"[Dice={dice:.6f}; {transform}]"
        )

    print(
        "\nInterpretation guide"
    )
    print("-" * 72)

    print(
        "MRI correlations close to 1.0 indicate a strong "
        "modality match. Correlation is insensitive to simple "
        "linear intensity rescaling."
    )

    print(
        "Mask Dice values of 1.0 indicate exact agreement "
        "between an H5 mask channel and the corresponding "
        "BraTS segmentation label."
    )

    print(
        "The reported transform shows whether the historical "
        "conversion preserved the NIfTI slice orientation or "
        "applied a flip/transpose."
    )

    print(
        "The z-score verification compares the historical H5 "
        "intensities against three explicit normalization "
        "strategies."
    )


if __name__ == "__main__":
    main()