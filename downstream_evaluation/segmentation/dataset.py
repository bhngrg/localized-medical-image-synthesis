#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset

from src.data.preprocessing import load_h5_full


class DownstreamBraTSSegmentationDataset(Dataset):
    """
    Slice-level BraTS dataset for downstream tumor segmentation.

    Each row in the supplied manifest must contain:
    - h5_relative_path
    - volume
    - slice_index

    Real H5 slices are resolved relative to ``h5_root`` and loaded using
    the repository's established preprocessing contract:
    - image channel 0 (FLAIR)
    - percentile clipping and [0, 1] scaling
    - binary whole-tumor mask via max(mask_channels) > 0
    """

    def __init__(
        self,
        manifest_path: str | Path,
        h5_root: str | Path,
        image_channel: int = 0,
        transform=None,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.h5_root = Path(h5_root)
        self.image_channel = int(image_channel)
        self.transform = transform

        if not self.manifest_path.is_file():
            raise FileNotFoundError(
                f"Manifest not found: {self.manifest_path}"
            )

        if not self.h5_root.is_dir():
            raise FileNotFoundError(
                f"H5 root not found: {self.h5_root}"
            )

        self.manifest = pd.read_csv(self.manifest_path)

        required_columns = {
            "h5_relative_path",
            "volume",
            "slice_index",
        }

        missing = required_columns - set(self.manifest.columns)

        if missing:
            raise ValueError(
                "Manifest is missing required columns: "
                + ", ".join(sorted(missing))
            )

        if self.manifest.duplicated(
            subset=["volume", "slice_index"]
        ).any():
            raise ValueError(
                "Manifest contains duplicate (volume, slice_index) rows."
            )

        self.manifest = (
            self.manifest
            .sort_values(["volume", "slice_index"])
            .reset_index(drop=True)
        )

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(
        self,
        index: int,
    ) -> dict[str, torch.Tensor | int | str]:
        row = self.manifest.iloc[index]

        h5_path = self.h5_root / str(
            row["h5_relative_path"]
        )

        if not h5_path.is_file():
            raise FileNotFoundError(
                f"H5 slice not found: {h5_path}"
            )

        image, mask, _ = load_h5_full(
            h5_path,
            image_channel=self.image_channel,
        )

        if self.transform is not None:
            augmented = self.transform(
                image=image.squeeze(0).numpy()[..., None],
                mask=mask.squeeze(0).numpy(),
            )

            image = augmented["image"]
            mask = augmented["mask"]

            if mask.ndim == 2:
                mask = mask.unsqueeze(0)

            image = image.float()
            mask = mask.float()

        return {
            "image": image,
            "mask": mask,
            "volume": int(row["volume"]),
            "slice_index": int(row["slice_index"]),
            "h5_relative_path": str(row["h5_relative_path"]),
        }
