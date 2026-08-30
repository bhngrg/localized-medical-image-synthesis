#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class BRLoRAPosteriorMeanSegmentationDataset(Dataset):
    """
    Segmentation dataset for the fixed BR-LoRA posterior-mean library.

    Each synthetic image is paired with the whole-tumor mask from the
    corresponding donor H5 slice specified by the frozen library design
    manifest.
    """

    REQUIRED_COLUMNS = (
        "library_index",
        "library_case_id",
        "batch_id",
        "source_case_id",
        "donor_volume",
        "donor_slice_index",
        "donor_h5_file",
        "donor_mask_pixels",
    )

    def __init__(
        self,
        manifest_path: str | Path,
        library_root: str | Path,
        h5_root: str | Path,
        transform=None,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.library_root = Path(library_root)
        self.h5_root = Path(h5_root)
        self.transform = transform

        if not self.manifest_path.is_file():
            raise FileNotFoundError(
                f"Synthetic manifest not found: {self.manifest_path}"
            )

        if not self.library_root.is_dir():
            raise FileNotFoundError(
                f"BR-LoRA library root not found: {self.library_root}"
            )

        if not self.h5_root.is_dir():
            raise FileNotFoundError(
                f"BraTS H5 root not found: {self.h5_root}"
            )

        self.manifest = pd.read_csv(
            self.manifest_path,
        )

        missing_columns = [
            column
            for column in self.REQUIRED_COLUMNS
            if column not in self.manifest.columns
        ]

        if missing_columns:
            raise ValueError(
                "Synthetic manifest is missing required columns: "
                + ", ".join(missing_columns)
            )

        if self.manifest["library_index"].duplicated().any():
            raise ValueError(
                "Synthetic manifest contains duplicate library_index values."
            )

        self.manifest = (
            self.manifest
            .sort_values("library_index")
            .reset_index(drop=True)
        )

    def __len__(self) -> int:
        return len(self.manifest)

    @staticmethod
    def _case_directory(row: pd.Series) -> str:
        source_case_id = row["source_case_id"]

        if pd.notna(source_case_id):
            return str(source_case_id)

        return str(row["library_case_id"])

    def _posterior_mean_path(
        self,
        row: pd.Series,
    ) -> Path:
        return (
            self.library_root
            / str(row["batch_id"])
            / self._case_directory(row)
            / "posterior_mean.pt"
        )

    def _donor_h5_path(
        self,
        row: pd.Series,
    ) -> Path:
        return (
            self.h5_root
            / str(row["donor_h5_file"])
        )

    @staticmethod
    def _load_posterior_mean(
        path: Path,
    ) -> torch.Tensor:
        if not path.is_file():
            raise FileNotFoundError(
                f"Posterior-mean file not found: {path}"
            )

        image = torch.load(
            path,
            map_location="cpu",
        )

        if not isinstance(image, torch.Tensor):
            raise TypeError(
                f"Expected tensor in {path}, got {type(image)}."
            )

        image = image.detach().to(
            dtype=torch.float32,
        )

        if image.ndim == 4:
            if image.shape[0] != 1:
                raise ValueError(
                    f"Expected leading batch dimension 1 in {path}; "
                    f"got shape {tuple(image.shape)}."
                )
            image = image.squeeze(0)

        if image.shape != (1, 240, 240):
            raise ValueError(
                f"Expected synthetic image shape (1, 240, 240) in {path}; "
                f"got {tuple(image.shape)}."
            )

        if not torch.isfinite(image).all():
            raise ValueError(
                f"Synthetic image contains non-finite values: {path}"
            )

        return image

    @staticmethod
    def _load_whole_tumor_mask(
        path: Path,
    ) -> torch.Tensor:
        if not path.is_file():
            raise FileNotFoundError(
                f"Donor H5 file not found: {path}"
            )

        with h5py.File(path, "r") as h5_file:
            if "mask" not in h5_file:
                raise KeyError(
                    f"Dataset 'mask' not found in {path}"
                )

            mask_array = np.asarray(
                h5_file["mask"]
            )

        if mask_array.ndim != 3:
            raise ValueError(
                f"Expected donor mask shape [H,W,C] in {path}; "
                f"got {mask_array.shape}."
            )

        whole_tumor = (
            mask_array.max(axis=-1) > 0
        ).astype(
            np.float32,
            copy=False,
        )

        if whole_tumor.shape != (240, 240):
            raise ValueError(
                f"Expected whole-tumor mask shape (240, 240) in {path}; "
                f"got {whole_tumor.shape}."
            )

        return torch.from_numpy(
            whole_tumor
        ).unsqueeze(0)

    def __getitem__(
        self,
        index: int,
    ) -> dict[str, object]:
        row = self.manifest.iloc[index]

        posterior_mean_path = self._posterior_mean_path(
            row,
        )

        donor_h5_path = self._donor_h5_path(
            row,
        )

        image = self._load_posterior_mean(
            posterior_mean_path,
        )

        mask = self._load_whole_tumor_mask(
            donor_h5_path,
        )

        expected_mask_pixels = int(
            row["donor_mask_pixels"]
        )

        observed_mask_pixels = int(
            mask.sum().item()
        )

        if observed_mask_pixels != expected_mask_pixels:
            raise ValueError(
                "Donor-mask pixel count mismatch for "
                f"{row['library_case_id']}: "
                f"manifest={expected_mask_pixels}, "
                f"observed={observed_mask_pixels}."
            )

        if self.transform is not None:
            transformed = self.transform(
                image=(
                    image.squeeze(0)
                    .numpy()[..., None]
                ),
                mask=mask.squeeze(0).numpy(),
            )

            image = transformed["image"]
            mask = transformed["mask"]

            if mask.ndim == 2:
                mask = mask.unsqueeze(0)

            image = image.to(
                dtype=torch.float32,
            )

            mask = mask.to(
                dtype=torch.float32,
            )

        return {
            "image": image,
            "mask": mask,
            "library_index": int(
                row["library_index"]
            ),
            "library_case_id": str(
                row["library_case_id"]
            ),
            "batch_id": str(
                row["batch_id"]
            ),
            "donor_volume": int(
                row["donor_volume"]
            ),
            "donor_slice_index": int(
                row["donor_slice_index"]
            ),
            "donor_h5_file": str(
                row["donor_h5_file"]
            ),
            "posterior_mean_path": str(
                posterior_mean_path
            ),
        }
