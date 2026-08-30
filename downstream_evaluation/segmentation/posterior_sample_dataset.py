#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class BRLoRAPosteriorSampleSegmentationDataset(Dataset):
    """
    Epoch-aware segmentation dataset for BR-LoRA posterior sampling.

    Each synthetic case has 100 stored posterior realizations.
    A deterministic, case-specific permutation of those 100
    realizations is generated from the experiment seed.

    For epoch e in {0, ..., 19}, one distinct realization is
    selected for each synthetic case. Thus, each case contributes
    one synthetic image per epoch, with no repeated realization
    across the 20-epoch training run.
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

    POSTERIOR_SAMPLES = 100

    def __init__(
        self,
        manifest_path: str | Path,
        library_root: str | Path,
        h5_root: str | Path,
        seed: int,
        transform=None,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.library_root = Path(library_root)
        self.h5_root = Path(h5_root)
        self.seed = int(seed)
        self.transform = transform
        self.epoch = 0

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

        self.realization_schedule = self._build_schedule()

    def __len__(self) -> int:
        return len(self.manifest)

    def set_epoch(
        self,
        epoch: int,
    ) -> None:
        epoch = int(epoch)

        if epoch < 0:
            raise ValueError(
                "Epoch must be non-negative."
            )

        if epoch >= self.POSTERIOR_SAMPLES:
            raise ValueError(
                f"Epoch {epoch} exceeds the available "
                f"{self.POSTERIOR_SAMPLES} posterior realizations."
            )

        self.epoch = epoch

    def _build_schedule(
        self,
    ) -> np.ndarray:
        schedule = np.empty(
            (
                len(self.manifest),
                self.POSTERIOR_SAMPLES,
            ),
            dtype=np.int16,
        )

        for row_index, row in self.manifest.iterrows():
            library_index = int(
                row["library_index"]
            )

            case_seed = (
                self.seed
                + library_index
            )

            rng = np.random.default_rng(
                case_seed
            )

            schedule[row_index] = rng.permutation(
                self.POSTERIOR_SAMPLES
            )

        return schedule

    @staticmethod
    def _case_directory(
        row: pd.Series,
    ) -> str:
        source_case_id = row["source_case_id"]

        if pd.notna(source_case_id):
            return str(source_case_id)

        return str(row["library_case_id"])

    def _posterior_samples_path(
        self,
        row: pd.Series,
    ) -> Path:
        return (
            self.library_root
            / str(row["batch_id"])
            / self._case_directory(row)
            / "posterior_samples.pt"
        )

    def _donor_h5_path(
        self,
        row: pd.Series,
    ) -> Path:
        return (
            self.h5_root
            / str(row["donor_h5_file"])
        )

    def _load_selected_realization(
        self,
        path: Path,
        realization_index: int,
    ) -> torch.Tensor:
        if not path.is_file():
            raise FileNotFoundError(
                f"Posterior-sample file not found: {path}"
            )

        obj = torch.load(
            path,
            map_location="cpu",
            mmap=True,
        )

        if not isinstance(obj, dict):
            raise TypeError(
                f"Expected dict in {path}, got {type(obj)}."
            )

        if "prediction_samples" not in obj:
            raise KeyError(
                f"'prediction_samples' not found in {path}"
            )

        if "posterior_samples" not in obj:
            raise KeyError(
                f"'posterior_samples' not found in {path}"
            )

        stored_count = int(
            obj["posterior_samples"]
        )

        if stored_count != self.POSTERIOR_SAMPLES:
            raise ValueError(
                f"Expected {self.POSTERIOR_SAMPLES} posterior samples "
                f"in {path}; got {stored_count}."
            )

        samples = obj["prediction_samples"]

        if not isinstance(samples, torch.Tensor):
            raise TypeError(
                f"Expected prediction_samples tensor in {path}."
            )

        expected_shape = (
            self.POSTERIOR_SAMPLES,
            1,
            1,
            240,
            240,
        )

        if tuple(samples.shape) != expected_shape:
            raise ValueError(
                f"Expected prediction_samples shape {expected_shape} "
                f"in {path}; got {tuple(samples.shape)}."
            )

        image = (
            samples[realization_index]
            .squeeze(0)
            .detach()
            .to(dtype=torch.float32)
        )

        if image.shape != (1, 240, 240):
            raise ValueError(
                f"Expected selected image shape (1, 240, 240) "
                f"in {path}; got {tuple(image.shape)}."
            )

        if not torch.isfinite(image).all():
            raise ValueError(
                f"Selected posterior realization contains "
                f"non-finite values: {path}"
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
                f"Expected whole-tumor mask shape (240, 240) "
                f"in {path}; got {whole_tumor.shape}."
            )

        return torch.from_numpy(
            whole_tumor
        ).unsqueeze(0)

    def __getitem__(
        self,
        index: int,
    ) -> dict[str, object]:
        row = self.manifest.iloc[index]

        realization_index = int(
            self.realization_schedule[
                index,
                self.epoch,
            ]
        )

        posterior_samples_path = (
            self._posterior_samples_path(
                row,
            )
        )

        donor_h5_path = self._donor_h5_path(
            row,
        )

        image = self._load_selected_realization(
            posterior_samples_path,
            realization_index,
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
            "posterior_realization_index": realization_index,
            "epoch": self.epoch,
            "donor_volume": int(
                row["donor_volume"]
            ),
            "donor_slice_index": int(
                row["donor_slice_index"]
            ),
            "donor_h5_file": str(
                row["donor_h5_file"]
            ),
            "posterior_samples_path": str(
                posterior_samples_path
            ),
        }
