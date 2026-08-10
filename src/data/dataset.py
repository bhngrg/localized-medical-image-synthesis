"""
BraTS H5 dataset used by the patch-conditioned x0 diffusion baseline.

This module refactors notebook Cell 3 while preserving its sample semantics.
The only engineering change is that slice filtering can use the precomputed
manifest rather than reopening every H5 mask during Dataset construction.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset

from .preprocessing import load_h5_full


REQUIRED_MANIFEST_COLUMNS = {
    "slice_path",
    "volume",
    "slice",
    "label0_pxl_cnt",
    "label1_pxl_cnt",
    "label2_pxl_cnt",
}


class BraTSH5PatchX0Dataset(Dataset):
    """
    Patch-conditioned H5 dataset matching the reference notebook.

    Parameters
    ----------
    root
        Directory containing the reconstructed H5 slices.
    manifest_path
        Path to ``manifest.csv``. Using the manifest avoids the expensive
        mask scan performed by the original notebook.
    image_channel
        MRI channel index. Notebook default: 0 (FLAIR).
    min_tumor_pixels
        Minimum tumor-pixel count required for a slice. Notebook default: 300.
    use_whole_tumor
        If True, filtering uses the sum of all three H5 mask-channel counts.
        If False, filtering uses mask channel 1 only, matching Cell 3.

        Important: this parameter changes slice *selection* only. Returned
        samples still use the whole-tumor mask because ``load_h5_full()``
        reproduces the notebook's ``mask.max(axis=-1)`` behavior.
    """

    def __init__(
        self,
        root: str | Path,
        manifest_path: str | Path,
        image_channel: int = 0,
        min_tumor_pixels: int = 300,
        use_whole_tumor: bool = True,
    ) -> None:
        self.root = Path(
            root
        ).expanduser().resolve()

        self.manifest_path = Path(
            manifest_path
        ).expanduser().resolve()

        self.image_channel = int(
            image_channel
        )

        self.min_tumor_pixels = int(
            min_tumor_pixels
        )

        self.use_whole_tumor = bool(
            use_whole_tumor
        )

        if not self.root.is_dir():
            raise ValueError(
                "H5 dataset directory does not exist or is not a directory:\n"
                f"{self.root}"
            )

        if not self.manifest_path.is_file():
            raise ValueError(
                "Manifest file does not exist:\n"
                f"{self.manifest_path}"
            )

        if self.min_tumor_pixels < 0:
            raise ValueError(
                "min_tumor_pixels must be non-negative."
            )

        manifest = pd.read_csv(
            self.manifest_path
        )

        missing_columns = sorted(
            REQUIRED_MANIFEST_COLUMNS
            - set(
                manifest.columns
            )
        )

        if missing_columns:
            raise ValueError(
                "Manifest is missing required column(s): "
                + ", ".join(
                    missing_columns
                )
            )

        if self.use_whole_tumor:
            tumor_pixel_count = (
                manifest[
                    "label0_pxl_cnt"
                ]
                + manifest[
                    "label1_pxl_cnt"
                ]
                + manifest[
                    "label2_pxl_cnt"
                ]
            )
        else:
            # Exact notebook behavior:
            # mask[:, :, 1] is used only for slice filtering.
            tumor_pixel_count = manifest[
                "label1_pxl_cnt"
            ]

        selected = manifest.loc[
            tumor_pixel_count
            >= self.min_tumor_pixels
        ].copy()

        if selected.empty:
            raise RuntimeError(
                "No tumor-containing slices satisfy the configured "
                "minimum tumor-pixel threshold."
            )

        self.samples = [
            self.root
            / Path(
                str(
                    slice_path
                )
            ).name
            for slice_path in selected[
                "slice_path"
            ].tolist()
        ]

        missing_files = [
            path
            for path in self.samples
            if not path.is_file()
        ]

        if missing_files:
            preview = "\n".join(
                f"  {path}"
                for path in missing_files[
                    :10
                ]
            )

            raise ValueError(
                "Manifest-selected H5 file(s) are missing.\n\n"
                f"Missing: {len(missing_files):,}\n"
                f"{preview}"
            )

        # Keep aligned metadata for downstream inspection/debugging.
        self.sample_metadata = selected.reset_index(
            drop=True
        )

        print(
            f"Found {len(self.samples)} tumor-containing H5 slices."
        )

    def __len__(
        self,
    ) -> int:
        return len(
            self.samples
        )

    def __getitem__(
        self,
        idx: int,
    ) -> dict[str, torch.Tensor | str]:
        path = self.samples[
            idx
        ]

        x0, mask, cond = load_h5_full(
            path,
            image_channel=self.image_channel,
        )

        known = x0 * (
            1.0 - mask
        )

        # Exact notebook semantics: the donor patch is the real tumor
        # appearance from this same training slice.
        donor_patch = x0 * mask

        return {
            "x0": x0,
            "known": known,
            "mask": mask,
            "donor_patch": donor_patch,
            "cond": cond,
            "path": str(
                path
            ),
        }
