#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from downstream_evaluation.segmentation.feather_composition import (
    inner_feather_composite,
)
from src.inference.adaptation import (
    SUPPORTED_ADAPTATION_METHODS,
)


class DeterministicAdaptationSegmentationDataset(Dataset):
    """
    Segmentation dataset for a fixed deterministic PEFT synthetic library.

    The frozen 10,000-case BR-LoRA conditioning design is reused unchanged.
    Each deterministic synthetic case contains one model prediction together
    with the exact base image and transferred lesion mask used for synthesis.

    The downstream-training image can use either the deterministic model
    prediction directly or inner-only distance feathering with the stored
    base image and transferred lesion mask.
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

    REQUIRED_PAYLOAD_KEYS = (
        "adaptation_method",
        "prediction",
        "base_image",
        "transferred_mask",
    )

    def __init__(
        self,
        manifest_path: str | Path,
        library_root: str | Path,
        h5_root: str | Path,
        adaptation_method: str,
        composition_method: str,
        feather_width: int | None,
        transform=None,
    ) -> None:
        self.manifest_path = Path(
            manifest_path
        )
        self.library_root = Path(
            library_root
        )
        self.h5_root = Path(
            h5_root
        )
        self.adaptation_method = str(
            adaptation_method
        )
        self.composition_method = str(
            composition_method
        )
        self.feather_width = (
            None
            if feather_width is None
            else int(feather_width)
        )
        self.transform = transform

        if (
            self.adaptation_method
            not in SUPPORTED_ADAPTATION_METHODS
        ):
            raise ValueError(
                "Unsupported deterministic adaptation method: "
                f"{self.adaptation_method!r}"
            )

        if self.composition_method not in (
            "direct_prediction",
            "inner_only_distance_feather",
        ):
            raise ValueError(
                "Unsupported composition method: "
                f"{self.composition_method!r}"
            )

        if (
            self.composition_method
            == "inner_only_distance_feather"
        ):
            if (
                self.feather_width is None
                or self.feather_width <= 0
            ):
                raise ValueError(
                    "feather_width must be positive for "
                    "inner_only_distance_feather."
                )

        if not self.manifest_path.is_file():
            raise FileNotFoundError(
                "Synthetic manifest not found: "
                f"{self.manifest_path}"
            )

        if not self.library_root.is_dir():
            raise FileNotFoundError(
                "Deterministic adaptation library root not found: "
                f"{self.library_root}"
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
                + ", ".join(
                    missing_columns
                )
            )

        if self.manifest[
            "library_index"
        ].duplicated().any():
            raise ValueError(
                "Synthetic manifest contains duplicate "
                "library_index values."
            )

        self.manifest = (
            self.manifest
            .sort_values(
                "library_index"
            )
            .reset_index(
                drop=True
            )
        )

    def __len__(
        self,
    ) -> int:
        return len(
            self.manifest
        )

    @staticmethod
    def _case_directory(
        row: pd.Series,
    ) -> str:
        """
        Return the canonical deterministic-library case directory name.

        All production case directories use library_case_id.
        source_case_id is retained only as frozen-design provenance.
        """

        return str(
            row[
                "library_case_id"
            ]
        )

    def _payload_path(
        self,
        row: pd.Series,
    ) -> Path:
        return (
            self.library_root
            / str(
                row[
                    "batch_id"
                ]
            )
            / self._case_directory(
                row
            )
            / "synthesis_payload.pt"
        )

    def _donor_h5_path(
        self,
        row: pd.Series,
    ) -> Path:
        return (
            self.h5_root
            / str(
                row[
                    "donor_h5_file"
                ]
            )
        )

    @staticmethod
    def _require_image_tensor(
        value: object,
        *,
        name: str,
        path: Path,
    ) -> torch.Tensor:
        if not isinstance(
            value,
            torch.Tensor,
        ):
            raise TypeError(
                f"{name!r} in {path} must be a torch.Tensor."
            )

        tensor = (
            value
            .detach()
            .to(
                dtype=torch.float32
            )
        )

        if tensor.ndim == 4:
            if tensor.shape[
                0
            ] != 1:
                raise ValueError(
                    f"{name!r} in {path} has unexpected shape "
                    f"{tuple(tensor.shape)}."
                )

            tensor = tensor.squeeze(
                0
            )

        if tensor.shape != (
            1,
            240,
            240,
        ):
            raise ValueError(
                f"{name!r} in {path} must have shape "
                "(1, 240, 240); "
                f"observed {tuple(tensor.shape)}."
            )

        if not torch.isfinite(
            tensor
        ).all():
            raise ValueError(
                f"{name!r} in {path} contains non-finite values."
            )

        return tensor

    @staticmethod
    def _load_whole_tumor_mask(
        path: Path,
    ) -> torch.Tensor:
        if not path.is_file():
            raise FileNotFoundError(
                f"Donor H5 file not found: {path}"
            )

        with h5py.File(
            path,
            "r",
        ) as h5_file:
            if "mask" not in h5_file:
                raise KeyError(
                    f"Dataset 'mask' not found in {path}"
                )

            mask_array = np.asarray(
                h5_file[
                    "mask"
                ]
            )

        if mask_array.ndim != 3:
            raise ValueError(
                "Expected donor mask shape [H,W,C] in "
                f"{path}; got {mask_array.shape}."
            )

        whole_tumor = (
            mask_array.max(
                axis=-1
            )
            > 0
        ).astype(
            np.float32,
            copy=False,
        )

        if whole_tumor.shape != (
            240,
            240,
        ):
            raise ValueError(
                "Expected whole-tumor mask shape "
                f"(240, 240) in {path}; "
                f"got {whole_tumor.shape}."
            )

        return torch.from_numpy(
            whole_tumor
        ).unsqueeze(
            0
        )

    def __getitem__(
        self,
        index: int,
    ) -> dict[str, object]:
        row = self.manifest.iloc[
            index
        ]

        payload_path = self._payload_path(
            row
        )

        if not payload_path.is_file():
            raise FileNotFoundError(
                "Deterministic synthesis payload not found: "
                f"{payload_path}"
            )

        payload = torch.load(
            payload_path,
            map_location="cpu",
            weights_only=False,
        )

        if not isinstance(
            payload,
            dict,
        ):
            raise TypeError(
                "Expected deterministic synthesis payload to contain "
                f"a dictionary: {payload_path}"
            )

        missing_keys = [
            key
            for key in self.REQUIRED_PAYLOAD_KEYS
            if key not in payload
        ]

        if missing_keys:
            raise KeyError(
                "Deterministic synthesis payload is missing required "
                "key(s): "
                + ", ".join(
                    missing_keys
                )
                + f"\nPayload: {payload_path}"
            )

        observed_method = str(
            payload[
                "adaptation_method"
            ]
        )

        if (
            observed_method
            != self.adaptation_method
        ):
            raise ValueError(
                "Deterministic synthesis method mismatch.\n"
                f"Expected: {self.adaptation_method}\n"
                f"Observed: {observed_method}\n"
                f"Payload:  {payload_path}"
            )

        prediction = self._require_image_tensor(
            payload[
                "prediction"
            ],
            name="prediction",
            path=payload_path,
        )

        base_image = self._require_image_tensor(
            payload[
                "base_image"
            ],
            name="base_image",
            path=payload_path,
        )

        transferred_mask = self._require_image_tensor(
            payload[
                "transferred_mask"
            ],
            name="transferred_mask",
            path=payload_path,
        )

        if self.composition_method == "direct_prediction":
            image = prediction
        else:
            image = inner_feather_composite(
                prediction=prediction,
                base_image=base_image,
                transferred_mask=transferred_mask,
                width=self.feather_width,
            )

        donor_h5_path = self._donor_h5_path(
            row
        )

        mask = self._load_whole_tumor_mask(
            donor_h5_path
        )

        if not torch.equal(
            transferred_mask,
            mask,
        ):
            raise ValueError(
                "Stored transferred mask does not exactly match the "
                "donor whole-tumor mask for "
                f"{row['library_case_id']}."
            )

        expected_mask_pixels = int(
            row[
                "donor_mask_pixels"
            ]
        )

        observed_mask_pixels = int(
            mask.sum().item()
        )

        if (
            observed_mask_pixels
            != expected_mask_pixels
        ):
            raise ValueError(
                "Donor-mask pixel count mismatch for "
                f"{row['library_case_id']}: "
                f"manifest={expected_mask_pixels}, "
                f"observed={observed_mask_pixels}."
            )

        if self.transform is not None:
            transformed = self.transform(
                image=(
                    image
                    .squeeze(
                        0
                    )
                    .numpy()[
                        ...,
                        None,
                    ]
                ),
                mask=(
                    mask
                    .squeeze(
                        0
                    )
                    .numpy()
                ),
            )

            image = transformed[
                "image"
            ]

            mask = transformed[
                "mask"
            ]

            if mask.ndim == 2:
                mask = mask.unsqueeze(
                    0
                )

            image = image.to(
                dtype=torch.float32
            )

            mask = mask.to(
                dtype=torch.float32
            )

        return {
            "image": image,
            "mask": mask,
            "library_index": int(
                row[
                    "library_index"
                ]
            ),
            "library_case_id": str(
                row[
                    "library_case_id"
                ]
            ),
            "batch_id": str(
                row[
                    "batch_id"
                ]
            ),
            "adaptation_method": (
                self.adaptation_method
            ),
            "donor_volume": int(
                row[
                    "donor_volume"
                ]
            ),
            "donor_slice_index": int(
                row[
                    "donor_slice_index"
                ]
            ),
            "donor_h5_file": str(
                row[
                    "donor_h5_file"
                ]
            ),
            "synthesis_payload_path": str(
                payload_path
            ),
        }
