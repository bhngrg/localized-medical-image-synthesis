#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from downstream_evaluation.segmentation.posterior_sample_dataset import (
    BRLoRAPosteriorSampleSegmentationDataset,
)


class BRLoRAPosteriorShardSegmentationDataset(
    BRLoRAPosteriorSampleSegmentationDataset
):
    """
    Cache-backed form of the deterministic BR-LoRA posterior dataset.

    The accepted per-case posterior library remains the scientific source of
    truth. This dataset reads exact source-derived realizations from a verified
    epoch/shard cache while preserving the original case-specific schedule,
    donor masks, transforms, and returned sample metadata.
    """

    CACHE_TYPE = "downstream_br_lora_posterior_epoch_shards"
    CACHE_SCHEMA_VERSION = 1
    LOADER_MODE = "posterior_shard_cache"

    def __init__(
        self,
        manifest_path: str | Path,
        library_root: str | Path,
        h5_root: str | Path,
        cache_root: str | Path,
        seed: int,
        expected_epochs: int,
        transform=None,
    ) -> None:
        super().__init__(
            manifest_path=manifest_path,
            library_root=library_root,
            h5_root=h5_root,
            seed=seed,
            transform=transform,
        )

        self.cache_root = Path(
            cache_root
        ).expanduser().resolve()
        self.expected_epochs = int(
            expected_epochs
        )

        if self.expected_epochs < 1:
            raise ValueError(
                "expected_epochs must be positive."
            )

        if self.expected_epochs > self.POSTERIOR_SAMPLES:
            raise ValueError(
                "expected_epochs exceeds the available posterior samples."
            )

        if not self.cache_root.is_dir():
            raise FileNotFoundError(
                "Posterior shard-cache root not found: "
                f"{self.cache_root}"
            )

        self.cache_manifest_path = (
            self.cache_root / "cache_manifest.json"
        )

        if not self.cache_manifest_path.is_file():
            raise FileNotFoundError(
                "Posterior shard-cache manifest not found: "
                f"{self.cache_manifest_path}"
            )

        self.cache_manifest_sha256 = self._sha256_file(
            self.cache_manifest_path
        )

        with self.cache_manifest_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            self.cache_manifest = json.load(file)

        self._validate_cache_manifest()

        self._cached_epoch = None
        self._shard_cache: dict[int, dict[str, object]] = {}

    @staticmethod
    def _sha256_file(
        path: Path,
    ) -> str:
        digest = hashlib.sha256()

        with path.open("rb") as file:
            for chunk in iter(
                lambda: file.read(1024 * 1024),
                b"",
            ):
                digest.update(chunk)

        return digest.hexdigest()

    def _validate_cache_manifest(
        self,
    ) -> None:
        payload = self.cache_manifest

        if int(payload.get("schema_version", -1)) != self.CACHE_SCHEMA_VERSION:
            raise ValueError(
                "Unsupported posterior shard-cache schema version."
            )

        if payload.get("cache_type") != self.CACHE_TYPE:
            raise ValueError(
                "Unexpected posterior shard-cache type."
            )

        scientific = payload.get(
            "scientific_contract",
            {},
        )

        if int(scientific.get("seed", -1)) != self.seed:
            raise ValueError(
                "Posterior shard-cache seed does not match the "
                "downstream training seed."
            )

        cache_epochs = int(
            scientific.get(
                "epochs",
                -1,
            )
        )

        if cache_epochs < self.expected_epochs:
            raise ValueError(
                "Posterior shard cache does not contain enough epochs "
                f"for training: cache={cache_epochs}, "
                f"requested={self.expected_epochs}."
            )

        if int(
            scientific.get(
                "posterior_samples_available",
                -1,
            )
        ) != self.POSTERIOR_SAMPLES:
            raise ValueError(
                "Posterior shard-cache sample count does not match "
                "the downstream posterior contract."
            )

        source = payload.get(
            "source",
            {},
        )

        expected_manifest_sha256 = self._sha256_file(
            self.manifest_path
        )

        if (
            source.get("manifest_sha256")
            != expected_manifest_sha256
        ):
            raise ValueError(
                "Posterior shard-cache source manifest SHA-256 does "
                "not match the configured synthetic manifest."
            )

        layout = payload.get(
            "layout",
            {},
        )

        if int(layout.get("cases", -1)) != len(self.manifest):
            raise ValueError(
                "Posterior shard-cache case count does not match "
                "the synthetic manifest."
            )

        self.shard_size = int(
            layout.get(
                "shard_size",
                -1,
            )
        )
        self.shards_per_epoch = int(
            layout.get(
                "shards_per_epoch",
                -1,
            )
        )

        if self.shard_size < 1 or self.shards_per_epoch < 1:
            raise ValueError(
                "Posterior shard-cache layout is invalid."
            )

        expected_shards_per_epoch = (
            len(self.manifest)
            + self.shard_size
            - 1
        ) // self.shard_size

        if self.shards_per_epoch != expected_shards_per_epoch:
            raise ValueError(
                "Posterior shard-cache shard count does not match "
                "the manifest and shard size."
            )

        verification = payload.get(
            "verification",
            {},
        )

        if not bool(
            verification.get(
                "all_shards_verified",
                False,
            )
        ):
            raise ValueError(
                "Posterior shard-cache manifest does not report "
                "complete exact verification."
            )

        cache_expected_shards = (
            self.shards_per_epoch
            * cache_epochs
        )

        if int(
            verification.get(
                "expected_shards",
                -1,
            )
        ) != cache_expected_shards:
            raise ValueError(
                "Posterior shard-cache verification count is inconsistent."
            )

        if int(
            verification.get(
                "verified_shards",
                -1,
            )
        ) != cache_expected_shards:
            raise ValueError(
                "Posterior shard-cache is not fully verified."
            )

        records = payload.get(
            "shards",
            [],
        )

        self._shard_records: dict[
            tuple[int, int],
            dict[str, object],
        ] = {}

        for record in records:
            epoch = int(
                record.get(
                    "epoch",
                    -1,
                )
            )
            shard_index = int(
                record.get(
                    "shard_index",
                    -1,
                )
            )

            key = (
                epoch,
                shard_index,
            )

            if key in self._shard_records:
                raise ValueError(
                    "Posterior shard-cache manifest contains duplicate "
                    f"record for epoch={epoch}, shard={shard_index}."
                )

            if not bool(
                record.get(
                    "verified_exact",
                    False,
                )
            ):
                raise ValueError(
                    "Posterior shard-cache contains an unverified shard "
                    f"record: epoch={epoch}, shard={shard_index}."
                )

            self._shard_records[key] = record

        for epoch in range(
            self.expected_epochs
        ):
            for shard_index in range(
                self.shards_per_epoch
            ):
                key = (
                    epoch,
                    shard_index,
                )

                if key not in self._shard_records:
                    raise ValueError(
                        "Posterior shard-cache manifest is missing "
                        f"epoch={epoch}, shard={shard_index}."
                    )

                record = self._shard_records[key]
                shard_path = (
                    self.cache_root
                    / str(record["path"])
                )

                if not shard_path.is_file():
                    raise FileNotFoundError(
                        "Posterior shard-cache file not found: "
                        f"{shard_path}"
                    )

    @property
    def loader_mode(
        self,
    ) -> str:
        return self.LOADER_MODE

    def set_epoch(
        self,
        epoch: int,
    ) -> None:
        super().set_epoch(epoch)

        if self.epoch >= self.expected_epochs:
            raise ValueError(
                f"Epoch {self.epoch} is not present in this cache."
            )

        self._cached_epoch = None
        self._shard_cache = {}

    def _load_shard(
        self,
        shard_index: int,
    ) -> dict[str, object]:
        if self._cached_epoch != self.epoch:
            self._cached_epoch = self.epoch
            self._shard_cache = {}

        if shard_index in self._shard_cache:
            return self._shard_cache[
                shard_index
            ]

        key = (
            self.epoch,
            shard_index,
        )
        record = self._shard_records[
            key
        ]
        shard_path = (
            self.cache_root
            / str(record["path"])
        )

        obj = torch.load(
            shard_path,
            map_location="cpu",
            mmap=True,
        )

        if not isinstance(obj, dict):
            raise TypeError(
                f"Expected dict in cache shard {shard_path}."
            )

        expected_start = (
            shard_index
            * self.shard_size
        )
        expected_stop = min(
            expected_start + self.shard_size,
            len(self.manifest),
        )
        expected_count = (
            expected_stop - expected_start
        )

        if int(obj.get("seed", -1)) != self.seed:
            raise ValueError(
                f"Seed mismatch in cache shard {shard_path}."
            )

        if int(obj.get("epoch", -1)) != self.epoch:
            raise ValueError(
                f"Epoch mismatch in cache shard {shard_path}."
            )

        if int(
            obj.get(
                "shard_index",
                -1,
            )
        ) != shard_index:
            raise ValueError(
                f"Shard-index mismatch in cache shard {shard_path}."
            )

        if int(
            obj.get(
                "start_position",
                -1,
            )
        ) != expected_start:
            raise ValueError(
                f"Start-position mismatch in cache shard {shard_path}."
            )

        if int(
            obj.get(
                "stop_position",
                -1,
            )
        ) != expected_stop:
            raise ValueError(
                f"Stop-position mismatch in cache shard {shard_path}."
            )

        samples = obj.get(
            "prediction_samples"
        )

        if not isinstance(
            samples,
            torch.Tensor,
        ):
            raise TypeError(
                f"prediction_samples is not a tensor in {shard_path}."
            )

        expected_shape = (
            expected_count,
            1,
            240,
            240,
        )

        if tuple(samples.shape) != expected_shape:
            raise ValueError(
                "Unexpected cache shard tensor shape in "
                f"{shard_path}: {tuple(samples.shape)}."
            )

        if samples.dtype != torch.float32:
            raise ValueError(
                f"Unexpected cache shard dtype in {shard_path}: "
                f"{samples.dtype}."
            )

        expected_library_indices = torch.tensor(
            self.manifest.iloc[
                expected_start:expected_stop
            ]["library_index"].astype(int).to_numpy(),
            dtype=torch.int64,
        )

        stored_library_indices = obj.get(
            "library_indices"
        )

        if not isinstance(
            stored_library_indices,
            torch.Tensor,
        ) or not torch.equal(
            stored_library_indices,
            expected_library_indices,
        ):
            raise ValueError(
                "Library-index metadata mismatch in cache shard "
                f"{shard_path}."
            )

        expected_realizations = torch.tensor(
            self.realization_schedule[
                expected_start:expected_stop,
                self.epoch,
            ],
            dtype=torch.int16,
        )

        stored_realizations = obj.get(
            "original_realization_indices"
        )

        if not isinstance(
            stored_realizations,
            torch.Tensor,
        ) or not torch.equal(
            stored_realizations,
            expected_realizations,
        ):
            raise ValueError(
                "Posterior-realization metadata mismatch in cache shard "
                f"{shard_path}."
            )

        self._shard_cache[
            shard_index
        ] = obj

        return obj

    def __getitem__(
        self,
        index: int,
    ) -> dict[str, object]:
        row = self.manifest.iloc[
            index
        ]

        realization_index = int(
            self.realization_schedule[
                index,
                self.epoch,
            ]
        )

        shard_index = (
            index // self.shard_size
        )
        local_index = (
            index % self.shard_size
        )

        shard_obj = self._load_shard(
            shard_index
        )
        samples = shard_obj[
            "prediction_samples"
        ]

        image = (
            samples[local_index]
            .detach()
            .clone()
            .to(dtype=torch.float32)
        )

        if image.shape != (1, 240, 240):
            raise ValueError(
                "Expected cached selected image shape (1, 240, 240); "
                f"got {tuple(image.shape)}."
            )

        if not torch.isfinite(image).all():
            raise ValueError(
                "Cached posterior realization contains non-finite values."
            )

        donor_h5_path = self._donor_h5_path(
            row
        )
        mask = self._load_whole_tumor_mask(
            donor_h5_path
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

        original_path = self._posterior_samples_path(
            row
        )
        shard_record = self._shard_records[
            (
                self.epoch,
                shard_index,
            )
        ]
        shard_path = (
            self.cache_root
            / str(shard_record["path"])
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
                original_path
            ),
            "posterior_cache_shard_path": str(
                shard_path
            ),
        }
