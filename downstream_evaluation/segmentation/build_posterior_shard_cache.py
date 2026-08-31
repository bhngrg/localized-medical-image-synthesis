#!/usr/bin/env python3

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch


POSTERIOR_SAMPLES = 100
DEFAULT_EPOCHS = 20
DEFAULT_SEED = 42
DEFAULT_SHARD_SIZE = 500


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a deterministic sharded runtime cache for downstream "
            "BR-LoRA posterior-sampling training."
        )
    )

    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Frozen 10,000-case BR-LoRA synthetic design manifest.",
    )
    parser.add_argument(
        "--library-root",
        type=Path,
        required=True,
        help="Original BR-LoRA library root containing posterior_samples.pt.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="New non-overwriting cache directory.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
    )
    parser.add_argument(
        "--shard-size",
        type=int,
        default=DEFAULT_SHARD_SIZE,
    )

    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def case_directory(row: pd.Series) -> str:
    source_case_id = row["source_case_id"]

    if pd.notna(source_case_id):
        return str(source_case_id)

    return str(row["library_case_id"])


def posterior_path(
    library_root: Path,
    row: pd.Series,
) -> Path:
    return (
        library_root
        / str(row["batch_id"])
        / case_directory(row)
        / "posterior_samples.pt"
    )


def main() -> None:
    args = parse_args()

    manifest_path = args.manifest.expanduser().resolve()
    library_root = args.library_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()

    seed = int(args.seed)
    epochs = int(args.epochs)
    shard_size = int(args.shard_size)

    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Manifest not found: {manifest_path}"
        )

    if not library_root.is_dir():
        raise FileNotFoundError(
            f"BR-LoRA library root not found: {library_root}"
        )

    if epochs < 1 or epochs > POSTERIOR_SAMPLES:
        raise ValueError(
            f"--epochs must be between 1 and {POSTERIOR_SAMPLES}."
        )

    if shard_size < 1:
        raise ValueError("--shard-size must be positive.")

    if output_root.exists():
        raise RuntimeError(
            "Refusing to overwrite existing cache directory:\n"
            f"{output_root}"
        )

    manifest = (
        pd.read_csv(manifest_path)
        .sort_values("library_index")
        .reset_index(drop=True)
    )

    required_columns = (
        "library_index",
        "library_case_id",
        "batch_id",
        "source_case_id",
    )

    missing = [
        column
        for column in required_columns
        if column not in manifest.columns
    ]

    if missing:
        raise ValueError(
            "Manifest is missing required columns: "
            + ", ".join(missing)
        )

    if manifest["library_index"].duplicated().any():
        raise ValueError(
            "Manifest contains duplicate library_index values."
        )

    if manifest.empty:
        raise ValueError(
            "Manifest contains no synthetic cases."
        )

    # Create the derived cache only after the source manifest has passed
    # its structural validation. This avoids leaving an empty cache
    # directory behind when the source contract is invalid.
    output_root.mkdir(
        parents=True,
        exist_ok=False,
    )

    n_cases = len(manifest)
    n_shards = (n_cases + shard_size - 1) // shard_size

    schedule = np.empty(
        (n_cases, epochs),
        dtype=np.int16,
    )

    for row_index, row in manifest.iterrows():
        library_index = int(row["library_index"])

        rng = np.random.default_rng(
            seed + library_index
        )

        schedule[row_index] = (
            rng.permutation(POSTERIOR_SAMPLES)[:epochs]
        )

    shard_records = []
    verified_shards = 0

    print("Posterior shard-cache build")
    print("=" * 78)
    print("Cases:", f"{n_cases:,}")
    print("Seed:", seed)
    print("Epochs:", epochs)
    print("Shard size:", shard_size)
    print("Shards per epoch:", n_shards)
    print("Output:", output_root)
    print()

    for shard_index in range(n_shards):
        start = shard_index * shard_size
        stop = min(
            start + shard_size,
            n_cases,
        )
        count = stop - start

        # Load each original posterior file once for this group of cases
        # and collect the exact realization selected for every epoch.
        shard_tensor = torch.empty(
            (
                epochs,
                count,
                1,
                240,
                240,
            ),
            dtype=torch.float32,
        )

        library_indices = []

        for local_index, row_index in enumerate(
            range(start, stop)
        ):
            row = manifest.iloc[row_index]
            library_index = int(
                row["library_index"]
            )

            source_path = posterior_path(
                library_root,
                row,
            )

            if not source_path.is_file():
                raise FileNotFoundError(
                    "Posterior-sample file not found:\n"
                    f"{source_path}"
                )

            obj = torch.load(
                source_path,
                map_location="cpu",
                mmap=True,
            )

            if not isinstance(obj, dict):
                raise TypeError(
                    f"Expected dict in {source_path}."
                )

            if "prediction_samples" not in obj:
                raise KeyError(
                    "'prediction_samples' missing from "
                    f"{source_path}"
                )

            samples = obj["prediction_samples"]

            expected_shape = (
                POSTERIOR_SAMPLES,
                1,
                1,
                240,
                240,
            )

            if tuple(samples.shape) != expected_shape:
                raise ValueError(
                    "Unexpected posterior tensor shape in "
                    f"{source_path}: {tuple(samples.shape)}"
                )

            for epoch in range(epochs):
                realization_index = int(
                    schedule[row_index, epoch]
                )

                selected = (
                    samples[realization_index]
                    .squeeze(0)
                    .detach()
                    .to(dtype=torch.float32)
                )

                if selected.shape != (1, 240, 240):
                    raise ValueError(
                        "Unexpected selected image shape for "
                        f"library_index={library_index}, "
                        f"epoch={epoch}."
                    )

                if not torch.isfinite(selected).all():
                    raise ValueError(
                        "Non-finite selected realization for "
                        f"library_index={library_index}, "
                        f"epoch={epoch}."
                    )

                shard_tensor[
                    epoch,
                    local_index,
                ].copy_(selected)

            library_indices.append(
                library_index
            )

        # Save one file per epoch for this shard. Training therefore
        # reads only the current epoch's selected realizations.
        for epoch in range(epochs):
            epoch_dir = (
                output_root
                / f"epoch_{epoch:02d}"
            )
            epoch_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            shard_path = (
                epoch_dir
                / f"shard_{shard_index:04d}.pt"
            )

            payload = {
                "prediction_samples": (
                    shard_tensor[epoch].clone()
                ),
                "seed": seed,
                "epoch": epoch,
                "shard_index": shard_index,
                "start_position": start,
                "stop_position": stop,
                "library_indices": torch.tensor(
                    library_indices,
                    dtype=torch.int64,
                ),
                "original_realization_indices": torch.tensor(
                    schedule[start:stop, epoch],
                    dtype=torch.int16,
                ),
            }

            torch.save(
                payload,
                shard_path,
            )

            # Reload the written shard and require exact equality with
            # the selected source-derived tensor before recording it
            # as verified cache output.
            written = torch.load(
                shard_path,
                map_location="cpu",
                mmap=True,
            )

            if not torch.equal(
                written["prediction_samples"],
                shard_tensor[epoch],
            ):
                raise RuntimeError(
                    "Post-write equality verification failed for "
                    f"{shard_path}"
                )

            if int(written["seed"]) != seed:
                raise RuntimeError(
                    f"Seed metadata mismatch in {shard_path}"
                )

            if int(written["epoch"]) != epoch:
                raise RuntimeError(
                    f"Epoch metadata mismatch in {shard_path}"
                )

            if not torch.equal(
                written["library_indices"],
                torch.tensor(
                    library_indices,
                    dtype=torch.int64,
                ),
            ):
                raise RuntimeError(
                    "Library-index metadata mismatch in "
                    f"{shard_path}"
                )

            if not torch.equal(
                written["original_realization_indices"],
                torch.tensor(
                    schedule[start:stop, epoch],
                    dtype=torch.int16,
                ),
            ):
                raise RuntimeError(
                    "Posterior-realization metadata mismatch in "
                    f"{shard_path}"
                )

            shard_sha256 = sha256_file(
                shard_path
            )
            verified_shards += 1

            shard_records.append(
                {
                    "epoch": epoch,
                    "shard_index": shard_index,
                    "start_position": start,
                    "stop_position": stop,
                    "cases": count,
                    "path": str(
                        shard_path.relative_to(
                            output_root
                        )
                    ),
                    "size_bytes": (
                        shard_path.stat().st_size
                    ),
                    "sha256": shard_sha256,
                    "verified_exact": True,
                }
            )

        print(
            f"Built shard {shard_index + 1:03d}/"
            f"{n_shards:03d} "
            f"(manifest positions {start}:{stop})"
        )

    cache_manifest = {
        "schema_version": 1,
        "cache_type": (
            "downstream_br_lora_posterior_epoch_shards"
        ),
        "created_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "scientific_contract": {
            "note": (
                "Storage-only optimization. Posterior realization "
                "selection is identical to the original deterministic "
                "case-specific schedule."
            ),
            "seed": seed,
            "epochs": epochs,
            "posterior_samples_available": (
                POSTERIOR_SAMPLES
            ),
            "schedule_rule": (
                "numpy.default_rng(seed + library_index)"
                ".permutation(100)[:epochs]"
            ),
        },
        "source": {
            "manifest": str(manifest_path),
            "manifest_sha256": sha256_file(
                manifest_path
            ),
            "library_root": str(library_root),
        },
        "layout": {
            "cases": n_cases,
            "shard_size": shard_size,
            "shards_per_epoch": n_shards,
            "tensor_shape_per_shard": (
                "[cases_in_shard, 1, 240, 240]"
            ),
            "dtype": "float32",
        },
        "verification": {
            "method": (
                "Post-write torch.equal against the exact "
                "source-derived selected tensor, with shard SHA-256."
            ),
            "verified_shards": verified_shards,
            "expected_shards": n_shards * epochs,
            "all_shards_verified": (
                verified_shards == n_shards * epochs
            ),
        },
        "shards": shard_records,
    }

    manifest_output = (
        output_root
        / "cache_manifest.json"
    )

    with manifest_output.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            cache_manifest,
            file,
            indent=2,
            sort_keys=True,
        )
        file.write("\n")

    print()
    print("Cache build complete.")
    print("Manifest:", manifest_output)
    print(
        "Total shard files:",
        len(shard_records),
    )


if __name__ == "__main__":
    main()
