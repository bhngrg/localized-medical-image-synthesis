#!/usr/bin/env python3

"""
Evaluate downstream tumor-segmentation checkpoints on UCSF-PDGM.

The external cohort is defined by the frozen tracked 202-subject manifest.
Machine-specific UCSF-PDGM data paths are resolved using the repository's
standard folders.yaml convention.

Path precedence
---------------
UCSF-PDGM data root:
    CLI > data/folders.yaml

Frozen UCSF-PDGM manifest:
    CLI > data/folders.yaml > tracked repository default

Checkpoint paths are deliberately run-specific. They must be supplied
explicitly on the command line and are not written to folders.yaml.

Scientific behavior
-------------------
This hardened interface preserves the preprocessing and evaluation rules used
for the preliminary UCSF-PDGM analysis:

* FLAIR input only.
* Per-slice standardization followed by repository image normalization.
* Whole-tumor target defined as segmentation > 0.
* Checkpoint-specific probability threshold, defaulting to 0.5 when absent.
* Empty-prediction/empty-target slices receive Dice = IoU = 1.
* Slice-level Dice/IoU are reported for all slices and tumor-positive slices.
* Subject-level volumetric Dice/IoU are computed from the full 3D prediction
  and target volumes.

Provenance
----------
The downstream segmentation idea and vanilla U-Net structure were adapted in
part from:

    https://github.com/edaaydinea/Low-Grade-Glioma-Segmentation

The present implementation was rewritten for the BraTS/UCSF-PDGM setting and
uses this repository's preprocessing, frozen manifests, loss/metric
definitions, and experiment workflow.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

import nibabel as nib
import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from downstream_evaluation.segmentation.model import VanillaUNet
from src.config import (
    load_folders_config,
    resolve_path,
    save_folders_config,
)
from src.data.preprocessing import (
    normalize_image_channel,
    standardize_nifti_slice,
)


DEFAULT_MANIFEST = (
    PROJECT_ROOT
    / "downstream_evaluation"
    / "manifests"
    / "ucsf_pdgm_external_202_subjects.csv"
)

EXPERIMENT_NAMES = (
    "real_only",
    "real_plus_br_lora_posterior_mean",
    "real_plus_br_lora_posterior_sampling",
)

# Fixed UCSF-PDGM/BraTS-compatible data contract, not a tunable experiment
# parameter.
EXPECTED_SUBJECTS = 202
EXPECTED_SHAPE = (240, 240, 155)

# Preliminary evaluation default; exposed through CLI for user control.
DEFAULT_BATCH_SIZE = 26


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate downstream segmentation checkpoints on the frozen "
            "202-subject UCSF-PDGM external-validation cohort."
        )
    )

    parser.add_argument(
        "--folders-file",
        type=Path,
        default=Path("data/folders.yaml"),
        help="Machine-specific path configuration YAML.",
    )

    parser.add_argument(
        "--ucsf-pdgm-root",
        type=Path,
        default=None,
        help=(
            "Root containing the downloaded UCSF-PDGM NIfTI subject "
            "directories. Overrides ucsf_pdgm_root in --folders-file."
        ),
    )

    parser.add_argument(
        "--ucsf-pdgm-manifest",
        type=Path,
        default=None,
        help=(
            "Frozen UCSF-PDGM subject manifest. CLI overrides folders YAML; "
            "otherwise the tracked repository manifest is used."
        ),
    )

    parser.add_argument(
        "--real-only-checkpoint",
        type=Path,
        required=True,
        help="Checkpoint trained using real BraTS data only.",
    )

    parser.add_argument(
        "--posterior-mean-checkpoint",
        type=Path,
        required=True,
        help=(
            "Checkpoint trained using real BraTS data plus BR-LoRA "
            "posterior-mean synthetic images."
        ),
    )

    parser.add_argument(
        "--posterior-sampling-checkpoint",
        type=Path,
        required=True,
        help=(
            "Checkpoint trained using real BraTS data plus BR-LoRA "
            "posterior-sampled synthetic images."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Evaluation output directory. If omitted, a non-overwriting "
            "path under outputs/downstream_segmentation/evaluations is "
            "generated."
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=(
            "Inference batch size. Default: "
            f"{DEFAULT_BATCH_SIZE}, matching the preliminary evaluation."
        ),
    )

    parser.add_argument(
        "--device",
        choices=[
            "auto",
            "cpu",
            "mps",
            "cuda",
        ],
        default="auto",
        help="Inference device.",
    )

    parser.add_argument(
        "--validate-only",
        action="store_true",
        help=(
            "Validate paths, the frozen cohort contract, and checkpoint "
            "compatibility without running inference or creating outputs."
        ),
    )

    return parser.parse_args()


def resolve_device(
    requested: str,
) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but is not available."
            )
        return torch.device("cuda")

    if requested == "mps":
        if (
            not hasattr(torch.backends, "mps")
            or not torch.backends.mps.is_available()
        ):
            raise RuntimeError(
                "MPS was requested but is not available."
            )
        return torch.device("mps")

    if requested == "cpu":
        return torch.device("cpu")

    if torch.cuda.is_available():
        return torch.device("cuda")

    if (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        return torch.device("mps")

    return torch.device("cpu")


def resolve_optional_repo_path(
    *,
    key: str,
    cli_value: Path | None,
    folders_config: dict[str, str],
    default: Path,
) -> Path:
    """
    Resolve CLI > folders YAML > tracked repository default.

    CLI overrides are persisted to the machine-specific folders mapping.
    """
    if cli_value is not None:
        path = Path(cli_value)
        folders_config[key] = str(path)
        return path

    configured = folders_config.get(key)

    if configured:
        return Path(configured)

    return default


def resolve_paths(
    args: argparse.Namespace,
) -> tuple[Path, Path]:
    folders_config = load_folders_config(
        args.folders_file
    )

    raw_root = Path(
        resolve_path(
            key="ucsf_pdgm_root",
            cli_value=args.ucsf_pdgm_root,
            config=folders_config,
            selector=None,
        )
    )

    manifest_path = resolve_optional_repo_path(
        key="ucsf_pdgm_manifest",
        cli_value=args.ucsf_pdgm_manifest,
        folders_config=folders_config,
        default=DEFAULT_MANIFEST,
    )

    save_folders_config(
        args.folders_file,
        folders_config,
    )

    return raw_root, Path(manifest_path)


def resolve_checkpoints(
    args: argparse.Namespace,
) -> dict[str, Path]:
    return {
        "real_only": args.real_only_checkpoint.expanduser().resolve(),
        "real_plus_br_lora_posterior_mean": (
            args.posterior_mean_checkpoint.expanduser().resolve()
        ),
        "real_plus_br_lora_posterior_sampling": (
            args.posterior_sampling_checkpoint.expanduser().resolve()
        ),
    }


def sha256_file(
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


def git_commit() -> str | None:
    try:
        result = subprocess.run(
            [
                "git",
                "rev-parse",
                "HEAD",
            ],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (
        OSError,
        subprocess.CalledProcessError,
    ):
        return None

    return result.stdout.strip()


def package_version(
    name: str,
) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def display_path(
    path: Path,
) -> str:
    resolved = path.expanduser().resolve()

    try:
        return str(
            resolved.relative_to(PROJECT_ROOT)
        )
    except ValueError:
        return str(resolved)


def default_output_dir() -> Path:
    partition = os.environ.get(
        "SLURM_JOB_PARTITION",
        "local",
    )

    job_id = os.environ.get(
        "SLURM_JOB_ID"
    )

    if job_id:
        run_id = f"job_{job_id}"
    else:
        timestamp = datetime.now(
            timezone.utc
        ).strftime("%Y%m%dT%H%M%SZ")
        run_id = f"run_{timestamp}"

    return (
        PROJECT_ROOT
        / "outputs"
        / "downstream_segmentation"
        / "evaluations"
        / "ucsf_pdgm_external_202"
        / f"{partition}_{run_id}"
    )


def prepare_output_directory(
    path: Path,
) -> Path:
    path = path.expanduser().resolve()

    if path.exists():
        if not path.is_dir():
            raise ValueError(
                f"Output path exists and is not a directory: {path}"
            )

        if any(path.iterdir()):
            raise RuntimeError(
                "Refusing to overwrite a non-empty output directory:\n"
                f"{path}"
            )

    path.mkdir(
        parents=True,
        exist_ok=True,
    )

    return path


def validate_file(
    path: Path,
    description: str,
) -> None:
    if not path.is_file():
        raise FileNotFoundError(
            f"{description} not found: {path}"
        )


def load_manifest(
    manifest_path: Path,
) -> list[dict[str, str]]:
    with manifest_path.open(
        newline="",
        encoding="utf-8",
    ) as file:
        rows = list(
            csv.DictReader(file)
        )

    if len(rows) != EXPECTED_SUBJECTS:
        raise RuntimeError(
            f"Expected {EXPECTED_SUBJECTS} UCSF-PDGM subjects, "
            f"found {len(rows)}."
        )

    required_columns = {
        "subject_id",
        "flair_relative_path",
        "segmentation_relative_path",
    }

    if not rows:
        raise RuntimeError(
            "UCSF-PDGM manifest is empty."
        )

    missing_columns = (
        required_columns
        - set(rows[0].keys())
    )

    if missing_columns:
        raise RuntimeError(
            "UCSF-PDGM manifest is missing required columns: "
            + ", ".join(
                sorted(missing_columns)
            )
        )

    subject_ids = [
        row["subject_id"]
        for row in rows
    ]

    if any(
        not subject_id
        for subject_id in subject_ids
    ):
        raise RuntimeError(
            "UCSF-PDGM manifest contains an empty subject_id."
        )

    if len(set(subject_ids)) != len(subject_ids):
        raise RuntimeError(
            "UCSF-PDGM manifest contains duplicate subject IDs."
        )

    return rows


def validate_external_dataset_paths(
    manifest_rows: list[dict[str, str]],
    raw_root: Path,
) -> None:
    if not raw_root.is_dir():
        raise FileNotFoundError(
            f"UCSF-PDGM root not found: {raw_root}"
        )

    for row in manifest_rows:
        subject_id = row["subject_id"]

        flair_path = (
            raw_root
            / row["flair_relative_path"]
        )

        mask_path = (
            raw_root
            / row["segmentation_relative_path"]
        )

        if not flair_path.is_file():
            raise FileNotFoundError(
                f"{subject_id} FLAIR not found: {flair_path}"
            )

        if not mask_path.is_file():
            raise FileNotFoundError(
                f"{subject_id} segmentation not found: {mask_path}"
            )

        if "whole_tumor_rule" in row:
            rule = row["whole_tumor_rule"].strip()

            if rule and rule != "segmentation > 0":
                raise RuntimeError(
                    f"{subject_id} has unexpected whole_tumor_rule "
                    f"{rule!r}; expected 'segmentation > 0'."
                )


def validate_checkpoint(
    checkpoint_path: Path,
    experiment_name: str,
) -> dict:
    validate_file(
        checkpoint_path,
        f"{experiment_name} checkpoint",
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
    )

    required_keys = {
        "model_state_dict",
        "epoch",
    }

    missing = (
        required_keys
        - set(checkpoint.keys())
    )

    if missing:
        raise RuntimeError(
            f"{experiment_name} checkpoint is missing required keys: "
            + ", ".join(
                sorted(missing)
            )
        )

    threshold = float(
        checkpoint.get(
            "threshold",
            0.5,
        )
    )

    if not np.isfinite(threshold):
        raise RuntimeError(
            f"{experiment_name} checkpoint has a non-finite threshold."
        )

    if not 0.0 <= threshold <= 1.0:
        raise RuntimeError(
            f"{experiment_name} checkpoint threshold must be in [0, 1], "
            f"found {threshold}."
        )

    model = VanillaUNet(
        in_channels=1,
        out_channels=1,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    return checkpoint


def prepare_subject(
    row: dict[str, str],
    raw_root: Path,
) -> tuple[np.ndarray, np.ndarray]:
    flair_path = (
        raw_root
        / row["flair_relative_path"]
    )

    mask_path = (
        raw_root
        / row["segmentation_relative_path"]
    )

    flair = np.asarray(
        nib.load(flair_path).dataobj,
        dtype=np.float64,
    )

    segmentation = np.asarray(
        nib.load(mask_path).dataobj,
    )

    if flair.shape != EXPECTED_SHAPE:
        raise RuntimeError(
            f"{row['subject_id']} FLAIR shape {flair.shape} "
            f"does not match {EXPECTED_SHAPE}."
        )

    if segmentation.shape != EXPECTED_SHAPE:
        raise RuntimeError(
            f"{row['subject_id']} segmentation shape "
            f"{segmentation.shape} does not match {EXPECTED_SHAPE}."
        )

    images = np.empty(
        (
            EXPECTED_SHAPE[2],
            1,
            EXPECTED_SHAPE[0],
            EXPECTED_SHAPE[1],
        ),
        dtype=np.float32,
    )

    masks = np.empty_like(
        images,
        dtype=np.float32,
    )

    for z in range(
        EXPECTED_SHAPE[2]
    ):
        normalized = normalize_image_channel(
            standardize_nifti_slice(
                flair[:, :, z]
            )
        )

        images[z, 0] = normalized

        masks[z, 0] = (
            segmentation[:, :, z] > 0
        ).astype(
            np.float32
        )

    if not np.isfinite(images).all():
        raise RuntimeError(
            f"{row['subject_id']} contains non-finite model inputs."
        )

    return images, masks


def per_slice_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    predictions = (
        torch.sigmoid(logits)
        >= threshold
    ).float()

    dims = (
        1,
        2,
        3,
    )

    intersection = (
        predictions
        * targets
    ).sum(
        dim=dims
    )

    prediction_sum = predictions.sum(
        dim=dims
    )

    target_sum = targets.sum(
        dim=dims
    )

    denominator = (
        prediction_sum
        + target_sum
    )

    dice = torch.where(
        denominator > 0,
        2.0
        * intersection
        / denominator,
        torch.ones_like(
            denominator
        ),
    )

    union = (
        predictions
        + targets
        - predictions
        * targets
    ).sum(
        dim=dims
    )

    iou = torch.where(
        union > 0,
        intersection
        / union,
        torch.ones_like(
            union
        ),
    )

    return dice, iou


def evaluate_checkpoint(
    *,
    experiment_name: str,
    checkpoint_path: Path,
    manifest_rows: list[dict[str, str]],
    raw_root: Path,
    output_root: Path,
    batch_size: int,
    device: torch.device,
) -> dict:
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
    )

    threshold = float(
        checkpoint.get(
            "threshold",
            0.5,
        )
    )

    model = VanillaUNet(
        in_channels=1,
        out_channels=1,
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model = model.to(
        device
    )

    model.eval()

    slice_rows = []
    subject_rows = []

    all_dice = []
    all_iou = []
    positive_dice = []
    positive_iou = []

    for subject_index, row in enumerate(
        manifest_rows,
        start=1,
    ):
        subject_id = row[
            "subject_id"
        ]

        images_np, masks_np = prepare_subject(
            row,
            raw_root,
        )

        subject_dice = []
        subject_iou = []
        subject_positive_dice = []
        subject_positive_iou = []

        subject_intersection = 0.0
        subject_prediction_sum = 0.0
        subject_target_sum = 0.0

        for start in range(
            0,
            EXPECTED_SHAPE[2],
            batch_size,
        ):
            stop = min(
                start
                + batch_size,
                EXPECTED_SHAPE[2],
            )

            images = torch.from_numpy(
                images_np[
                    start:stop
                ]
            ).to(
                device,
                non_blocking=True,
            )

            masks = torch.from_numpy(
                masks_np[
                    start:stop
                ]
            ).to(
                device,
                non_blocking=True,
            )

            with torch.no_grad():
                logits = model(
                    images
                )

                dice, iou = per_slice_metrics(
                    logits,
                    masks,
                    threshold,
                )

                predictions = (
                    torch.sigmoid(
                        logits
                    )
                    >= threshold
                ).float()

                subject_intersection += float(
                    (
                        predictions
                        * masks
                    ).sum().item()
                )

                subject_prediction_sum += float(
                    predictions.sum().item()
                )

                subject_target_sum += float(
                    masks.sum().item()
                )

            dice_cpu = (
                dice.cpu().numpy()
            )

            iou_cpu = (
                iou.cpu().numpy()
            )

            positive_cpu = (
                masks.sum(
                    dim=(
                        1,
                        2,
                        3,
                    )
                )
                > 0
            ).cpu().numpy()

            for local_index in range(
                stop - start
            ):
                z = (
                    start
                    + local_index
                )

                d = float(
                    dice_cpu[
                        local_index
                    ]
                )

                j = float(
                    iou_cpu[
                        local_index
                    ]
                )

                is_positive = bool(
                    positive_cpu[
                        local_index
                    ]
                )

                slice_rows.append(
                    {
                        "experiment": experiment_name,
                        "subject_id": subject_id,
                        "slice_index": z,
                        "tumor_positive": is_positive,
                        "dice": d,
                        "iou": j,
                    }
                )

                all_dice.append(
                    d
                )

                all_iou.append(
                    j
                )

                subject_dice.append(
                    d
                )

                subject_iou.append(
                    j
                )

                if is_positive:
                    positive_dice.append(
                        d
                    )

                    positive_iou.append(
                        j
                    )

                    subject_positive_dice.append(
                        d
                    )

                    subject_positive_iou.append(
                        j
                    )

        subject_denominator = (
            subject_prediction_sum
            + subject_target_sum
        )

        if subject_denominator > 0:
            subject_volume_dice = (
                2.0
                * subject_intersection
                / subject_denominator
            )
        else:
            subject_volume_dice = 1.0

        subject_union = (
            subject_prediction_sum
            + subject_target_sum
            - subject_intersection
        )

        if subject_union > 0:
            subject_volume_iou = (
                subject_intersection
                / subject_union
            )
        else:
            subject_volume_iou = 1.0

        subject_rows.append(
            {
                "experiment": experiment_name,
                "subject_id": subject_id,
                "n_slices": len(
                    subject_dice
                ),
                "n_positive_slices": len(
                    subject_positive_dice
                ),
                "volumetric_dice": float(
                    subject_volume_dice
                ),
                "volumetric_iou": float(
                    subject_volume_iou
                ),
                "mean_dice_all_slices": float(
                    np.mean(
                        subject_dice
                    )
                ),
                "mean_iou_all_slices": float(
                    np.mean(
                        subject_iou
                    )
                ),
                "mean_dice_positive_slices": float(
                    np.mean(
                        subject_positive_dice
                    )
                ),
                "mean_iou_positive_slices": float(
                    np.mean(
                        subject_positive_iou
                    )
                ),
            }
        )

        print(
            f"[{experiment_name}] "
            f"{subject_index:03d}/{len(manifest_rows)} "
            f"{subject_id} "
            f"positive_slices={len(subject_positive_dice)}"
        )

    output_dir = (
        output_root
        / experiment_name
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    slice_path = (
        output_dir
        / "slice_metrics.csv"
    )

    subject_path = (
        output_dir
        / "subject_metrics.csv"
    )

    summary_path = (
        output_dir
        / "summary.csv"
    )

    with slice_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(
                slice_rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            slice_rows
        )

    with subject_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(
                subject_rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            subject_rows
        )

    summary_row = {
        "experiment": experiment_name,
        "checkpoint_path": display_path(
            checkpoint_path
        ),
        "checkpoint_epoch": int(
            checkpoint["epoch"]
        ),
        "threshold": threshold,
        "n_subjects": len(
            manifest_rows
        ),
        "n_slices": len(
            all_dice
        ),
        "n_positive_slices": len(
            positive_dice
        ),
        "mean_dice_all_slices": float(
            np.mean(
                all_dice
            )
        ),
        "mean_iou_all_slices": float(
            np.mean(
                all_iou
            )
        ),
        "mean_dice_positive_slices": float(
            np.mean(
                positive_dice
            )
        ),
        "mean_iou_positive_slices": float(
            np.mean(
                positive_iou
            )
        ),
        "mean_subject_volumetric_dice": float(
            np.mean(
                [
                    row[
                        "volumetric_dice"
                    ]
                    for row
                    in subject_rows
                ]
            )
        ),
        "mean_subject_volumetric_iou": float(
            np.mean(
                [
                    row[
                        "volumetric_iou"
                    ]
                    for row
                    in subject_rows
                ]
            )
        ),
        "mean_subject_dice_all_slices": float(
            np.mean(
                [
                    row[
                        "mean_dice_all_slices"
                    ]
                    for row
                    in subject_rows
                ]
            )
        ),
        "mean_subject_iou_all_slices": float(
            np.mean(
                [
                    row[
                        "mean_iou_all_slices"
                    ]
                    for row
                    in subject_rows
                ]
            )
        ),
        "mean_subject_dice_positive_slices": float(
            np.mean(
                [
                    row[
                        "mean_dice_positive_slices"
                    ]
                    for row
                    in subject_rows
                ]
            )
        ),
        "mean_subject_iou_positive_slices": float(
            np.mean(
                [
                    row[
                        "mean_iou_positive_slices"
                    ]
                    for row
                    in subject_rows
                ]
            )
        ),
    }

    with summary_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(
                summary_row.keys()
            ),
        )

        writer.writeheader()
        writer.writerow(
            summary_row
        )

    print()
    print(
        f"Completed: {experiment_name}"
    )

    print(
        "Mean positive-slice Dice:",
        summary_row[
            "mean_dice_positive_slices"
        ],
    )

    print(
        "Mean subject positive-slice Dice:",
        summary_row[
            "mean_subject_dice_positive_slices"
        ],
    )

    print(
        "Mean subject volumetric Dice:",
        summary_row[
            "mean_subject_volumetric_dice"
        ],
    )

    print(
        "Summary:",
        summary_path,
    )

    print()

    return summary_row


def write_json(
    path: Path,
    payload: dict,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            payload,
            file,
            indent=2,
            sort_keys=True,
        )

        file.write("\n")


def build_run_metadata(
    *,
    manifest_path: Path,
    raw_root: Path,
    checkpoints: dict[str, Path],
    output_dir: Path,
    batch_size: int,
    device: torch.device,
) -> dict:
    return {
        "created_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "cohort": {
            "name": "UCSF-PDGM",
            "frozen_subject_count": EXPECTED_SUBJECTS,
            "manifest": {
                "path": display_path(
                    manifest_path
                ),
                "sha256": sha256_file(
                    manifest_path
                ),
            },
            "raw_root": str(
                raw_root
            ),
            "expected_image_shape": list(
                EXPECTED_SHAPE
            ),
            "image_modality": "FLAIR",
            "whole_tumor_rule": "segmentation > 0",
        },
        "checkpoints": {
            experiment_name: {
                "path": display_path(
                    checkpoint_path
                ),
                "sha256": sha256_file(
                    checkpoint_path
                ),
            }
            for experiment_name, checkpoint_path
            in checkpoints.items()
        },
        "evaluation": {
            "batch_size": batch_size,
            "output_dir": str(
                output_dir
            ),
            "empty_empty_dice": 1.0,
            "empty_empty_iou": 1.0,
        },
        "git_commit": git_commit(),
        "runtime": {
            "python": sys.version,
            "torch": torch.__version__,
            "numpy": package_version(
                "numpy"
            ),
            "nibabel": package_version(
                "nibabel"
            ),
            "cuda_available": torch.cuda.is_available(),
            "cuda_build": torch.version.cuda,
            "device": str(
                device
            ),
            "gpu_name": (
                torch.cuda.get_device_name(
                    0
                )
                if device.type == "cuda"
                else None
            ),
            "slurm_job_id": os.environ.get(
                "SLURM_JOB_ID"
            ),
            "slurm_partition": os.environ.get(
                "SLURM_JOB_PARTITION"
            ),
            "host": os.environ.get(
                "HOSTNAME"
            ),
        },
        "provenance": {
            "reference_repository": (
                "https://github.com/edaaydinea/"
                "Low-Grade-Glioma-Segmentation"
            ),
            "note": (
                "Downstream segmentation idea and vanilla U-Net structure "
                "adapted in part from the reference repository; the current "
                "implementation was rewritten for the BraTS/UCSF-PDGM and "
                "BR-LoRA workflow."
            ),
        },
    }


def main() -> None:
    args = parse_args()

    if args.batch_size <= 0:
        raise ValueError(
            "--batch-size must be a positive integer."
        )

    raw_root, manifest_path = resolve_paths(
        args
    )

    raw_root = (
        raw_root
        .expanduser()
        .resolve()
    )

    manifest_path = (
        manifest_path
        .expanduser()
        .resolve()
    )

    validate_file(
        manifest_path,
        "UCSF-PDGM manifest",
    )

    manifest_rows = load_manifest(
        manifest_path
    )

    validate_external_dataset_paths(
        manifest_rows,
        raw_root,
    )

    checkpoints = resolve_checkpoints(
        args
    )

    checkpoint_metadata = {}

    for experiment_name in EXPERIMENT_NAMES:
        checkpoint = validate_checkpoint(
            checkpoints[
                experiment_name
            ],
            experiment_name,
        )

        checkpoint_metadata[
            experiment_name
        ] = {
            "epoch": int(
                checkpoint["epoch"]
            ),
            "threshold": float(
                checkpoint.get(
                    "threshold",
                    0.5,
                )
            ),
        }

    device = resolve_device(
        args.device
    )

    print(
        "Device:",
        device,
    )

    print(
        "External manifest:",
        manifest_path,
    )

    print(
        "Raw root:",
        raw_root,
    )

    print(
        "Subjects:",
        len(
            manifest_rows
        ),
    )

    print(
        "Batch size:",
        args.batch_size,
    )

    print()

    for experiment_name in EXPERIMENT_NAMES:
        print(
            f"{experiment_name}: "
            f"{checkpoints[experiment_name]}"
        )

        print(
            "  epoch:",
            checkpoint_metadata[
                experiment_name
            ]["epoch"],
        )

        print(
            "  threshold:",
            checkpoint_metadata[
                experiment_name
            ]["threshold"],
        )

    print()

    if args.validate_only:
        print(
            "UCSF-PDGM evaluator validation: PASS"
        )

        print(
            "No inference was run and no evaluation output "
            "directory was created."
        )

        return

    output_dir = (
        default_output_dir()
        if args.output_dir is None
        else args.output_dir
    )

    output_dir = prepare_output_directory(
        output_dir
    )

    metadata = build_run_metadata(
        manifest_path=manifest_path,
        raw_root=raw_root,
        checkpoints=checkpoints,
        output_dir=output_dir,
        batch_size=args.batch_size,
        device=device,
    )

    metadata_path = (
        output_dir
        / "run_metadata.json"
    )

    write_json(
        metadata_path,
        metadata,
    )

    summaries = {}

    for experiment_name in EXPERIMENT_NAMES:
        summaries[
            experiment_name
        ] = evaluate_checkpoint(
            experiment_name=experiment_name,
            checkpoint_path=checkpoints[
                experiment_name
            ],
            manifest_rows=manifest_rows,
            raw_root=raw_root,
            output_root=output_dir,
            batch_size=args.batch_size,
            device=device,
        )

    metadata[
        "completed_utc"
    ] = datetime.now(
        timezone.utc
    ).isoformat()

    metadata[
        "summaries"
    ] = summaries

    write_json(
        metadata_path,
        metadata,
    )

    print(
        "All UCSF-PDGM evaluations complete."
    )

    print(
        "Output directory:",
        output_dir,
    )


if __name__ == "__main__":
    main()
