#!/usr/bin/env python3

"""
Analyze baseline-generator response to controlled mask perturbations.

This Stage-7 analysis uses the frozen full-training baseline checkpoint and
the deterministic signed-distance perturbations validated by Stage 6.

For each baseline-training slice, the clean image, original appearance
conditioning vector, diffusion timestep, and explicit diffusion-noise tensor
are held fixed across all perturbation conditions. Only the spatial binary
conditioning mask and tensors derived directly from that mask are changed.

The primary comparison is paired against the radius-zero generation from the
same slice and diffusion realization.

No training is performed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd
import torch
from scipy.ndimage import distance_transform_edt


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )

from src.config import load_folders_config, resolve_path
from src.data import BraTSH5PatchX0Dataset
from src.diffusion import DiffusionSchedule
from src.models import AppearanceX0UNet


EXPECTED_SLICES = 19_941
EXPECTED_TIMESTEPS = 200

DEFAULT_TIMESTEP = 150
DEFAULT_SEED = 2026

SELECTED_RADII = (
    -4,
    -2,
    -1,
    0,
    1,
    2,
    4,
)

DEFAULT_CHECKPOINT = (
    PROJECT_ROOT
    / "checkpoints"
    / "baseline"
    / "full_train"
    / "final_patch_x0_diffusion_full_train.pt"
)

DEFAULT_FOLDERS = (
    PROJECT_ROOT
    / "data"
    / "folders.yaml"
)

DEFAULT_SIGNED_DISTANCE_CATALOG = (
    PROJECT_ROOT
    / "outputs"
    / "baseline_mask_sensitivity"
    / "signed_distance"
    / "signed_distance_perturbation_catalog.csv"
)

DEFAULT_TOPOLOGY_AUDIT = (
    PROJECT_ROOT
    / "outputs"
    / "baseline_mask_sensitivity"
    / "signed_distance"
    / "signed_distance_topology_audit.csv"
)

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "baseline_mask_sensitivity"
    / "perturbation_response"
)

REQUIRED_PERTURBATION_COLUMNS = {
    "volume",
    "slice",
    "slice_path",
    "radius_pixels",
    "perturbation",
    "original_mask_pixels",
    "perturbed_mask_pixels",
    "is_empty",
}

REQUIRED_TOPOLOGY_COLUMNS = {
    "volume",
    "slice",
    "radius_pixels",
    "original_beta_0",
    "perturbed_beta_0",
    "delta_beta_0",
    "original_beta_1",
    "perturbed_beta_1",
    "delta_beta_1",
    "original_euler_characteristic",
    "perturbed_euler_characteristic",
    "delta_euler_characteristic",
    "topology_changed",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure paired baseline-generator response to validated "
            "signed-distance mask perturbations."
        )
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
    )
    parser.add_argument(
        "--folders-file",
        type=Path,
        default=DEFAULT_FOLDERS,
    )
    parser.add_argument(
        "--signed-distance-catalog",
        type=Path,
        default=DEFAULT_SIGNED_DISTANCE_CATALOG,
    )
    parser.add_argument(
        "--topology-audit",
        type=Path,
        default=DEFAULT_TOPOLOGY_AUDIT,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cuda",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
    )
    parser.add_argument(
        "--timestep",
        type=int,
        default=DEFAULT_TIMESTEP,
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Optional diagnostic slice limit. Omit for the full "
            "scientific cohort."
        ),
    )

    args = parser.parse_args()

    if args.timestep < 0 or args.timestep >= EXPECTED_TIMESTEPS:
        parser.error(
            f"--timestep must be between 0 and "
            f"{EXPECTED_TIMESTEPS - 1}."
        )

    if args.limit is not None and args.limit <= 0:
        parser.error(
            "--limit must be positive."
        )

    return args


def resolve_project_path(
    path: Path,
) -> Path:
    path = path.expanduser()

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    return path.resolve()


def sha256_file(
    path: Path,
) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def parse_volume_slice(
    path: str | Path,
) -> tuple[int, int]:
    match = re.search(
        r"volume_(\d+)_slice_(\d+)\.h5$",
        Path(path).name,
    )

    if match is None:
        raise ValueError(
            f"Could not parse volume/slice identifiers from {path}"
        )

    return (
        int(match.group(1)),
        int(match.group(2)),
    )


def boolean_column(
    series: pd.Series,
) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)

    normalized = (
        series.astype(str)
        .str.strip()
        .str.lower()
    )

    unexpected = sorted(
        set(normalized.unique())
        - {"true", "false"}
    )

    if unexpected:
        raise RuntimeError(
            "Unexpected boolean values: "
            f"{unexpected}"
        )

    return normalized.map(
        {
            "true": True,
            "false": False,
        }
    ).astype(bool)


def perturb_mask(
    *,
    original_mask: np.ndarray,
    radius: int,
    interior_distance: np.ndarray,
    exterior_distance: np.ndarray,
) -> np.ndarray:
    """
    Reproduce the validated Stage-3/Stage-6 EDT perturbation convention.
    """
    if radius == 0:
        return original_mask.copy()

    if radius < 0:
        erosion_distance = abs(radius)

        return (
            original_mask
            & (
                interior_distance
                > float(erosion_distance)
            )
        )

    return (
        original_mask
        | (
            (~original_mask)
            & (
                exterior_distance
                <= float(radius)
            )
        )
    )


def signed_distance(
    mask: np.ndarray,
) -> np.ndarray:
    binary = np.asarray(
        mask,
        dtype=bool,
    )

    if binary.ndim != 2:
        raise ValueError(
            "Mask must be two-dimensional."
        )

    if not binary.any():
        raise ValueError(
            "Cannot construct signed distance for an empty mask."
        )

    inside = distance_transform_edt(
        binary
    )
    outside = distance_transform_edt(
        ~binary
    )

    signed = inside.astype(
        np.float64,
        copy=True,
    )

    signed[~binary] = (
        -outside[~binary]
    )

    return signed


def exact_distance_selector(
    signed: np.ndarray,
    distance: int,
) -> np.ndarray:
    if distance == 0:
        raise ValueError(
            "The binary EDT representation has no d=0 raster band."
        )

    if distance > 0:
        return (
            (signed > distance - 1)
            & (signed <= distance)
        )

    magnitude = abs(distance)

    return (
        (-signed > magnitude - 1)
        & (-signed <= magnitude)
    )


def mean_abs_difference(
    first: np.ndarray,
    second: np.ndarray,
    selector: np.ndarray | None = None,
) -> float:
    difference = np.abs(
        first - second
    )

    if selector is not None:
        if not selector.any():
            return float("nan")

        difference = difference[
            selector
        ]

    return float(
        difference.mean()
    )


def rms_difference(
    first: np.ndarray,
    second: np.ndarray,
) -> float:
    difference = (
        first - second
    )

    return float(
        np.sqrt(
            np.mean(
                difference ** 2
            )
        )
    )


def total_abs_difference(
    first: np.ndarray,
    second: np.ndarray,
    selector: np.ndarray | None = None,
) -> float:
    difference = np.abs(
        first - second
    )

    if selector is not None:
        difference = difference[
            selector
        ]

    return float(
        difference.sum()
    )


def changed_region(
    original_mask: np.ndarray,
    perturbed_mask: np.ndarray,
) -> np.ndarray:
    return np.logical_xor(
        original_mask,
        perturbed_mask,
    )


def expanded_region(
    region: np.ndarray,
    *,
    radius: int = 4,
) -> np.ndarray:
    """
    Return pixels within `radius` Euclidean pixels of a non-empty region.
    """
    if not region.any():
        return np.zeros_like(
            region,
            dtype=bool,
        )

    distance = distance_transform_edt(
        ~region
    )

    return distance <= float(radius)


@torch.no_grad()
def run_mask_condition(
    *,
    model: torch.nn.Module,
    x0: torch.Tensor,
    mask: torch.Tensor,
    cond: torch.Tensor,
    x_t_full: torch.Tensor,
    timestep: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Run one mask condition while reusing the already-fixed q(x_t | x_0).
    """
    known = (
        x0
        * (1.0 - mask)
    )

    donor_patch = (
        x0
        * mask
    )

    x_t = (
        x0
        * (1.0 - mask)
        + x_t_full
        * mask
    )

    model_input = torch.cat(
        [
            x_t,
            known,
            mask,
            donor_patch,
        ],
        dim=1,
    )

    t = torch.full(
        (x0.shape[0],),
        timestep,
        device=x0.device,
        dtype=torch.long,
    )

    prediction = model(
        model_input,
        t,
        cond,
    )

    composite = (
        x0
        * (1.0 - mask)
        + prediction
        * mask
    )

    return (
        prediction,
        composite,
    )


def response_metrics(
    *,
    reference_prediction: np.ndarray,
    reference_composite: np.ndarray,
    prediction: np.ndarray,
    composite: np.ndarray,
    original_mask: np.ndarray,
    perturbed_mask: np.ndarray,
    original_signed: np.ndarray,
) -> dict[str, float | int]:
    inside = original_mask
    outside = ~original_mask

    boundary_shell = (
        exact_distance_selector(
            original_signed,
            -1,
        )
        | exact_distance_selector(
            original_signed,
            1,
        )
    )

    changed = changed_region(
        original_mask,
        perturbed_mask,
    )

    changed_neighborhood = expanded_region(
        changed,
        radius=4,
    )

    away_from_change = (
        ~changed_neighborhood
    )

    prediction_total = total_abs_difference(
        prediction,
        reference_prediction,
    )

    composite_total = total_abs_difference(
        composite,
        reference_composite,
    )

    prediction_near = total_abs_difference(
        prediction,
        reference_prediction,
        changed_neighborhood,
    )

    composite_near = total_abs_difference(
        composite,
        reference_composite,
        changed_neighborhood,
    )

    return {
        "changed_pixels": int(
            changed.sum()
        ),
        "changed_neighborhood_pixels": int(
            changed_neighborhood.sum()
        ),
        "prediction_mae_global": mean_abs_difference(
            prediction,
            reference_prediction,
        ),
        "composite_mae_global": mean_abs_difference(
            composite,
            reference_composite,
        ),
        "prediction_rmse_global": rms_difference(
            prediction,
            reference_prediction,
        ),
        "composite_rmse_global": rms_difference(
            composite,
            reference_composite,
        ),
        "prediction_max_abs_change": float(
            np.max(
                np.abs(
                    prediction
                    - reference_prediction
                )
            )
        ),
        "composite_max_abs_change": float(
            np.max(
                np.abs(
                    composite
                    - reference_composite
                )
            )
        ),
        "prediction_mae_original_inside": mean_abs_difference(
            prediction,
            reference_prediction,
            inside,
        ),
        "composite_mae_original_inside": mean_abs_difference(
            composite,
            reference_composite,
            inside,
        ),
        "prediction_mae_original_outside": mean_abs_difference(
            prediction,
            reference_prediction,
            outside,
        ),
        "composite_mae_original_outside": mean_abs_difference(
            composite,
            reference_composite,
            outside,
        ),
        "prediction_mae_original_boundary": mean_abs_difference(
            prediction,
            reference_prediction,
            boundary_shell,
        ),
        "composite_mae_original_boundary": mean_abs_difference(
            composite,
            reference_composite,
            boundary_shell,
        ),
        "prediction_mae_changed_region": mean_abs_difference(
            prediction,
            reference_prediction,
            changed,
        ),
        "composite_mae_changed_region": mean_abs_difference(
            composite,
            reference_composite,
            changed,
        ),
        "prediction_mae_away_from_change": mean_abs_difference(
            prediction,
            reference_prediction,
            away_from_change,
        ),
        "composite_mae_away_from_change": mean_abs_difference(
            composite,
            reference_composite,
            away_from_change,
        ),
        "prediction_abs_response_total": prediction_total,
        "composite_abs_response_total": composite_total,
        "prediction_abs_response_near_change_fraction": (
            prediction_near / prediction_total
            if prediction_total > 0
            else 0.0
        ),
        "composite_abs_response_near_change_fraction": (
            composite_near / composite_total
            if composite_total > 0
            else 0.0
        ),
    }


def distance_profile_rows(
    *,
    volume: int,
    slice_index: int,
    radius: int,
    reference_prediction: np.ndarray,
    reference_composite: np.ndarray,
    prediction: np.ndarray,
    composite: np.ndarray,
    original_signed: np.ndarray,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    distances = (
        list(range(-8, 0))
        + list(range(1, 9))
    )

    for distance in distances:
        selector = exact_distance_selector(
            original_signed,
            distance,
        )

        rows.append(
            {
                "volume": volume,
                "slice": slice_index,
                "radius_pixels": radius,
                "distance_pixels": distance,
                "pixel_count": int(
                    selector.sum()
                ),
                "prediction_mae_vs_original_mask": mean_abs_difference(
                    prediction,
                    reference_prediction,
                    selector,
                ),
                "composite_mae_vs_original_mask": mean_abs_difference(
                    composite,
                    reference_composite,
                    selector,
                ),
            }
        )

    return rows


def write_csv(
    path: Path,
    rows: list[dict[str, object]],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not rows:
        raise RuntimeError(
            f"Refusing to write empty CSV: {path}"
        )

    temporary = path.with_suffix(
        path.suffix + ".tmp"
    )

    try:
        with temporary.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=list(
                    rows[0].keys()
                ),
            )

            writer.writeheader()
            writer.writerows(
                rows
            )

        temporary.replace(
            path
        )

    except Exception:
        if temporary.exists():
            temporary.unlink()

        raise


def main() -> None:
    args = parse_args()

    if (
        args.device == "cuda"
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "CUDA was requested but is unavailable."
        )

    device = torch.device(
        args.device
    )

    folders_file = resolve_project_path(
        args.folders_file
    )

    folders = load_folders_config(
        folders_file
    )

    h5_root = resolve_path(
        key="h5_root",
        cli_value=None,
        config=folders,
    ).expanduser().resolve()

    manifest = resolve_path(
        key="manifest_path",
        cli_value=None,
        config=folders,
    ).expanduser().resolve()

    checkpoint_path = resolve_project_path(
        args.checkpoint
    )

    perturbation_path = resolve_project_path(
        args.signed_distance_catalog
    )

    topology_path = resolve_project_path(
        args.topology_audit
    )

    output_dir = resolve_project_path(
        args.output_dir
    )

    for required_path in (
        checkpoint_path,
        perturbation_path,
        topology_path,
    ):
        if not required_path.is_file():
            raise RuntimeError(
                f"Required input does not exist: {required_path}"
            )

    perturbations = pd.read_csv(
        perturbation_path
    )

    topology = pd.read_csv(
        topology_path
    )

    missing = sorted(
        REQUIRED_PERTURBATION_COLUMNS
        - set(perturbations.columns)
    )

    if missing:
        raise RuntimeError(
            "Perturbation catalog missing columns: "
            f"{missing}"
        )

    missing = sorted(
        REQUIRED_TOPOLOGY_COLUMNS
        - set(topology.columns)
    )

    if missing:
        raise RuntimeError(
            "Topology audit missing columns: "
            f"{missing}"
        )

    perturbations["is_empty"] = boolean_column(
        perturbations["is_empty"]
    )

    perturbations = perturbations[
        perturbations["radius_pixels"].isin(
            SELECTED_RADII
        )
    ].copy()

    topology = topology[
        topology["radius_pixels"].isin(
            SELECTED_RADII
        )
    ].copy()

    expected_rows = (
        EXPECTED_SLICES
        * len(SELECTED_RADII)
    )

    if len(perturbations) != expected_rows:
        raise RuntimeError(
            "Unexpected selected perturbation row count: "
            f"{len(perturbations):,}; expected "
            f"{expected_rows:,}."
        )

    if len(topology) != expected_rows:
        raise RuntimeError(
            "Unexpected selected topology row count: "
            f"{len(topology):,}; expected "
            f"{expected_rows:,}."
        )

    merged = perturbations.merge(
        topology[
            [
                "volume",
                "slice",
                "radius_pixels",
                "original_beta_0",
                "perturbed_beta_0",
                "delta_beta_0",
                "original_beta_1",
                "perturbed_beta_1",
                "delta_beta_1",
                "original_euler_characteristic",
                "perturbed_euler_characteristic",
                "delta_euler_characteristic",
                "topology_changed",
            ]
        ],
        on=[
            "volume",
            "slice",
            "radius_pixels",
        ],
        how="left",
        validate="one_to_one",
    )

    if merged[
        "original_beta_0"
    ].isna().any():
        raise RuntimeError(
            "Stage-6 topology join produced missing rows."
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    timesteps = int(
        checkpoint["timesteps"]
    )

    if timesteps != EXPECTED_TIMESTEPS:
        raise RuntimeError(
            f"Expected {EXPECTED_TIMESTEPS} diffusion timesteps; "
            f"observed {timesteps}."
        )

    model = AppearanceX0UNet(
        in_ch=4,
        out_ch=1,
        base=int(
            checkpoint["base_channels"]
        ),
        time_dim=128,
        cond_dim=int(
            checkpoint["cond_dim"]
        ),
    ).to(
        device
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model.eval()

    schedule = DiffusionSchedule(
        timesteps=timesteps,
        beta_start=1e-4,
        beta_end=0.02,
        device=device,
    )

    dataset = BraTSH5PatchX0Dataset(
        root=h5_root,
        manifest_path=manifest,
        image_channel=int(
            checkpoint["image_channel"]
        ),
        min_tumor_pixels=int(
            checkpoint["min_tumor_pixels"]
        ),
        use_whole_tumor=True,
    )

    if len(dataset) != EXPECTED_SLICES:
        raise RuntimeError(
            "Unexpected baseline cohort size: "
            f"{len(dataset):,}; expected "
            f"{EXPECTED_SLICES:,}."
        )

    n_cases = (
        len(dataset)
        if args.limit is None
        else min(
            args.limit,
            len(dataset),
        )
    )

    print("=" * 68)
    print("STAGE 7: BASELINE PERTURBATION RESPONSE")
    print("=" * 68)
    print(f"H5 root:                {h5_root}")
    print(f"Manifest:               {manifest}")
    print(f"Checkpoint:             {checkpoint_path}")
    print(
        f"Checkpoint SHA-256:     "
        f"{sha256_file(checkpoint_path)}"
    )
    print(f"Perturbation catalog:   {perturbation_path}")
    print(f"Topology audit:         {topology_path}")
    print(f"Dataset slices:         {len(dataset):,}")
    print(f"Slices analyzed:        {n_cases:,}")
    print(
        "Selected radii:         "
        + ", ".join(
            f"{radius:+d}"
            for radius in SELECTED_RADII
        )
    )
    print(f"Device:                 {device}")
    print(f"Inference timestep:     {args.timestep}")
    print(f"Noise seed base:        {args.seed}")
    print("Noise pairing:          fixed within slice")
    print("Appearance cond:        fixed from original mask")
    print("Reference generation:   radius 0")
    print("Training:               none")
    print()

    response_rows: list[
        dict[str, object]
    ] = []

    profile_rows: list[
        dict[str, object]
    ] = []

    skipped_empty = 0

    for index in range(
        n_cases
    ):
        sample = dataset[
            index
        ]

        x0 = sample[
            "x0"
        ].unsqueeze(0).to(
            device
        )

        original_mask_tensor = sample[
            "mask"
        ].unsqueeze(0).to(
            device
        )

        original_cond = sample[
            "cond"
        ].unsqueeze(0).to(
            device
        )

        volume, slice_index = parse_volume_slice(
            sample["path"]
        )

        case_catalog = merged[
            (merged["volume"] == volume)
            & (merged["slice"] == slice_index)
        ].copy()

        if len(case_catalog) != len(
            SELECTED_RADII
        ):
            raise RuntimeError(
                "Expected seven perturbation rows for "
                f"volume={volume}, slice={slice_index}; "
                f"observed {len(case_catalog)}."
            )

        original_mask = (
            original_mask_tensor
            .detach()
            .cpu()
            .numpy()[0, 0]
            > 0.5
        )

        original_pixels = int(
            original_mask.sum()
        )

        catalog_original_pixels = (
            case_catalog[
                "original_mask_pixels"
            ]
            .astype(int)
            .unique()
        )

        if (
            len(catalog_original_pixels) != 1
            or int(
                catalog_original_pixels[0]
            ) != original_pixels
        ):
            raise RuntimeError(
                "Dataset original mask disagrees with "
                "the perturbation catalog."
            )

        interior_distance = distance_transform_edt(
            original_mask
        )

        exterior_distance = distance_transform_edt(
            ~original_mask
        )

        original_signed = signed_distance(
            original_mask
        )

        noise_seed = (
            args.seed
            + index
        )

        generator = torch.Generator(
            device=device
        )

        generator.manual_seed(
            noise_seed
        )

        noise = torch.randn(
            x0.shape,
            dtype=x0.dtype,
            device=device,
            generator=generator,
        )

        t = torch.full(
            (x0.shape[0],),
            args.timestep,
            device=device,
            dtype=torch.long,
        )

        x_t_full = schedule.q_sample(
            x0=x0,
            t=t,
            noise=noise,
        )

        reference_prediction_tensor, reference_composite_tensor = (
            run_mask_condition(
                model=model,
                x0=x0,
                mask=original_mask_tensor,
                cond=original_cond,
                x_t_full=x_t_full,
                timestep=args.timestep,
            )
        )

        reference_prediction = (
            reference_prediction_tensor
            .detach()
            .cpu()
            .numpy()[0, 0]
        )

        reference_composite = (
            reference_composite_tensor
            .detach()
            .cpu()
            .numpy()[0, 0]
        )

        for radius in SELECTED_RADII:
            row = case_catalog[
                case_catalog[
                    "radius_pixels"
                ] == radius
            ]

            if len(row) != 1:
                raise RuntimeError(
                    "Expected exactly one perturbation row for "
                    f"volume={volume}, slice={slice_index}, "
                    f"radius={radius:+d}."
                )

            row = row.iloc[0]

            perturbed_mask = perturb_mask(
                original_mask=original_mask,
                radius=radius,
                interior_distance=interior_distance,
                exterior_distance=exterior_distance,
            )

            perturbed_pixels = int(
                perturbed_mask.sum()
            )

            expected_pixels = int(
                row[
                    "perturbed_mask_pixels"
                ]
            )

            if perturbed_pixels != expected_pixels:
                raise RuntimeError(
                    "Stage-7 reconstructed mask disagrees with "
                    "validated perturbation catalog.\n"
                    f"Volume:   {volume}\n"
                    f"Slice:    {slice_index}\n"
                    f"Radius:   {radius:+d}\n"
                    f"Expected: {expected_pixels}\n"
                    f"Observed: {perturbed_pixels}"
                )

            is_empty = bool(
                perturbed_pixels == 0
            )

            if is_empty != bool(
                row["is_empty"]
            ):
                raise RuntimeError(
                    "Stage-7 empty-mask state disagrees with "
                    "validated perturbation catalog."
                )

            base_row = {
                "volume": volume,
                "slice": slice_index,
                "slice_path": sample["path"],
                "radius_pixels": radius,
                "perturbation": str(
                    row["perturbation"]
                ),
                "original_mask_pixels": original_pixels,
                "perturbed_mask_pixels": perturbed_pixels,
                "noise_seed": noise_seed,
                "timestep": args.timestep,
                "is_empty": is_empty,
                "inference_skipped": is_empty,
                "original_beta_0": int(
                    row["original_beta_0"]
                ),
                "perturbed_beta_0": int(
                    row["perturbed_beta_0"]
                ),
                "delta_beta_0": int(
                    row["delta_beta_0"]
                ),
                "original_beta_1": int(
                    row["original_beta_1"]
                ),
                "perturbed_beta_1": int(
                    row["perturbed_beta_1"]
                ),
                "delta_beta_1": int(
                    row["delta_beta_1"]
                ),
                "original_euler_characteristic": int(
                    row[
                        "original_euler_characteristic"
                    ]
                ),
                "perturbed_euler_characteristic": int(
                    row[
                        "perturbed_euler_characteristic"
                    ]
                ),
                "delta_euler_characteristic": int(
                    row[
                        "delta_euler_characteristic"
                    ]
                ),
                "topology_changed": bool(
                    row["topology_changed"]
                ),
            }

            if is_empty:
                skipped_empty += 1

                response_rows.append(
                    {
                        **base_row,
                        "changed_pixels": original_pixels,
                        "changed_neighborhood_pixels": None,
                        "prediction_mae_global": None,
                        "composite_mae_global": None,
                        "prediction_rmse_global": None,
                        "composite_rmse_global": None,
                        "prediction_max_abs_change": None,
                        "composite_max_abs_change": None,
                        "prediction_mae_original_inside": None,
                        "composite_mae_original_inside": None,
                        "prediction_mae_original_outside": None,
                        "composite_mae_original_outside": None,
                        "prediction_mae_original_boundary": None,
                        "composite_mae_original_boundary": None,
                        "prediction_mae_changed_region": None,
                        "composite_mae_changed_region": None,
                        "prediction_mae_away_from_change": None,
                        "composite_mae_away_from_change": None,
                        "prediction_abs_response_total": None,
                        "composite_abs_response_total": None,
                        "prediction_abs_response_near_change_fraction": None,
                        "composite_abs_response_near_change_fraction": None,
                    }
                )

                continue

            mask_tensor = torch.from_numpy(
                perturbed_mask.astype(
                    np.float32
                )[None, None]
            ).to(
                device=device,
                dtype=x0.dtype,
            )

            if radius == 0:
                prediction_tensor = (
                    reference_prediction_tensor
                )

                composite_tensor = (
                    reference_composite_tensor
                )
            else:
                prediction_tensor, composite_tensor = (
                    run_mask_condition(
                        model=model,
                        x0=x0,
                        mask=mask_tensor,
                        cond=original_cond,
                        x_t_full=x_t_full,
                        timestep=args.timestep,
                    )
                )

            prediction = (
                prediction_tensor
                .detach()
                .cpu()
                .numpy()[0, 0]
            )

            composite = (
                composite_tensor
                .detach()
                .cpu()
                .numpy()[0, 0]
            )

            metrics = response_metrics(
                reference_prediction=reference_prediction,
                reference_composite=reference_composite,
                prediction=prediction,
                composite=composite,
                original_mask=original_mask,
                perturbed_mask=perturbed_mask,
                original_signed=original_signed,
            )

            if radius == 0:
                zero_metrics = [
                    metrics[
                        "prediction_mae_global"
                    ],
                    metrics[
                        "composite_mae_global"
                    ],
                    metrics[
                        "prediction_rmse_global"
                    ],
                    metrics[
                        "composite_rmse_global"
                    ],
                    metrics[
                        "prediction_max_abs_change"
                    ],
                    metrics[
                        "composite_max_abs_change"
                    ],
                ]

                if not all(
                    value == 0.0
                    for value in zero_metrics
                ):
                    raise RuntimeError(
                        "Radius-zero paired-response control failed."
                    )

                if not np.array_equal(
                    perturbed_mask,
                    original_mask,
                ):
                    raise RuntimeError(
                        "Radius-zero mask does not exactly reproduce "
                        "the original mask."
                    )

            response_rows.append(
                {
                    **base_row,
                    **metrics,
                }
            )

            profile_rows.extend(
                distance_profile_rows(
                    volume=volume,
                    slice_index=slice_index,
                    radius=radius,
                    reference_prediction=reference_prediction,
                    reference_composite=reference_composite,
                    prediction=prediction,
                    composite=composite,
                    original_signed=original_signed,
                )
            )

        if (
            (index + 1) % 100 == 0
            or index + 1 == n_cases
        ):
            print(
                f"Processed {index + 1:,} / {n_cases:,} slices"
            )

    expected_response_rows = (
        n_cases
        * len(SELECTED_RADII)
    )

    if len(response_rows) != expected_response_rows:
        raise RuntimeError(
            "Unexpected Stage-7 response row count: "
            f"{len(response_rows):,}; expected "
            f"{expected_response_rows:,}."
        )

    nonempty_conditions = sum(
        not bool(
            row["inference_skipped"]
        )
        for row in response_rows
    )

    expected_profile_rows = (
        nonempty_conditions
        * 16
    )

    if len(profile_rows) != expected_profile_rows:
        raise RuntimeError(
            "Unexpected Stage-7 profile row count: "
            f"{len(profile_rows):,}; expected "
            f"{expected_profile_rows:,}."
        )

    if args.limit is None:
        response_path = (
            output_dir
            / "baseline_perturbation_response_catalog.csv"
        )

        profile_path = (
            output_dir
            / "baseline_perturbation_distance_profiles.csv"
        )

    else:
        response_path = (
            output_dir
            / (
                f"diagnostic_n{n_cases}_"
                "perturbation_response_catalog.csv"
            )
        )

        profile_path = (
            output_dir
            / (
                f"diagnostic_n{n_cases}_"
                "perturbation_distance_profiles.csv"
            )
        )

    write_csv(
        response_path,
        response_rows,
    )

    write_csv(
        profile_path,
        profile_rows,
    )

    print()
    print("===== STAGE-7 VALIDATION =====")
    print(
        f"Response rows:          {len(response_rows):,}"
    )
    print(
        f"Inference conditions:   {nonempty_conditions:,}"
    )
    print(
        f"Empty/skipped:          {skipped_empty:,}"
    )
    print(
        f"Distance-profile rows:  {len(profile_rows):,}"
    )
    print()
    print("===== OUTPUTS =====")
    print(f"Response: {response_path}")
    print(f"Profiles: {profile_path}")
    print()
    print(
        "PASS: Stage-7 baseline perturbation-response analysis completed."
    )


if __name__ == "__main__":
    main()
