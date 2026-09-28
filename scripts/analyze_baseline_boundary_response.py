#!/usr/bin/env python3

"""
Analyze baseline generator response relative to the original tumor boundary.

This Stage-4 analysis uses the frozen full-training baseline checkpoint and
all eligible tumor-containing training slices.  Each slice receives one
deterministic diffusion-noise realization.  Masks are not perturbed.

The analysis compares the real normalized FLAIR image, the network pred_x0,
and the final composited synthesis as functions of signed Euclidean distance
from the original whole-tumor mask boundary.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path
import re
import sys

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt, sobel


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_folders_config, resolve_path
from src.data import BraTSH5PatchX0Dataset
from src.diffusion import DiffusionSchedule
from src.models import AppearanceX0UNet


EXPECTED_SLICES = 19_941
EXPECTED_TIMESTEPS = 200
DEFAULT_TIMESTEP = 150
DEFAULT_SEED = 2026
MIN_DISTANCE = -8
MAX_DISTANCE = 8

DEFAULT_CHECKPOINT = (
    PROJECT_ROOT
    / "checkpoints"
    / "baseline"
    / "full_train"
    / "final_patch_x0_diffusion_full_train.pt"
)

DEFAULT_FOLDERS = PROJECT_ROOT / "data" / "folders.yaml"

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "baseline_mask_sensitivity"
    / "boundary_response"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze baseline synthesis behavior as a function of signed "
            "distance from the original whole-tumor boundary."
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
            "Optional diagnostic slice limit. Omit for the scientific "
            "full-cohort analysis."
        ),
    )

    args = parser.parse_args()

    if args.timestep < 0 or args.timestep >= EXPECTED_TIMESTEPS:
        parser.error(
            f"--timestep must be between 0 and "
            f"{EXPECTED_TIMESTEPS - 1}."
        )

    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive.")

    return args


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def parse_volume_slice(path: str | Path) -> tuple[int, int]:
    match = re.search(
        r"volume_(\d+)_slice_(\d+)\.h5$",
        Path(path).name,
    )

    if match is None:
        raise ValueError(
            f"Could not parse volume/slice identifiers from {path}"
        )

    return int(match.group(1)), int(match.group(2))


def signed_distance(mask: np.ndarray) -> np.ndarray:
    binary = np.asarray(mask, dtype=bool)

    if binary.ndim != 2:
        raise ValueError("Mask must be two-dimensional.")

    if not binary.any():
        raise ValueError("Mask is empty.")

    inside = distance_transform_edt(binary)
    outside = distance_transform_edt(~binary)

    signed = inside.astype(np.float64, copy=True)
    signed[~binary] = -outside[~binary]

    return signed


def gradient_magnitude(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)

    gx = sobel(
        image,
        axis=1,
        mode="nearest",
    )
    gy = sobel(
        image,
        axis=0,
        mode="nearest",
    )

    # scipy.ndimage.sobel has the conventional factor-of-eight scale in 2-D.
    # Divide by eight so magnitudes remain interpretable on normalized FLAIR.
    return np.hypot(gx, gy) / 8.0


def exact_distance_selector(
    signed: np.ndarray,
    distance: int,
) -> np.ndarray:
    if distance == 0:
        raise ValueError(
            "The binary EDT representation has no d=0 raster band."
        )

    # EDT values are generally non-integer away from axis-aligned boundary
    # points.  A one-pixel-wide shell is therefore represented by distance
    # intervals centered on each integer distance.
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


def array_mean(
    values: np.ndarray,
    selector: np.ndarray,
) -> float:
    if not selector.any():
        return float("nan")

    return float(values[selector].mean())


def array_mae(
    first: np.ndarray,
    second: np.ndarray,
    selector: np.ndarray,
) -> float:
    if not selector.any():
        return float("nan")

    return float(
        np.abs(first[selector] - second[selector]).mean()
    )


@torch.no_grad()
def run_explicit_noise_inference(
    *,
    model: torch.nn.Module,
    schedule: DiffusionSchedule,
    x0: torch.Tensor,
    mask: torch.Tensor,
    cond: torch.Tensor,
    noise: torch.Tensor,
    timestep: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    known = x0 * (1.0 - mask)
    donor_patch = x0 * mask

    t = torch.full(
        (x0.shape[0],),
        timestep,
        device=x0.device,
        dtype=torch.long,
    )

    x_t_full = schedule.q_sample(
        x0=x0,
        t=t,
        noise=noise,
    )

    x_t = (
        x0 * (1.0 - mask)
        + x_t_full * mask
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

    pred_x0 = model(
        model_input,
        t,
        cond,
    )

    composite = (
        x0 * (1.0 - mask)
        + pred_x0 * mask
    )

    return pred_x0, composite


def compact_metrics(
    *,
    real: np.ndarray,
    prediction: np.ndarray,
    composite: np.ndarray,
    signed: np.ndarray,
) -> dict[str, float | int]:
    outside_1 = exact_distance_selector(
        signed,
        -1,
    )
    inside_1 = exact_distance_selector(
        signed,
        1,
    )

    inside_1_4 = (
        (signed >= 1)
        & (signed <= 4)
    )
    inside_deep = signed >= 5

    real_out = array_mean(
        real,
        outside_1,
    )
    real_in = array_mean(
        real,
        inside_1,
    )
    pred_out = array_mean(
        prediction,
        outside_1,
    )
    pred_in = array_mean(
        prediction,
        inside_1,
    )
    comp_out = array_mean(
        composite,
        outside_1,
    )
    comp_in = array_mean(
        composite,
        inside_1,
    )

    real_jump = real_in - real_out
    pred_jump = pred_in - pred_out
    comp_jump = comp_in - comp_out

    real_grad = gradient_magnitude(
        real
    )
    pred_grad = gradient_magnitude(
        prediction
    )
    comp_grad = gradient_magnitude(
        composite
    )

    boundary_shell = outside_1 | inside_1

    return {
        "real_boundary_jump_signed": real_jump,
        "real_boundary_jump_abs": abs(real_jump),
        "pred_boundary_jump_signed": pred_jump,
        "pred_boundary_jump_abs": abs(pred_jump),
        "composite_boundary_jump_signed": comp_jump,
        "composite_boundary_jump_abs": abs(comp_jump),
        "composite_excess_jump_signed": comp_jump - real_jump,
        "composite_excess_jump_abs": (
            abs(comp_jump) - abs(real_jump)
        ),
        "pred_excess_jump_signed": pred_jump - real_jump,
        "pred_excess_jump_abs": (
            abs(pred_jump) - abs(real_jump)
        ),
        "real_boundary_gradient_mean": array_mean(
            real_grad,
            boundary_shell,
        ),
        "pred_boundary_gradient_mean": array_mean(
            pred_grad,
            boundary_shell,
        ),
        "composite_boundary_gradient_mean": array_mean(
            comp_grad,
            boundary_shell,
        ),
        "composite_excess_boundary_gradient": (
            array_mean(comp_grad, boundary_shell)
            - array_mean(real_grad, boundary_shell)
        ),
        "pred_excess_boundary_gradient": (
            array_mean(pred_grad, boundary_shell)
            - array_mean(real_grad, boundary_shell)
        ),
        "pred_mae_inside_1_4": array_mae(
            prediction,
            real,
            inside_1_4,
        ),
        "composite_mae_inside_1_4": array_mae(
            composite,
            real,
            inside_1_4,
        ),
        "pred_mae_inside_deep": array_mae(
            prediction,
            real,
            inside_deep,
        ),
        "composite_mae_inside_deep": array_mae(
            composite,
            real,
            inside_deep,
        ),
    }


def distance_profile_rows(
    *,
    volume: int,
    slice_index: int,
    path: str,
    real: np.ndarray,
    prediction: np.ndarray,
    composite: np.ndarray,
    signed: np.ndarray,
) -> list[dict[str, float | int | str]]:
    real_grad = gradient_magnitude(
        real
    )
    pred_grad = gradient_magnitude(
        prediction
    )
    comp_grad = gradient_magnitude(
        composite
    )

    rows: list[
        dict[str, float | int | str]
    ] = []

    for distance in list(
        range(MIN_DISTANCE, 0)
    ) + list(
        range(1, MAX_DISTANCE + 1)
    ):
        selector = exact_distance_selector(
            signed,
            distance,
        )

        rows.append(
            {
                "volume": volume,
                "slice": slice_index,
                "slice_path": path,
                "distance_pixels": distance,
                "pixel_count": int(selector.sum()),
                "real_intensity_mean": array_mean(
                    real,
                    selector,
                ),
                "pred_intensity_mean": array_mean(
                    prediction,
                    selector,
                ),
                "composite_intensity_mean": array_mean(
                    composite,
                    selector,
                ),
                "pred_mae": array_mae(
                    prediction,
                    real,
                    selector,
                ),
                "composite_mae": array_mae(
                    composite,
                    real,
                    selector,
                ),
                "real_gradient_mean": array_mean(
                    real_grad,
                    selector,
                ),
                "pred_gradient_mean": array_mean(
                    pred_grad,
                    selector,
                ),
                "composite_gradient_mean": array_mean(
                    comp_grad,
                    selector,
                ),
            }
        )

    return rows


def write_csv(
    path: Path,
    rows: list[dict],
) -> None:
    if not rows:
        raise RuntimeError(
            f"Refusing to write empty CSV: {path}"
        )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if path.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing result: {path}"
        )

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but is unavailable."
        )

    device = torch.device(
        args.device
    )

    folders = load_folders_config(
        args.folders_file
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

    checkpoint_path = (
        args.checkpoint.expanduser().resolve()
    )
    output_dir = (
        args.output_dir.expanduser().resolve()
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
            f"Expected {EXPECTED_TIMESTEPS} timesteps; "
            f"observed {timesteps}."
        )

    model = AppearanceX0UNet(
        in_ch=4,
        out_ch=1,
        base=int(checkpoint["base_channels"]),
        time_dim=128,
        cond_dim=int(checkpoint["cond_dim"]),
    ).to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"]
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
        image_channel=int(checkpoint["image_channel"]),
        min_tumor_pixels=int(checkpoint["min_tumor_pixels"]),
        use_whole_tumor=True,
    )

    if len(dataset) != EXPECTED_SLICES:
        raise RuntimeError(
            "Unexpected baseline cohort size: "
            f"{len(dataset):,}; expected {EXPECTED_SLICES:,}."
        )

    n_cases = (
        len(dataset)
        if args.limit is None
        else min(args.limit, len(dataset))
    )

    if args.limit is None:
        run_label = "full scientific cohort"
    else:
        run_label = f"diagnostic first {n_cases:,} slices"

    print("=" * 60)
    print("BASELINE BOUNDARY-RESPONSE ANALYSIS")
    print("=" * 60)
    print(f"H5 root:             {h5_root}")
    print(f"Manifest:            {manifest}")
    print(f"Checkpoint:          {checkpoint_path}")
    print(f"Checkpoint SHA-256:  {sha256_file(checkpoint_path)}")
    print(f"Dataset slices:      {len(dataset):,}")
    print(f"Slices analyzed:     {n_cases:,}")
    print(f"Run type:            {run_label}")
    print(f"Device:              {device}")
    print(f"Diffusion timesteps: {timesteps}")
    print(f"Inference timestep:  {args.timestep}")
    print(f"Noise seed base:     {args.seed}")
    print(
        f"Distance profile:    {MIN_DISTANCE}..-1, +1..{MAX_DISTANCE}"
    )
    print("FLAIR scale:         notebook-normalized per slice")
    print("Mask perturbation:   none")
    print()

    slice_rows: list[dict] = []
    profile_rows: list[dict] = []

    for index in range(n_cases):
        sample = dataset[index]

        x0 = sample["x0"].unsqueeze(0).to(
            device
        )
        mask = sample["mask"].unsqueeze(0).to(
            device
        )
        cond = sample["cond"].unsqueeze(0).to(
            device
        )

        noise_seed = args.seed + index
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

        pred_x0, composite = run_explicit_noise_inference(
            model=model,
            schedule=schedule,
            x0=x0,
            mask=mask,
            cond=cond,
            noise=noise,
            timestep=args.timestep,
        )

        real_np = (
            x0.detach()
            .cpu()
            .numpy()[0, 0]
        )
        mask_np = (
            mask.detach()
            .cpu()
            .numpy()[0, 0]
            > 0.5
        )
        pred_np = (
            pred_x0.detach()
            .cpu()
            .numpy()[0, 0]
        )
        comp_np = (
            composite.detach()
            .cpu()
            .numpy()[0, 0]
        )

        signed = signed_distance(
            mask_np
        )

        volume, slice_index = parse_volume_slice(
            sample["path"]
        )

        metrics = compact_metrics(
            real=real_np,
            prediction=pred_np,
            composite=comp_np,
            signed=signed,
        )

        slice_rows.append(
            {
                "volume": volume,
                "slice": slice_index,
                "slice_path": sample["path"],
                "mask_pixels": int(mask_np.sum()),
                "noise_seed": noise_seed,
                "timestep": args.timestep,
                **metrics,
            }
        )

        profile_rows.extend(
            distance_profile_rows(
                volume=volume,
                slice_index=slice_index,
                path=sample["path"],
                real=real_np,
                prediction=pred_np,
                composite=comp_np,
                signed=signed,
            )
        )

        if (
            (index + 1) % 1000 == 0
            or index + 1 == n_cases
        ):
            print(
                f"Processed {index + 1:,} / {n_cases:,}"
            )

    expected_profiles = n_cases * 16

    if len(slice_rows) != n_cases:
        raise RuntimeError(
            "Unexpected slice-row count."
        )

    if len(profile_rows) != expected_profiles:
        raise RuntimeError(
            "Unexpected distance-profile row count: "
            f"{len(profile_rows):,}; expected "
            f"{expected_profiles:,}."
        )

    if args.limit is None:
        slice_path = (
            output_dir
            / "baseline_boundary_response_catalog.csv"
        )
        profile_path = (
            output_dir
            / "baseline_boundary_distance_profiles.csv"
        )
    else:
        slice_path = (
            output_dir
            / f"diagnostic_n{n_cases}_boundary_response_catalog.csv"
        )
        profile_path = (
            output_dir
            / f"diagnostic_n{n_cases}_distance_profiles.csv"
        )

    write_csv(
        slice_path,
        slice_rows,
    )
    write_csv(
        profile_path,
        profile_rows,
    )

    print()
    print(f"Wrote {len(slice_rows):,} slice rows:")
    print(slice_path)
    print()
    print(
        f"Wrote {len(profile_rows):,} signed-distance profile rows:"
    )
    print(profile_path)


if __name__ == "__main__":
    main()
