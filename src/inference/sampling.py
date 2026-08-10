"""
Baseline inference utilities extracted from notebook Cell 19.

These functions preserve the notebook's patch-conditioned x0 reconstruction
behavior while avoiding notebook-global state.
"""

from __future__ import annotations

import torch

from src.diffusion import DiffusionSchedule


@torch.no_grad()
def reconstruct_batch(
    *,
    model: torch.nn.Module,
    batch: dict,
    schedule: DiffusionSchedule,
    device: torch.device,
    timestep_fraction: float = 0.75,
    max_samples: int = 4,
) -> dict[str, torch.Tensor]:
    """
    Reproduce the reference notebook's reconstruction test.

    Notebook defaults
    -----------------
    timestep_fraction = 0.75
    max_samples = 4

    The timestep is computed exactly as:
        int(timestep_fraction * schedule.timesteps)

    Fresh Gaussian noise is sampled for every call, matching the notebook.

    Returns
    -------
    dict
        Tensors required for reconstruction inspection and visualization.
    """
    if not (
        0.0 <= timestep_fraction < 1.0
    ):
        raise ValueError(
            "timestep_fraction must satisfy 0 <= value < 1."
        )

    if max_samples <= 0:
        raise ValueError(
            "max_samples must be positive."
        )

    model.eval()

    x0 = batch[
        "x0"
    ][
        :max_samples
    ].to(
        device
    )

    known = batch[
        "known"
    ][
        :max_samples
    ].to(
        device
    )

    mask = batch[
        "mask"
    ][
        :max_samples
    ].to(
        device
    )

    donor_patch = batch[
        "donor_patch"
    ][
        :max_samples
    ].to(
        device
    )

    cond = batch[
        "cond"
    ][
        :max_samples
    ].to(
        device
    )

    t_value = int(
        timestep_fraction
        * schedule.timesteps
    )

    t = torch.full(
        (
            x0.shape[0],
        ),
        t_value,
        device=device,
        dtype=torch.long,
    )

    noise = torch.randn_like(
        x0
    )

    x_t_full = schedule.q_sample(
        x0=x0,
        t=t,
        noise=noise,
    )

    x_t = (
        x0
        * (
            1.0
            - mask
        )
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

    pred_x0 = model(
        model_input,
        t,
        cond,
    )

    composite = (
        x0
        * (
            1.0
            - mask
        )
        + pred_x0
        * mask
    )

    error = torch.abs(
        composite
        - x0
    )

    return {
        "x0": x0,
        "mask": mask,
        "known": known,
        "x_t": x_t,
        "donor_patch": donor_patch,
        "cond": cond,
        "t": t,
        "pred_x0": pred_x0,
        "composite": composite,
        "error": error,
    }
