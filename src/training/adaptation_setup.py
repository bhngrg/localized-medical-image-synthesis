"""Shared setup helpers for adaptation methods.

These helpers construct and strictly restore the validated
``AppearanceX0UNet`` baseline used by BR-LoRA and deterministic adaptation
comparators. Method-specific adapter configuration remains outside this
module.
"""

from __future__ import annotations

from pathlib import Path

import torch

from src.models import AppearanceX0UNet


def build_backbone(
    *,
    model_cfg: dict,
    device: torch.device,
) -> AppearanceX0UNet:
    """Construct the validated local AppearanceX0UNet."""

    return AppearanceX0UNet(
        in_ch=int(
            model_cfg.get(
                "in_channels",
                4,
            )
        ),
        out_ch=int(
            model_cfg.get(
                "out_channels",
                1,
            )
        ),
        base=int(
            model_cfg.get(
                "base_channels",
                32,
            )
        ),
        time_dim=int(
            model_cfg.get(
                "time_dim",
                128,
            )
        ),
        cond_dim=int(
            model_cfg.get(
                "cond_dim",
                4,
            )
        ),
    ).to(
        device
    )


def load_baseline_backbone(
    *,
    checkpoint_path: Path,
    model_cfg: dict,
    device: torch.device,
) -> tuple[
    AppearanceX0UNet,
    dict,
]:
    """Strictly restore the trained baseline backbone checkpoint."""

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    if not isinstance(
        checkpoint,
        dict,
    ):
        raise RuntimeError(
            "Baseline checkpoint must contain a dictionary."
        )

    if "model_state_dict" not in checkpoint:
        raise RuntimeError(
            "Baseline checkpoint is missing 'model_state_dict'."
        )

    model = build_backbone(
        model_cfg=model_cfg,
        device=device,
    )

    load_result = model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ],
        strict=True,
    )

    if (
        load_result.missing_keys
        or load_result.unexpected_keys
    ):
        raise RuntimeError(
            "Strict baseline checkpoint restoration unexpectedly "
            "reported state-dict mismatches."
        )

    return (
        model,
        checkpoint,
    )


__all__ = [
    "build_backbone",
    "load_baseline_backbone",
]
