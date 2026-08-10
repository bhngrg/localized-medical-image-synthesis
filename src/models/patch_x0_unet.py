"""
Patch-conditioned x0-prediction U-Net extracted from notebook Cell 6.

This module preserves the reference notebook architecture exactly:
- sinusoidal timestep embedding
- timestep MLP
- appearance-conditioning MLP
- conditional residual blocks
- three encoder stages
- bottleneck
- two decoder stages with skip concatenation
- sigmoid output

The model class keeps the notebook's original constructor defaults. The baseline
experiment configuration may override those defaults exactly as the notebook
does (for example, in_ch=4).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal timestep embedding from the reference notebook."""

    def __init__(
        self,
        dim: int,
    ) -> None:
        super().__init__()
        self.dim = dim

    def forward(
        self,
        t: torch.Tensor,
    ) -> torch.Tensor:
        half_dim = self.dim // 2
        emb_scale = math.log(
            10000
        ) / (
            half_dim - 1
        )

        emb = torch.exp(
            torch.arange(
                half_dim,
                device=t.device,
            )
            * -emb_scale
        )

        emb = (
            t.float()[
                :,
                None,
            ]
            * emb[
                None,
                :,
            ]
        )

        emb = torch.cat(
            [
                torch.sin(
                    emb
                ),
                torch.cos(
                    emb
                ),
            ],
            dim=1,
        )

        return emb


class TimeMLP(nn.Module):
    """Timestep embedding MLP from the reference notebook."""

    def __init__(
        self,
        time_dim: int,
    ) -> None:
        super().__init__()

        self.net = nn.Sequential(
            SinusoidalTimeEmbedding(
                time_dim
            ),
            nn.Linear(
                time_dim,
                time_dim * 4,
            ),
            nn.SiLU(),
            nn.Linear(
                time_dim * 4,
                time_dim,
            ),
        )

    def forward(
        self,
        t: torch.Tensor,
    ) -> torch.Tensor:
        return self.net(
            t
        )


class CondMLP(nn.Module):
    """Appearance-conditioning MLP from the reference notebook."""

    def __init__(
        self,
        cond_dim: int,
        time_dim: int,
    ) -> None:
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(
                cond_dim,
                time_dim,
            ),
            nn.SiLU(),
            nn.Linear(
                time_dim,
                time_dim,
            ),
        )

    def forward(
        self,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        return self.net(
            cond
        )


class CondResBlock(nn.Module):
    """Conditional residual block from the reference notebook."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        time_dim: int,
    ) -> None:
        super().__init__()

        groups = (
            8
            if out_ch >= 8
            else 1
        )

        self.conv1 = nn.Conv2d(
            in_ch,
            out_ch,
            3,
            padding=1,
        )

        self.norm1 = nn.GroupNorm(
            groups,
            out_ch,
        )

        self.emb_proj = nn.Linear(
            time_dim,
            out_ch,
        )

        self.conv2 = nn.Conv2d(
            out_ch,
            out_ch,
            3,
            padding=1,
        )

        self.norm2 = nn.GroupNorm(
            groups,
            out_ch,
        )

        if in_ch != out_ch:
            self.skip = nn.Conv2d(
                in_ch,
                out_ch,
                1,
            )
        else:
            self.skip = nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        emb: torch.Tensor,
    ) -> torch.Tensor:
        h = self.conv1(
            x
        )

        h = self.norm1(
            h
        )

        h = F.silu(
            h
        )

        emb_term = self.emb_proj(
            emb
        ).view(
            emb.shape[0],
            -1,
            1,
            1,
        )

        h = (
            h
            + emb_term
        )

        h = self.conv2(
            h
        )

        h = self.norm2(
            h
        )

        h = F.silu(
            h
        )

        return (
            h
            + self.skip(
                x
            )
        )


class AppearanceX0UNet(nn.Module):
    """
    Conditional U-Net used by the patch-conditioned x0 diffusion baseline.

    Constructor defaults are preserved from notebook Cell 6:
    - in_ch = 3
    - out_ch = 1
    - base = 32
    - time_dim = 128
    - cond_dim = 4

    The notebook's baseline experiment instantiates the model with in_ch=4.
    """

    def __init__(
        self,
        in_ch: int = 3,
        out_ch: int = 1,
        base: int = 32,
        time_dim: int = 128,
        cond_dim: int = 4,
    ) -> None:
        super().__init__()

        self.time_mlp = TimeMLP(
            time_dim
        )

        self.cond_mlp = CondMLP(
            cond_dim,
            time_dim,
        )

        self.enc1 = CondResBlock(
            in_ch,
            base,
            time_dim,
        )

        self.enc2 = CondResBlock(
            base,
            base * 2,
            time_dim,
        )

        self.enc3 = CondResBlock(
            base * 2,
            base * 4,
            time_dim,
        )

        self.pool = nn.MaxPool2d(
            2
        )

        self.mid = CondResBlock(
            base * 4,
            base * 4,
            time_dim,
        )

        self.up2 = nn.ConvTranspose2d(
            base * 4,
            base * 2,
            2,
            stride=2,
        )

        self.dec2 = CondResBlock(
            base * 4,
            base * 2,
            time_dim,
        )

        self.up1 = nn.ConvTranspose2d(
            base * 2,
            base,
            2,
            stride=2,
        )

        self.dec1 = CondResBlock(
            base * 2,
            base,
            time_dim,
        )

        self.out = nn.Conv2d(
            base,
            out_ch,
            1,
        )

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        t_emb = self.time_mlp(
            t
        )

        c_emb = self.cond_mlp(
            cond
        )

        emb = (
            t_emb
            + c_emb
        )

        e1 = self.enc1(
            x,
            emb,
        )

        e2 = self.enc2(
            self.pool(
                e1
            ),
            emb,
        )

        e3 = self.enc3(
            self.pool(
                e2
            ),
            emb,
        )

        b = self.mid(
            e3,
            emb,
        )

        d2 = self.up2(
            b
        )

        if (
            d2.shape[-2:]
            != e2.shape[-2:]
        ):
            d2 = F.interpolate(
                d2,
                size=e2.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        d2 = self.dec2(
            torch.cat(
                [
                    d2,
                    e2,
                ],
                dim=1,
            ),
            emb,
        )

        d1 = self.up1(
            d2
        )

        if (
            d1.shape[-2:]
            != e1.shape[-2:]
        ):
            d1 = F.interpolate(
                d1,
                size=e1.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        d1 = self.dec1(
            torch.cat(
                [
                    d1,
                    e1,
                ],
                dim=1,
            ),
            emb,
        )

        return torch.sigmoid(
            self.out(
                d1
            )
        )
