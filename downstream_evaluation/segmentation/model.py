#!/usr/bin/env python3

from __future__ import annotations

import torch
import torch.nn as nn


def double_conv(
    in_channels: int,
    out_channels: int,
) -> nn.Sequential:
    """
    Two consecutive 3x3 convolution + ReLU layers.

    This follows the vanilla U-Net block used in the reference
    Low-Grade-Glioma-Segmentation implementation.
    """
    return nn.Sequential(
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
        ),
        nn.ReLU(inplace=True),
        nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
        ),
        nn.ReLU(inplace=True),
    )


class VanillaUNet(nn.Module):
    """
    Vanilla 2D U-Net for binary tumor segmentation.

    The architecture follows the reference LGG implementation, with two
    intentional adaptations for the present downstream experiment:

    1. The input has one channel because the current BraTS downstream
       contract uses FLAIR only.
    2. The network returns logits rather than probabilities. Sigmoid is
       applied explicitly by the loss/metric code when probabilities are
       needed.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
    ) -> None:
        super().__init__()

        self.conv_down1 = double_conv(
            in_channels,
            64,
        )
        self.conv_down2 = double_conv(
            64,
            128,
        )
        self.conv_down3 = double_conv(
            128,
            256,
        )
        self.conv_down4 = double_conv(
            256,
            512,
        )

        self.maxpool = nn.MaxPool2d(2)

        self.upsample = nn.Upsample(
            scale_factor=2,
            mode="bilinear",
            align_corners=True,
        )

        self.conv_up3 = double_conv(
            256 + 512,
            256,
        )
        self.conv_up2 = double_conv(
            128 + 256,
            128,
        )
        self.conv_up1 = double_conv(
            128 + 64,
            64,
        )

        self.last_conv = nn.Conv2d(
            64,
            out_channels,
            kernel_size=1,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        conv1 = self.conv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.conv_down2(x)
        x = self.maxpool(conv2)

        conv3 = self.conv_down3(x)
        x = self.maxpool(conv3)

        x = self.conv_down4(x)
        x = self.upsample(x)

        x = torch.cat(
            [x, conv3],
            dim=1,
        )

        x = self.conv_up3(x)
        x = self.upsample(x)

        x = torch.cat(
            [x, conv2],
            dim=1,
        )

        x = self.conv_up2(x)
        x = self.upsample(x)

        x = torch.cat(
            [x, conv1],
            dim=1,
        )

        x = self.conv_up1(x)

        logits = self.last_conv(x)

        return logits
