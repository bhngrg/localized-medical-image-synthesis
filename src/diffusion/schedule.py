"""
Forward diffusion schedule utilities extracted from notebook Cell 5.

The numerical definitions in this module intentionally reproduce the reference
notebook:

    betas = torch.linspace(beta_start, beta_end, timesteps)
    alphas = 1 - betas
    alpha_bars = cumprod(alphas)
    q(x_t | x_0) = sqrt(alpha_bar_t) * x_0
                   + sqrt(1 - alpha_bar_t) * noise

The notebook stores these tensors as globals. The modular implementation keeps
them inside ``DiffusionSchedule`` so multiple schedules can coexist without
hidden global state.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


DEFAULT_TIMESTEPS = 200
DEFAULT_BETA_START = 1.0e-4
DEFAULT_BETA_END = 0.02


def make_beta_schedule(
    timesteps: int = DEFAULT_TIMESTEPS,
    beta_start: float = DEFAULT_BETA_START,
    beta_end: float = DEFAULT_BETA_END,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """
    Create the linear beta schedule used by the reference notebook.

    Parameters
    ----------
    timesteps
        Number of diffusion steps. Notebook default: 200.
    beta_start
        First beta value. Notebook default: 1e-4.
    beta_end
        Final beta value. Notebook default: 0.02.
    device
        Optional PyTorch device for the returned tensor.
    dtype
        Optional PyTorch dtype. If omitted, ``torch.linspace`` uses PyTorch's
        current default dtype, matching the notebook.

    Returns
    -------
    torch.Tensor
        One-dimensional beta schedule with shape ``[timesteps]``.
    """
    if timesteps <= 0:
        raise ValueError(
            "timesteps must be positive."
        )

    if beta_start <= 0:
        raise ValueError(
            "beta_start must be positive."
        )

    if beta_end <= 0:
        raise ValueError(
            "beta_end must be positive."
        )

    if beta_start > beta_end:
        raise ValueError(
            "beta_start must be less than or equal to beta_end."
        )

    kwargs = {
        "device": device,
    }

    if dtype is not None:
        kwargs[
            "dtype"
        ] = dtype

    return torch.linspace(
        beta_start,
        beta_end,
        timesteps,
        **kwargs,
    )


@dataclass
class DiffusionSchedule:
    """
    Precomputed tensors for the notebook's forward diffusion process.

    Parameters
    ----------
    timesteps
        Number of diffusion steps. Notebook default: 200.
    beta_start
        First beta value. Notebook default: 1e-4.
    beta_end
        Final beta value. Notebook default: 0.02.
    device
        Device on which schedule tensors are stored.
    dtype
        Optional tensor dtype. When omitted, PyTorch's default dtype is used,
        matching the notebook.
    """

    timesteps: int = DEFAULT_TIMESTEPS
    beta_start: float = DEFAULT_BETA_START
    beta_end: float = DEFAULT_BETA_END
    device: torch.device | str | None = None
    dtype: torch.dtype | None = None

    def __post_init__(
        self,
    ) -> None:
        self.betas = make_beta_schedule(
            timesteps=self.timesteps,
            beta_start=self.beta_start,
            beta_end=self.beta_end,
            device=self.device,
            dtype=self.dtype,
        )

        self.alphas = (
            1.0
            - self.betas
        )

        self.alpha_bars = torch.cumprod(
            self.alphas,
            dim=0,
        )

        self.sqrt_alpha_bars = torch.sqrt(
            self.alpha_bars
        )

        self.sqrt_one_minus_alpha_bars = torch.sqrt(
            1.0
            - self.alpha_bars
        )

    @property
    def device_resolved(
        self,
    ) -> torch.device:
        """Return the actual device holding the schedule tensors."""
        return self.betas.device

    def q_sample(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        """
        Sample ``x_t`` from the forward diffusion process.

        This reproduces notebook Cell 5 exactly:

        ``sqrt(alpha_bar_t) * x0 + sqrt(1 - alpha_bar_t) * noise``

        Parameters
        ----------
        x0
            Clean image tensor with shape ``[B, C, H, W]``.
        t
            One-dimensional integer timestep tensor with shape ``[B]``.
        noise
            Noise tensor with the same shape as ``x0``.

        Returns
        -------
        torch.Tensor
            Noised image tensor with the same shape as ``x0``.
        """
        if x0.shape != noise.shape:
            raise ValueError(
                "x0 and noise must have identical shapes.\n"
                f"x0 shape: {tuple(x0.shape)}\n"
                f"noise shape: {tuple(noise.shape)}"
            )

        if x0.ndim != 4:
            raise ValueError(
                "x0 must have shape [B, C, H, W].\n"
                f"Observed shape: {tuple(x0.shape)}"
            )

        if t.ndim != 1:
            raise ValueError(
                "t must be one-dimensional with shape [B].\n"
                f"Observed shape: {tuple(t.shape)}"
            )

        if t.shape[0] != x0.shape[0]:
            raise ValueError(
                "The number of timesteps must equal the batch size.\n"
                f"Batch size: {x0.shape[0]}\n"
                f"Timesteps: {t.shape[0]}"
            )

        if not torch.is_floating_point(
            x0
        ):
            raise ValueError(
                "x0 must be a floating-point tensor."
            )

        if t.dtype not in (
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        ):
            raise ValueError(
                "t must contain integer timestep indices."
            )

        if t.numel() > 0:
            t_min = int(
                t.min().item()
            )
            t_max = int(
                t.max().item()
            )

            if (
                t_min < 0
                or t_max >= self.timesteps
            ):
                raise ValueError(
                    "Timestep index is outside the valid range.\n"
                    f"Valid range: 0-{self.timesteps - 1}\n"
                    f"Observed min/max: {t_min}/{t_max}"
                )

        if x0.device != self.betas.device:
            raise ValueError(
                "x0 and the diffusion schedule must be on the same device.\n"
                f"x0 device: {x0.device}\n"
                f"schedule device: {self.betas.device}"
            )

        if noise.device != x0.device:
            raise ValueError(
                "noise and x0 must be on the same device."
            )

        if t.device != x0.device:
            raise ValueError(
                "t and x0 must be on the same device."
            )

        sqrt_ab = self.sqrt_alpha_bars[
            t
        ].view(
            -1,
            1,
            1,
            1,
        )

        sqrt_omab = self.sqrt_one_minus_alpha_bars[
            t
        ].view(
            -1,
            1,
            1,
            1,
        )

        return (
            sqrt_ab * x0
            + sqrt_omab * noise
        )

    def to(
        self,
        device: torch.device | str,
    ) -> "DiffusionSchedule":
        """
        Move all precomputed schedule tensors to another device.

        Returns ``self`` for convenient use, e.g.
        ``schedule = DiffusionSchedule().to(device)``.
        """
        self.betas = self.betas.to(
            device
        )

        self.alphas = self.alphas.to(
            device
        )

        self.alpha_bars = self.alpha_bars.to(
            device
        )

        self.sqrt_alpha_bars = self.sqrt_alpha_bars.to(
            device
        )

        self.sqrt_one_minus_alpha_bars = (
            self.sqrt_one_minus_alpha_bars.to(
                device
            )
        )

        self.device = device

        return self
