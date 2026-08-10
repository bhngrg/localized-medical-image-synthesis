"""
Mean-field Gaussian tensor parameters used by BR-LoRA.

Each adapted LoRA tensor is represented by a diagonal-Gaussian variational
posterior with trainable posterior mean and posterior rho parameters.

The posterior standard deviation is

    softplus(posterior_rho) + minimum_std

and samples are generated using the reparameterization trick.

Setting sample=False returns the posterior mean directly, which provides the
deterministic posterior-mean mode used when Bayesian realizations are disabled.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .base import AdaptationError


class VariationalParameterError(
    AdaptationError
):
    """Raised when a variational parameter is configured incorrectly."""


class DiagonalGaussianParameter(
    nn.Module
):
    """Trainable diagonal-Gaussian distribution over one tensor."""

    def __init__(
        self,
        initial_mean: Tensor,
        *,
        initial_std: float,
        prior_mean: float = 0.0,
        prior_std: float = 1.0,
        minimum_std: float = 1e-8,
    ) -> None:
        super().__init__()

        if not isinstance(
            initial_mean,
            Tensor,
        ):
            raise VariationalParameterError(
                "`initial_mean` must be a torch.Tensor."
            )

        if not initial_mean.is_floating_point():
            raise VariationalParameterError(
                "`initial_mean` must have a floating-point dtype."
            )

        if initial_mean.numel() == 0:
            raise VariationalParameterError(
                "`initial_mean` must contain at least one element."
            )

        initial_std_value = (
            _validate_positive_finite(
                initial_std,
                name="initial_std",
            )
        )

        prior_mean_value = (
            _validate_finite(
                prior_mean,
                name="prior_mean",
            )
        )

        prior_std_value = (
            _validate_positive_finite(
                prior_std,
                name="prior_std",
            )
        )

        minimum_std_value = (
            _validate_positive_finite(
                minimum_std,
                name="minimum_std",
            )
        )

        if (
            initial_std_value
            <= minimum_std_value
        ):
            raise VariationalParameterError(
                "`initial_std` must be greater than `minimum_std`."
            )

        rho_value = _inverse_softplus(
            initial_std_value
            - minimum_std_value
        )

        self.posterior_mean = nn.Parameter(
            initial_mean.detach().clone()
        )

        self.posterior_rho = nn.Parameter(
            torch.full_like(
                initial_mean,
                fill_value=rho_value,
            )
        )

        self.register_buffer(
            "prior_mean",
            torch.tensor(
                prior_mean_value,
                device=initial_mean.device,
                dtype=initial_mean.dtype,
            ),
        )

        self.register_buffer(
            "prior_std",
            torch.tensor(
                prior_std_value,
                device=initial_mean.device,
                dtype=initial_mean.dtype,
            ),
        )

        self.register_buffer(
            "minimum_std",
            torch.tensor(
                minimum_std_value,
                device=initial_mean.device,
                dtype=initial_mean.dtype,
            ),
        )

    @property
    def posterior_std(
        self,
    ) -> Tensor:
        """Return the positive posterior standard-deviation tensor."""

        return (
            F.softplus(
                self.posterior_rho
            )
            + self.minimum_std
        )

    def rsample(
        self,
        *,
        sample: bool = True,
        generator: torch.Generator
        | None = None,
    ) -> Tensor:
        """
        Return a reparameterized posterior draw or the posterior mean.

        sample=True
            Draw an independent Bayesian realization.

        sample=False
            Return the posterior mean without generating random numbers.
        """

        if not isinstance(
            sample,
            bool,
        ):
            raise VariationalParameterError(
                "`sample` must be a bool."
            )

        if not sample:
            return self.posterior_mean

        epsilon = torch.randn(
            self.posterior_mean.shape,
            device=self.posterior_mean.device,
            dtype=self.posterior_mean.dtype,
            generator=generator,
        )

        return (
            self.posterior_mean
            + self.posterior_std
            * epsilon
        )

    def kl_divergence(
        self,
        *,
        reduction: str = "sum",
    ) -> Tensor:
        """
        Return analytic KL divergence from posterior to prior.

        Supported reductions:
        - none
        - sum
        - mean
        """

        posterior_std = (
            self.posterior_std
        )

        variance_ratio = (
            posterior_std.square()
            / self.prior_std.square()
        )

        squared_mean_difference = (
            (
                self.posterior_mean
                - self.prior_mean
            ).square()
            / self.prior_std.square()
        )

        elementwise_kl = 0.5 * (
            variance_ratio
            + squared_mean_difference
            - 1.0
            + 2.0
            * (
                torch.log(
                    self.prior_std
                )
                - torch.log(
                    posterior_std
                )
            )
        )

        if reduction == "none":
            return elementwise_kl

        if reduction == "sum":
            return elementwise_kl.sum()

        if reduction == "mean":
            return elementwise_kl.mean()

        raise VariationalParameterError(
            "`reduction` must be one of "
            "'none', 'sum', or 'mean'."
        )


def _validate_finite(
    value: float,
    *,
    name: str,
) -> float:
    """Validate a finite numeric scalar."""

    if (
        isinstance(
            value,
            bool,
        )
        or not isinstance(
            value,
            (
                int,
                float,
            ),
        )
    ):
        raise VariationalParameterError(
            f"`{name}` must be numeric."
        )

    converted = float(
        value
    )

    if not math.isfinite(
        converted
    ):
        raise VariationalParameterError(
            f"`{name}` must be finite."
        )

    return converted


def _validate_positive_finite(
    value: float,
    *,
    name: str,
) -> float:
    """Validate a finite positive scalar."""

    converted = _validate_finite(
        value,
        name=name,
    )

    if converted <= 0.0:
        raise VariationalParameterError(
            f"`{name}` must be positive."
        )

    return converted


def _inverse_softplus(
    value: float,
) -> float:
    """Return the inverse softplus of a positive scalar."""

    if value > 20.0:
        return value

    return math.log(
        math.expm1(
            value
        )
    )


__all__ = [
    "DiagonalGaussianParameter",
    "VariationalParameterError",
]