"""Merton jump diffusion."""

import numpy as np
import numpy.typing as npt
from numpy.random import Generator

from .._compat import override
from ..utils.validation import check_nonnegative_number, check_numeric
from .jump_diffusion import JumpDiffusion


class MertonJumpDiffusion(JumpDiffusion):
    r"""Merton jump diffusion.

    A geometric Brownian motion with jumps whose sizes are normally
    distributed in the logarithm (Merton, 1976):

    .. math::

        S_t = S_0 \exp \left( \left( \mu - \frac{\sigma^2}{2}
        - \lambda \kappa \right) t + \sigma W_t + \sum_{k=1}^{N_t} Y_k \right),
        \quad Y_k \sim N(m, \delta^2)

    where :math:`N_t` is a Poisson process with rate :math:`\lambda` and
    :math:`\kappa = e^{m + \delta^2 / 2} - 1` is the mean relative jump. The
    expected value is :math:`S_0 e^{\mu t}`.

    :param drift: the expected rate of return, or :math:`\mu` above
    :param volatility: the volatility between jumps, or :math:`\sigma` above
    :param rate: the rate of the jumps, or :math:`\lambda` above
    :param jump_mean: the mean of the logarithm of a jump, or :math:`m` above
    :param jump_volatility: the standard deviation of the logarithm of a
        jump, or :math:`\delta` above
    :param t: the right hand endpoint of the time interval :math:`[0,t]`
        for the process
    :param rng: a custom random number generator
    """

    def __init__(
        self,
        *,
        drift: float = 0.0,
        volatility: float = 1.0,
        rate: float = 1.0,
        jump_mean: float = 0.0,
        jump_volatility: float = 1.0,
        t: float = 1.0,
        rng: Generator | None = None,
    ) -> None:
        super().__init__(drift=drift, volatility=volatility, rate=rate, t=t, rng=rng)

        self.jump_mean = jump_mean
        self.jump_volatility = jump_volatility

    def __str__(self) -> str:
        return (
            f"Merton jump diffusion with drift {self.drift}, volatility {self.volatility}, "
            f"rate {self.rate}, jump mean {self.jump_mean} and jump volatility "
            f"{self.jump_volatility} on [0, {self.t}]."
        )

    def __repr__(self) -> str:
        return (
            f"MertonJumpDiffusion(drift={self.drift}, volatility={self.volatility}, "
            f"rate={self.rate}, jump_mean={self.jump_mean}, "
            f"jump_volatility={self.jump_volatility}, t={self.t})"
        )

    @property
    def jump_mean(self) -> float:
        r"""Mean of the logarithm of a jump, or :math:`m`."""
        return self.__jump_mean

    @jump_mean.setter
    def jump_mean(self, value: float) -> None:
        check_numeric(value=value, name="Jump mean")

        self.__jump_mean = float(value)

    @property
    def jump_volatility(self) -> float:
        r"""Standard deviation of the logarithm of a jump, or :math:`\delta`."""
        return self.__jump_volatility

    @jump_volatility.setter
    def jump_volatility(self, value: float) -> None:
        check_nonnegative_number(value=value, name="Jump volatility")

        self.__jump_volatility = float(value)

    @property
    @override
    def mean_relative_jump(self) -> float:
        return self._exp(self.jump_mean + self.jump_volatility**2 / 2.0) - 1.0

    @override
    def _jump_increments(
        self,
        expected: float | npt.NDArray[np.float64],
        shape: tuple[int, ...],
    ) -> npt.NDArray[np.float64]:
        # The sum of k normal jump sizes is normal, with k times the mean and
        # the variance of one, so no jump needs to be drawn on its own.
        counts = self.rng.poisson(expected, size=shape)
        noise = self.rng.standard_normal(shape)

        increments: npt.NDArray[np.float64] = (
            self.jump_mean * counts + self.jump_volatility * np.sqrt(counts) * noise
        )
        return increments
