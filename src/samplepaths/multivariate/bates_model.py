"""Bates model."""

import math

import numpy as np
import numpy.typing as npt
from numpy.random import Generator

from .._compat import override
from ..utils.validation import check_nonnegative_number, check_numeric
from .heston_model import HestonModel


class BatesModel(HestonModel):
    r"""Bates model.

    .. image:: /_static/bates_model.png
        :scale: 50%

    A Heston model whose price also jumps (Bates, 1996):

    .. math::

        dS_t &= (\mu - \lambda k) S_t dt + \sqrt{V_t} S_t dW^S_t
        + S_{t^-} dJ_t \\
        dV_t &= \kappa (\theta - V_t) dt + \xi \sqrt{V_t} dW^V_t

    The jumps arrive at the rate :math:`\lambda`. A jump multiplies the
    price by :math:`e^Y`, where :math:`Y` is normal with mean :math:`m` and
    standard deviation :math:`\delta`, as in a Merton jump diffusion. The
    mean relative jump :math:`k = e^{m + \delta^2 / 2} - 1` is taken out of
    the drift, so that the mean of the price grows at the rate :math:`\mu`.

    The variance and the price between jumps are generated as in
    :py:class:`~samplepaths.multivariate.HestonModel`.

    :param drift: the drift of the price, or :math:`\mu` above
    :param speed: the speed of reversion of the variance, or :math:`\kappa`
    :param mean: the long-run variance, or :math:`\theta`
    :param vol: the volatility of the variance, or :math:`\xi`
    :param correlation: the correlation of the two Wiener processes
    :param rate: the rate of the jumps, or :math:`\lambda`
    :param jump_mean: the mean of the logarithm of a jump, or :math:`m`
    :param jump_volatility: the standard deviation of the logarithm of a
        jump, or :math:`\delta`
    :param initial_variance: the variance at time zero; the long-run
        variance if not given
    :param t: the right hand endpoint of the time interval :math:`[0,t]`
        for the process
    :param rng: a custom random number generator
    """

    def __init__(
        self,
        *,
        drift: float = 0.0,
        speed: float = 1.0,
        mean: float = 0.04,
        vol: float = 0.2,
        correlation: float = -0.5,
        rate: float = 1.0,
        jump_mean: float = -0.1,
        jump_volatility: float = 0.1,
        initial_variance: float | None = None,
        t: float = 1.0,
        rng: Generator | None = None,
    ) -> None:
        super().__init__(
            drift=drift,
            speed=speed,
            mean=mean,
            vol=vol,
            correlation=correlation,
            initial_variance=initial_variance,
            t=t,
            rng=rng,
        )

        self.rate = rate
        self.jump_mean = jump_mean
        self.jump_volatility = jump_volatility

    def __str__(self) -> str:
        return (
            f"Bates model with drift={self.drift}, speed={self.speed}, mean={self.mean}, "
            f"vol={self.vol}, correlation={self.correlation}, rate={self.rate}, "
            f"jump_mean={self.jump_mean}, jump_volatility={self.jump_volatility} "
            f"on [0, {self.t}]"
        )

    def __repr__(self) -> str:
        return (
            f"BatesModel(drift={self.drift}, speed={self.speed}, mean={self.mean}, "
            f"vol={self.vol}, correlation={self.correlation}, rate={self.rate}, "
            f"jump_mean={self.jump_mean}, jump_volatility={self.jump_volatility}, "
            f"initial_variance={self.initial_variance}, t={self.t})"
        )

    @property
    def rate(self) -> float:
        r"""Rate of the jumps, or :math:`\lambda`."""
        return self.__rate

    @rate.setter
    def rate(self, value: float) -> None:
        check_nonnegative_number(value=value, name="Rate")

        self.__rate = float(value)

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
    def mean_relative_jump(self) -> float:
        r"""Mean relative change of the price at a jump, or :math:`k`."""
        return math.exp(self.jump_mean + self.jump_volatility**2 / 2.0) - 1.0

    @override
    def _jumps(
        self, n: int, paths: int, delta_t: float | npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64] | float:
        # The sum of k normal jump sizes is normal, with k times the mean and
        # the variance of one, so no jump needs to be drawn on its own.
        counts = self.rng.poisson(self.rate * delta_t, size=(n, paths))
        jumps: npt.NDArray[np.float64] = self.jump_mean * counts + self.jump_volatility * np.sqrt(
            counts
        ) * self.rng.standard_normal((n, paths))
        jumps -= self.rate * self.mean_relative_jump * delta_t
        return jumps
