"""Kou jump diffusion."""

import numpy as np
import numpy.typing as npt
from numpy.random import Generator

from .._compat import override
from ..utils.validation import check_numeric, check_positive_number
from ._jumps import compound_increments
from .jump_diffusion import JumpDiffusion


class KouJumpDiffusion(JumpDiffusion):
    r"""Kou jump diffusion.

    A geometric Brownian motion with jumps whose sizes follow a double
    exponential distribution in the logarithm (Kou, 2002): a jump is upward
    with probability :math:`p`, with an exponential size of rate
    :math:`\eta_1`, and downward otherwise, with an exponential size of rate
    :math:`\eta_2`.

    .. math::

        S_t = S_0 \exp \left( \left( \mu - \frac{\sigma^2}{2}
        - \lambda \kappa \right) t + \sigma W_t + \sum_{k=1}^{N_t} Y_k \right)

    where :math:`N_t` is a Poisson process with rate :math:`\lambda` and
    :math:`\kappa = p \eta_1 / (\eta_1 - 1) + (1 - p) \eta_2 / (\eta_2 + 1) - 1`
    is the mean relative jump. The expected value is :math:`S_0 e^{\mu t}`.
    Upward and downward jumps can have different sizes, which reproduces the
    asymmetry of returns.

    :param drift: the expected rate of return, or :math:`\mu` above
    :param volatility: the volatility between jumps, or :math:`\sigma` above
    :param rate: the rate of the jumps, or :math:`\lambda` above
    :param up_probability: the probability that a jump is upward, or
        :math:`p` above
    :param up_rate: the rate of the size of an upward jump, or
        :math:`\eta_1` above, larger than 1; its mean size is
        :math:`1 / \eta_1`
    :param down_rate: the rate of the size of a downward jump, or
        :math:`\eta_2` above; its mean size is :math:`1 / \eta_2`
    :param t: the right hand endpoint of the time interval :math:`[0,t]`
        for the process
    :param rng: a custom random number generator
    """

    _parallel_paths = True

    def __init__(
        self,
        *,
        drift: float = 0.0,
        volatility: float = 1.0,
        rate: float = 1.0,
        up_probability: float = 0.5,
        up_rate: float = 10.0,
        down_rate: float = 10.0,
        t: float = 1.0,
        rng: Generator | None = None,
    ) -> None:
        super().__init__(drift=drift, volatility=volatility, rate=rate, t=t, rng=rng)

        self.up_probability = up_probability
        self.up_rate = up_rate
        self.down_rate = down_rate

    def __str__(self) -> str:
        return (
            f"Kou jump diffusion with drift {self.drift}, volatility {self.volatility}, "
            f"rate {self.rate}, up probability {self.up_probability}, up rate {self.up_rate} "
            f"and down rate {self.down_rate} on [0, {self.t}]."
        )

    def __repr__(self) -> str:
        return (
            f"KouJumpDiffusion(drift={self.drift}, volatility={self.volatility}, "
            f"rate={self.rate}, up_probability={self.up_probability}, "
            f"up_rate={self.up_rate}, down_rate={self.down_rate}, t={self.t})"
        )

    @property
    def up_probability(self) -> float:
        r"""Probability that a jump is upward, or :math:`p`."""
        return self.__up_probability

    @up_probability.setter
    def up_probability(self, value: float) -> None:
        check_numeric(value=value, name="Up probability")
        if value < 0 or value > 1:
            raise ValueError(f"Up probability value must be between 0 and 1; got {value}.")

        self.__up_probability = float(value)

    @property
    def up_rate(self) -> float:
        r"""Rate of the size of an upward jump, or :math:`\eta_1`."""
        return self.__up_rate

    @up_rate.setter
    def up_rate(self, value: float) -> None:
        check_positive_number(value=value, name="Up rate")
        if value <= 1:
            raise ValueError(
                f"Up rate value must be larger than 1 for the price to have a mean; got {value}."
            )

        self.__up_rate = float(value)

    @property
    def down_rate(self) -> float:
        r"""Rate of the size of a downward jump, or :math:`\eta_2`."""
        return self.__down_rate

    @down_rate.setter
    def down_rate(self, value: float) -> None:
        check_positive_number(value=value, name="Down rate")

        self.__down_rate = float(value)

    @property
    @override
    def mean_relative_jump(self) -> float:
        p = self.up_probability
        return (
            p * self.up_rate / (self.up_rate - 1.0)
            + (1.0 - p) * self.down_rate / (self.down_rate + 1.0)
            - 1.0
        )

    def _jump_sizes(self, rng: Generator, size: int) -> npt.NDArray[np.float64]:
        """Draw the logarithms of jumps from the double exponential distribution."""
        upward = rng.random(size) < self.up_probability
        sizes: npt.NDArray[np.float64] = np.where(
            upward,
            rng.exponential(1.0 / self.up_rate, size),
            -rng.exponential(1.0 / self.down_rate, size),
        )
        return sizes

    @override
    def _jump_increments(
        self,
        expected: float | npt.NDArray[np.float64],
        shape: tuple[int, ...],
    ) -> npt.NDArray[np.float64]:
        return compound_increments(self.rng, expected, shape, self._jump_sizes)
