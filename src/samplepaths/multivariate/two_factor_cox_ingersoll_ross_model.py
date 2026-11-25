"""Two-factor Cox-Ingersoll-Ross model."""

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
from numpy.random import Generator

from .._compat import override
from .._square_root import square_root_paths
from ..utils.validation import check_coefficients
from .vector_process import VectorProcess

#: Number of realizations from which the values of all realizations are
#: drawn together at each step, instead of one realization after the other.
_VECTORIZE_FROM_PATHS = 8

Pair = float | Sequence[float]


def _pair(value: object, name: str, *, positive: bool) -> tuple[float, float]:
    """Check one value for each factor, or one for both."""
    values = check_coefficients(value=value, name=name)
    if len(values) == 1:
        values = values * 2
    if len(values) != 2:
        raise ValueError(f"{name} must hold one value for each of the two factors; got {values}.")
    if min(values) < 0 or (positive and min(values) == 0):
        kind = "positive" if positive else "nonnegative"
        raise ValueError(f"{name} values must be {kind}; got {values}.")
    return values[0], values[1]


class TwoFactorCoxIngersollRossModel(VectorProcess):
    r"""Two-factor Cox-Ingersoll-Ross model.

    .. image:: /_static/two_factor_cox_ingersoll_ross_model.png
        :scale: 50%

    An interest rate which is the sum of two independent Cox-Ingersoll-Ross
    processes (Longstaff and Schwartz, 1992):

    .. math::

        r_t &= X^1_t + X^2_t \\
        dX^i_t &= \theta_i (\mu_i - X^i_t) dt + \sigma_i \sqrt{X^i_t} dW^i_t

    One factor is usually fast and the other slow, which gives the rate a
    short-lived and a lasting source of change. The rate is never negative.

    A realization has three components: the rate and the two factors. The
    factors are drawn from their exact distributions at each step, so the
    realizations do not depend on the number of steps.

    :param speed: the speeds of reversion :math:`\theta_1, \theta_2`,
        positive; a single number is used for both factors
    :param mean: the means :math:`\mu_1, \mu_2`, nonnegative
    :param vol: the volatility coefficients :math:`\sigma_1, \sigma_2`,
        nonnegative
    :param initial: the values of the two factors at time zero; their means
        if not given
    :param t: the right hand endpoint of the time interval :math:`[0,t]`
        for the process
    :param rng: a custom random number generator
    """

    components = ("rate", "first factor", "second factor")

    _paths_per_worker = 5000

    def __init__(
        self,
        *,
        speed: Pair = (2.0, 0.2),
        mean: Pair = (0.02, 0.03),
        vol: Pair = (0.1, 0.05),
        initial: Pair | None = None,
        t: float = 1.0,
        rng: Generator | None = None,
    ) -> None:
        super().__init__(t=t, rng=rng)

        self.speed = speed
        self.mean = mean
        self.vol = vol
        self.initial = initial

    def __str__(self) -> str:
        return (
            f"Two-factor Cox-Ingersoll-Ross model with speed={self.speed}, mean={self.mean}, "
            f"vol={self.vol} on [0, {self.t}]"
        )

    def __repr__(self) -> str:
        return (
            f"TwoFactorCoxIngersollRossModel(speed={self.speed}, mean={self.mean}, "
            f"vol={self.vol}, initial={self.initial}, t={self.t})"
        )

    @property
    def speed(self) -> tuple[float, float]:
        r"""Speeds of reversion of the factors, or :math:`\theta_i`."""
        return self.__speed

    @speed.setter
    def speed(self, value: Pair) -> None:
        self.__speed = _pair(value, "Speed", positive=True)

    @property
    def mean(self) -> tuple[float, float]:
        r"""Means of the factors, or :math:`\mu_i`."""
        return self.__mean

    @mean.setter
    def mean(self, value: Pair) -> None:
        self.__mean = _pair(value, "Mean", positive=False)

    @property
    def vol(self) -> tuple[float, float]:
        r"""Volatility coefficients of the factors, or :math:`\sigma_i`."""
        return self.__vol

    @vol.setter
    def vol(self, value: Pair) -> None:
        self.__vol = _pair(value, "Volatility", positive=False)

    @property
    def initial(self) -> tuple[float, float]:
        """Values of the two factors at time zero."""
        return self.mean if self.__initial is None else self.__initial

    @initial.setter
    def initial(self, value: Pair | None) -> None:
        self.__initial = None if value is None else _pair(value, "Initial", positive=False)

    @override
    def _sample_paths(self, n: int, paths: int) -> npt.NDArray[np.float64]:
        self._check_paths(n, paths)

        realizations = np.empty((paths, 3, n + 1))
        for index in range(2):
            factor = square_root_paths(
                self.rng,
                n,
                paths,
                speed=self.speed[index],
                mean=self.mean[index],
                vol=self.vol[index],
                initial=self.initial[index],
                t=self.t,
                vectorize_from=_VECTORIZE_FROM_PATHS,
            )
            realizations[:, index + 1, :] = factor.T
        np.add(realizations[:, 1, :], realizations[:, 2, :], out=realizations[:, 0, :])
        return realizations
