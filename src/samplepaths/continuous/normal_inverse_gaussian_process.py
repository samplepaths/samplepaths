"""Normal inverse Gaussian process."""

import math

import numpy as np
import numpy.typing as npt
from numpy.random import Generator

from .._compat import override
from ..base import BaseTimeProcess
from ..utils.validation import (
    check_numeric,
    check_positive_integer,
    check_positive_number,
    times_to_increments,
)


class NormalInverseGaussianProcess(BaseTimeProcess):
    r"""Normal inverse Gaussian process.

    A Brownian motion with drift run on a random clock given by an inverse
    Gaussian process,

    .. math::

        X_t = \mu t + \beta I_t + W_{I_t}

    where :math:`I_t` is an inverse Gaussian process with mean
    :math:`\delta t / \gamma` and :math:`\gamma = \sqrt{\alpha^2 - \beta^2}`.
    Its increments have the normal inverse Gaussian distribution, with
    heavier tails than the normal distribution and a skewness set by
    :math:`\beta`. It is a pure jump process.

    :param alpha: the tail heaviness, or :math:`\alpha` above; a smaller
        value gives heavier tails
    :param beta: the asymmetry, or :math:`\beta` above, smaller than
        ``alpha`` in absolute value
    :param delta: the scale, or :math:`\delta` above
    :param mu: the drift, or :math:`\mu` above
    :param t: the right hand endpoint of the time interval :math:`[0,t]`
        for the process
    :param rng: a custom random number generator
    """

    _parallel_paths = True

    def __init__(
        self,
        *,
        alpha: float = 1.0,
        beta: float = 0.0,
        delta: float = 1.0,
        mu: float = 0.0,
        t: float = 1.0,
        rng: Generator | None = None,
    ) -> None:
        super().__init__(t=t, rng=rng)

        check_positive_number(value=alpha, name="Alpha")
        check_numeric(value=beta, name="Beta")
        self._check_tails(alpha, beta)
        self.__alpha = float(alpha)
        self.__beta = float(beta)

        self.delta = delta
        self.mu = mu

    def __str__(self) -> str:
        return (
            f"Normal inverse Gaussian process with alpha={self.alpha}, beta={self.beta}, "
            f"delta={self.delta}, mu={self.mu} on [0, {self.t}]"
        )

    def __repr__(self) -> str:
        return (
            f"NormalInverseGaussianProcess(alpha={self.alpha}, beta={self.beta}, "
            f"delta={self.delta}, mu={self.mu}, t={self.t})"
        )

    @staticmethod
    def _check_tails(alpha: float, beta: float) -> None:
        """Ensure that the asymmetry is smaller than the tail heaviness."""
        if abs(beta) >= alpha:
            raise ValueError(
                "Beta must be smaller than alpha in absolute value; "
                f"got alpha={alpha}, beta={beta}."
            )

    @property
    def alpha(self) -> float:
        r"""Tail heaviness, or :math:`\alpha`."""
        return self.__alpha

    @alpha.setter
    def alpha(self, value: float) -> None:
        check_positive_number(value=value, name="Alpha")
        self._check_tails(value, self.beta)

        self.__alpha = float(value)

    @property
    def beta(self) -> float:
        r"""Asymmetry, or :math:`\beta`."""
        return self.__beta

    @beta.setter
    def beta(self, value: float) -> None:
        check_numeric(value=value, name="Beta")
        self._check_tails(self.alpha, value)

        self.__beta = float(value)

    @property
    def delta(self) -> float:
        r"""Scale, or :math:`\delta`."""
        return self.__delta

    @delta.setter
    def delta(self, value: float) -> None:
        check_positive_number(value=value, name="Delta")

        self.__delta = float(value)

    @property
    def mu(self) -> float:
        r"""Drift, or :math:`\mu`."""
        return self.__mu

    @mu.setter
    def mu(self, value: float) -> None:
        check_numeric(value=value, name="Mu")

        self.__mu = float(value)

    def _increments(
        self,
        lengths: float | npt.NDArray[np.float64],
        shape: tuple[int, ...],
    ) -> npt.NDArray[np.float64]:
        """Draw the increments of the process over intervals of given lengths."""
        gamma = math.sqrt(self.alpha**2 - self.beta**2)
        scaled = self.delta * lengths

        # The clock: an inverse Gaussian variate with mean delta dt / gamma.
        clock = self.rng.wald(mean=scaled / gamma, scale=scaled**2, size=shape)
        noise = self.rng.standard_normal(shape)

        increments: npt.NDArray[np.float64] = (
            self.mu * lengths + self.beta * clock + np.sqrt(clock) * noise
        )
        return increments

    @override
    def sample(self, n: int) -> npt.NDArray[np.float64]:
        check_positive_integer(n=n, name="Number of increments")
        self.set_times(n)

        return np.concatenate(([0.0], np.cumsum(self._increments(self.t / n, (n,)))))

    @override
    def sample_at(self, times: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        times = np.asarray(times, dtype=np.float64)
        zero = times[0] == 0
        if not zero:
            times = np.insert(times, 0, 0.0)
        lengths = times_to_increments(times=times)

        samples = np.cumsum(self._increments(lengths, lengths.shape))
        if zero:
            samples = np.insert(samples, 0, 0.0)

        return samples

    @override
    def _sample_paths(self, n: int, paths: int) -> npt.NDArray[np.float64]:
        check_positive_integer(n=n, name="Number of increments")
        self.set_times(n)

        increments = self._increments(self.t / n, (paths, n))
        return np.concatenate((np.zeros((paths, 1)), np.cumsum(increments, axis=1)), axis=1)
