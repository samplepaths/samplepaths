"""Base class of the geometric Brownian motions with jumps."""

import math
from abc import abstractmethod

import numpy as np
import numpy.typing as npt
from numpy.random import Generator

from .._compat import override
from ..noise.gaussian_noise import GaussianNoise
from ..utils.validation import (
    check_nonnegative_number,
    check_numeric,
    check_positive_number,
    times_to_increments,
)


class JumpDiffusion(GaussianNoise):
    r"""Base class of the geometric Brownian motions with jumps.

    A price :math:`S_t` which follows a geometric Brownian motion between
    the jumps of a Poisson process with rate :math:`\lambda`, and is
    multiplied by :math:`e^{Y}` at each jump:

    .. math::

        S_t = S_0 \exp \left( \left( \mu - \frac{\sigma^2}{2}
        - \lambda \kappa \right) t + \sigma W_t + \sum_{k=1}^{N_t} Y_k \right)

    where :math:`\kappa = E[e^{Y} - 1]` is the mean relative jump. The term
    :math:`\lambda \kappa` compensates the jumps, so that the expected value
    is :math:`S_0 e^{\mu t}` whatever the jumps.

    A subclass defines the distribution of the jump sizes :math:`Y`.

    :param drift: the expected rate of return, or :math:`\mu` above
    :param volatility: the volatility between jumps, or :math:`\sigma` above
    :param rate: the rate of the jumps, or :math:`\lambda` above
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
        t: float = 1.0,
        rng: Generator | None = None,
    ) -> None:
        super().__init__(t=t, rng=rng)

        self.drift = drift
        self.volatility = volatility
        self.rate = rate

    @property
    def drift(self) -> float:
        r"""Expected rate of return, or :math:`\mu`."""
        return self.__drift

    @drift.setter
    def drift(self, value: float) -> None:
        check_numeric(value=value, name="Drift")

        self.__drift = float(value)

    @property
    def volatility(self) -> float:
        r"""Volatility between jumps, or :math:`\sigma`."""
        return self.__volatility

    @volatility.setter
    def volatility(self, value: float) -> None:
        check_nonnegative_number(value=value, name="Volatility")

        self.__volatility = float(value)

    @property
    def rate(self) -> float:
        r"""Rate of the jumps, or :math:`\lambda`."""
        return self.__rate

    @rate.setter
    def rate(self, value: float) -> None:
        check_nonnegative_number(value=value, name="Rate")

        self.__rate = float(value)

    @property
    @abstractmethod
    def mean_relative_jump(self) -> float:
        r"""Mean relative jump, or :math:`\kappa = E[e^{Y} - 1]`."""

    @abstractmethod
    def _jump_increments(
        self,
        expected: float | npt.NDArray[np.float64],
        shape: tuple[int, ...],
    ) -> npt.NDArray[np.float64]:
        """Draw the sums of the jump sizes over the increments.

        :param expected: the expected number of jumps of each increment
        :param shape: the shape of the result
        :returns: the sums of the jump sizes
        """

    def _log_increments(
        self,
        noise: npt.NDArray[np.float64],
        lengths: float | npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Combine the noise and the jumps into the increments of the logarithm."""
        trend = self.drift - self.volatility**2 / 2.0 - self.rate * self.mean_relative_jump
        jumps = self._jump_increments(self.rate * lengths, noise.shape)

        increments: npt.NDArray[np.float64] = trend * lengths + self.volatility * noise + jumps
        return increments

    def _sample_from(self, n: int, initial: float) -> npt.NDArray[np.float64]:
        """Generate a realization from an initial value."""
        check_positive_number(value=initial, name="Initial")

        noise = self._sample_gaussian_noise(n)
        logarithm = np.concatenate(([0.0], np.cumsum(self._log_increments(noise, self.t / n))))

        realization: npt.NDArray[np.float64] = initial * np.exp(logarithm)
        return realization

    def _sample_at_from(
        self,
        times: npt.NDArray[np.float64],
        initial: float,
    ) -> npt.NDArray[np.float64]:
        """Generate a realization at specified times from an initial value."""
        check_positive_number(value=initial, name="Initial")

        times = np.asarray(times, dtype=np.float64)
        zero = times[0] == 0
        lengths = times_to_increments(times=times if zero else np.insert(times, 0, 0.0))

        noise = self._sample_gaussian_noise_at(times)
        logarithm = np.cumsum(self._log_increments(noise, lengths))
        if zero:
            logarithm = np.insert(logarithm, 0, 0.0)

        realization: npt.NDArray[np.float64] = initial * np.exp(logarithm)
        return realization

    def _sample_paths_from(
        self,
        n: int,
        paths: int,
        initial: float,
    ) -> npt.NDArray[np.float64]:
        """Generate several realizations from an initial value."""
        check_positive_number(value=initial, name="Initial")

        noise = self._sample_gaussian_noise_paths(n, paths)
        logarithm = np.concatenate(
            (
                np.zeros((paths, 1)),
                np.cumsum(self._log_increments(noise, self.t / n), axis=1),
            ),
            axis=1,
        )

        realizations: npt.NDArray[np.float64] = initial * np.exp(logarithm)
        return realizations

    @override
    def sample(self, n: int) -> npt.NDArray[np.float64]:
        return self._sample_from(n, 1.0)

    @override
    def sample_at(self, times: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return self._sample_at_from(times, 1.0)

    def sample_with_initial(self, n: int, initial: float = 1.0) -> npt.NDArray[np.float64]:
        """Generate a realization starting from an initial value.

        :param n: the number of increments to generate
        :param initial: the initial value of the process :math:`S_0`
        :returns: the realization
        """
        return self._sample_from(n, initial)

    def sample_at_with_initial(
        self,
        times: npt.NDArray[np.float64],
        initial: float = 1.0,
    ) -> npt.NDArray[np.float64]:
        """Generate a realization at specified times from an initial value.

        :param times: a vector of increasing time values at which to generate
            the realization
        :param initial: the initial value of the process :math:`S_0`
        :returns: the realization
        """
        return self._sample_at_from(times, initial)

    @override
    def _sample_paths(self, n: int, paths: int) -> npt.NDArray[np.float64]:
        return self._sample_paths_from(n, paths, 1.0)

    def sample_paths_with_initial(
        self,
        n: int,
        paths: int,
        initial: float = 1.0,
        workers: int = 1,
    ) -> npt.NDArray[np.float64]:
        """Generate several realizations starting from an initial value.

        :param n: the number of increments of each realization
        :param paths: the number of realizations
        :param initial: the initial value of the process :math:`S_0`
        :param workers: the number of threads to use
        :returns: a two-dimensional array with one realization per row
        """
        return np.asarray(
            self._run_paths(
                lambda process, count: process._sample_paths_from(n, count, initial),
                n,
                paths,
                workers,
            ),
            dtype=np.float64,
        )

    @staticmethod
    def _exp(value: float) -> float:
        """Exponential which overflows to infinity instead of raising."""
        try:
            return math.exp(value)
        except OverflowError:
            return math.inf
