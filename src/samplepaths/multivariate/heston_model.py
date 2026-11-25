"""Heston model."""

import numpy as np
import numpy.typing as npt
from numpy.random import Generator

from .._compat import override
from .._square_root import square_root_paths, square_root_values
from .._times import steps_from_zero
from ..utils.validation import check_nonnegative_number, check_numeric, check_positive_number
from ._andersen import log_price_changes
from .vector_process import VectorProcess

#: Number of realizations from which the variances of all realizations are
#: drawn together at each step, instead of one realization after the other.
_VECTORIZE_FROM_PATHS = 8


class HestonModel(VectorProcess):
    r"""Heston model.

    .. image:: /_static/heston_model.png
        :scale: 50%

    A price :math:`S_t` whose variance :math:`V_t` is itself random and
    reverts to a mean (Heston, 1993):

    .. math::

        dS_t &= \mu S_t dt + \sqrt{V_t} S_t dW^S_t \\
        dV_t &= \kappa (\theta - V_t) dt + \xi \sqrt{V_t} dW^V_t

    with Wiener processes :math:`W^S_t` and :math:`W^V_t` of correlation
    :math:`\rho`. A negative correlation makes the variance rise when the
    price falls, which skews the distribution of the returns as observed for
    stocks.

    A realization has two components, the price and the variance. The
    variance is drawn from its exact distribution at each step, a noncentral
    chi-square distribution, so it never becomes negative and does not depend
    on the size of the step. The logarithm of the price uses the
    discretization of Andersen (2008), which approximates the integral of the
    variance over a step by the mean of its two ends.

    :param drift: the drift of the price, or :math:`\mu` above
    :param speed: the speed of reversion of the variance, or :math:`\kappa`
    :param mean: the long-run variance, or :math:`\theta`
    :param vol: the volatility of the variance, or :math:`\xi`
    :param correlation: the correlation of the two Wiener processes, or
        :math:`\rho`
    :param initial_variance: the variance at time zero; the long-run
        variance if not given
    :param t: the right hand endpoint of the time interval :math:`[0,t]`
        for the process
    :param rng: a custom random number generator
    """

    components = ("price", "variance")

    _paths_per_worker = 5000

    def __init__(
        self,
        *,
        drift: float = 0.0,
        speed: float = 1.0,
        mean: float = 0.04,
        vol: float = 0.2,
        correlation: float = -0.5,
        initial_variance: float | None = None,
        t: float = 1.0,
        rng: Generator | None = None,
    ) -> None:
        super().__init__(t=t, rng=rng)

        self.drift = drift
        self.speed = speed
        self.mean = mean
        self.vol = vol
        self.correlation = correlation
        self.initial_variance = initial_variance

    def __str__(self) -> str:
        return (
            f"Heston model with drift={self.drift}, speed={self.speed}, mean={self.mean}, "
            f"vol={self.vol}, correlation={self.correlation} on [0, {self.t}]"
        )

    def __repr__(self) -> str:
        return (
            f"HestonModel(drift={self.drift}, speed={self.speed}, mean={self.mean}, "
            f"vol={self.vol}, correlation={self.correlation}, "
            f"initial_variance={self.initial_variance}, t={self.t})"
        )

    @property
    def drift(self) -> float:
        r"""Drift of the price, or :math:`\mu`."""
        return self.__drift

    @drift.setter
    def drift(self, value: float) -> None:
        check_numeric(value=value, name="Drift")

        self.__drift = float(value)

    @property
    def speed(self) -> float:
        r"""Speed of reversion of the variance, or :math:`\kappa`."""
        return self.__speed

    @speed.setter
    def speed(self, value: float) -> None:
        check_positive_number(value=value, name="Speed")

        self.__speed = float(value)

    @property
    def mean(self) -> float:
        r"""Long-run variance, or :math:`\theta`."""
        return self.__mean

    @mean.setter
    def mean(self, value: float) -> None:
        check_positive_number(value=value, name="Mean")

        self.__mean = float(value)

    @property
    def vol(self) -> float:
        r"""Volatility of the variance, or :math:`\xi`."""
        return self.__vol

    @vol.setter
    def vol(self, value: float) -> None:
        check_positive_number(value=value, name="Volatility")

        self.__vol = float(value)

    @property
    def correlation(self) -> float:
        r"""Correlation of the two Wiener processes, or :math:`\rho`."""
        return self.__correlation

    @correlation.setter
    def correlation(self, value: float) -> None:
        self.__correlation = self._check_correlation(value)

    @property
    def initial_variance(self) -> float:
        """Variance at time zero."""
        return self.mean if self.__initial_variance is None else self.__initial_variance

    @initial_variance.setter
    def initial_variance(self, value: float | None) -> None:
        if value is not None:
            check_nonnegative_number(value=value, name="Initial variance")
            value = float(value)

        self.__initial_variance = value

    def _variances(self, n: int, paths: int) -> npt.NDArray[np.float64]:
        """Generate the variances, in an array of shape ``(n + 1, paths)``."""
        return square_root_paths(
            self.rng,
            n,
            paths,
            speed=self.speed,
            mean=self.mean,
            vol=self.vol,
            initial=self.initial_variance,
            t=self.t,
            vectorize_from=_VECTORIZE_FROM_PATHS,
        )

    def _jumps(
        self, n: int, paths: int, delta_t: float | npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64] | float:
        """Return what jumps add to the logarithm of the price at each step.

        :param n: the number of steps
        :param paths: the number of realizations
        :param delta_t: the length of the steps, a number or a column with
            one length per step
        :returns: an array of shape ``(n, paths)``, or zero for a model
            without jumps
        """
        return 0.0

    def _sample_paths_from(self, n: int, paths: int, initial: float) -> npt.NDArray[np.float64]:
        """Generate several realizations from an initial price."""
        check_positive_number(value=initial, name="Initial")
        self._check_paths(n, paths)

        variances = self._variances(n, paths)
        changes = log_price_changes(self, variances, self.t / n)
        changes += self._jumps(n, paths, self.t / n)

        realizations = np.empty((paths, 2, n + 1))
        realizations[:, 0, 0] = initial
        realizations[:, 0, 1:] = initial * np.exp(np.cumsum(changes, axis=0)).T
        realizations[:, 1, :] = variances.T
        return realizations

    @override
    def _sample_paths(self, n: int, paths: int) -> npt.NDArray[np.float64]:
        return self._sample_paths_from(n, paths, 1.0)

    def sample_with_initial(self, n: int, initial: float = 1.0) -> npt.NDArray[np.float64]:
        """Generate a realization starting from an initial price.

        :param n: the number of increments to generate
        :param initial: the price at time zero
        :returns: the realization, with the price in the first row and the
            variance in the second
        """
        realization: npt.NDArray[np.float64] = self._sample_paths_from(n, 1, initial)[0]
        return realization

    def sample_paths_with_initial(
        self,
        n: int,
        paths: int,
        initial: float = 1.0,
        workers: int = 1,
    ) -> npt.NDArray[np.float64]:
        """Generate several realizations starting from an initial price.

        :param n: the number of increments of each realization
        :param paths: the number of realizations
        :param initial: the price at time zero
        :param workers: the number of threads to use
        :returns: a three-dimensional array of shape ``(paths, 2, n + 1)``
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

    @override
    def sample_at(self, times: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Generate a realization at specified times, from a price of 1.

        :param times: a vector of increasing time values at which to generate
            the realization
        :returns: the realization, with the price in the first row and the
            variance in the second, and one value per time
        """
        return self.sample_at_with_initial(times, 1.0)

    def sample_at_with_initial(
        self,
        times: npt.ArrayLike,
        initial: float = 1.0,
    ) -> npt.NDArray[np.float64]:
        """Generate a realization at specified times from an initial price.

        The variance is exact at every time. The price is as accurate as the
        longest step between two times is short.

        :param times: a vector of increasing time values at which to generate
            the realization
        :param initial: the price at time zero
        :returns: the realization, with the price in the first row and the
            variance in the second, and one value per time
        """
        check_positive_number(value=initial, name="Initial")
        steps, starts_at_zero = steps_from_zero(times)

        variances = square_root_values(
            self.rng,
            steps,
            1,
            speed=self.speed,
            mean=self.mean,
            vol=self.vol,
            initial=self.initial_variance,
            vectorize_from=_VECTORIZE_FROM_PATHS,
        )
        lengths = steps[:, np.newaxis]
        changes = log_price_changes(self, variances, lengths)
        changes += self._jumps(len(steps), 1, lengths)

        realization = np.empty((2, len(steps) + 1))
        realization[0, 0] = initial
        realization[0, 1:] = initial * np.exp(np.cumsum(changes[:, 0]))
        realization[1] = variances[:, 0]
        return realization if starts_at_zero else realization[:, 1:]
