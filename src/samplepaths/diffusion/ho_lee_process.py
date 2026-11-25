"""Ho-Lee process."""

from collections.abc import Callable

import numpy as np
import numpy.typing as npt
from numpy.random import Generator

from .._compat import override
from .._times import steps_from_zero
from ..noise.gaussian_noise import GaussianNoise
from ..utils import ensure_single_arg_constant_function
from ..utils.validation import (
    check_nonnegative_number,
    check_numeric,
    check_numeric_or_single_arg_callable,
    check_positive_integer,
)


class HoLeeProcess(GaussianNoise):
    r"""Ho-Lee process.

    .. image:: /_static/ho_lee_process.png
        :scale: 50%

    A model for the instantaneous interest rate without reversion to a mean
    (Ho and Lee, 1986). The process :math:`X_t` satisfies the following
    stochastic differential equation with Wiener process :math:`W_t`:

    .. math::

        dX_t = \theta_t dt + \sigma dW_t

    The drift :math:`\theta_t` is a number or a function of time. As a
    function it lets the model reproduce the interest rates observed today
    for every maturity.

    The noise is exact. The drift is added at every step with its value at
    the end of the step, which is exact for a constant drift.
    :py:meth:`sample` starts from 0.

    :param drift: the drift, or :math:`\theta_t` above: a number or a
        function of the time
    :param vol: volatility coefficient of the process, or :math:`\sigma`
        above
    :param t: the right hand endpoint of the time interval :math:`[0,t]`
        for the process
    :param rng: a custom random number generator
    """

    def __init__(
        self,
        *,
        drift: Callable[[float], float] | float = 0.0,
        vol: float = 1.0,
        t: float = 1.0,
        rng: Generator | None = None,
    ) -> None:
        super().__init__(t=t, rng=rng)

        self.drift = drift
        self.vol = vol

    def __str__(self) -> str:
        return f"Ho-Lee process with drift={self.drift}, vol={self.vol} on [0, {self.t}]"

    def __repr__(self) -> str:
        return f"HoLeeProcess(drift={self.drift}, vol={self.vol}, t={self.t})"

    @property
    def drift(self) -> Callable[[float], float]:
        r"""Drift, or :math:`\theta_t`, as a function of the time."""
        return self.__drift

    @drift.setter
    def drift(self, value: Callable[[float], float] | float) -> None:
        check_numeric_or_single_arg_callable(value=value, name="drift")

        self.__drift = ensure_single_arg_constant_function(value=value)
        self.__drift_value = None if callable(value) else float(value)

    @property
    def vol(self) -> float:
        r"""Volatility coefficient, or :math:`\sigma`."""
        return self.__vol

    @vol.setter
    def vol(self, value: float) -> None:
        check_nonnegative_number(value=value, name="Volatility")

        self.__vol = float(value)

    def _accumulate(
        self, noise: npt.NDArray[np.float64], initial: float
    ) -> npt.NDArray[np.float64]:
        """Turn realizations of the noise, one per row, into the process."""
        check_numeric(value=initial, name="Initial")
        n = noise.shape[-1]
        delta_t = self.t / n

        if self.__drift_value is not None:
            drifts: npt.NDArray[np.float64] | float = self.__drift_value * delta_t
        else:
            drifts = delta_t * np.array(
                [float(self.__drift(time)) for time in self.times[1:].tolist()]
            )

        realizations = np.empty((*noise.shape[:-1], n + 1))
        realizations[..., 0] = float(initial)
        np.cumsum(drifts + self.vol * noise, axis=-1, out=realizations[..., 1:])
        realizations[..., 1:] += float(initial)
        return realizations

    @override
    def sample(self, n: int) -> npt.NDArray[np.float64]:
        return self.sample_with_initial(n, 0.0)

    def sample_with_initial(self, n: int, initial: float = 0.0) -> npt.NDArray[np.float64]:
        """Generate a realization starting from an initial value.

        :param n: the number of increments to generate
        :param initial: the initial value of the process
        :returns: the realization
        """
        check_positive_integer(n=n)
        return self._accumulate(self._sample_gaussian_noise(n), initial)

    @override
    def sample_at(self, times: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Generate a realization at specified times, starting from 0.

        :param times: a vector of increasing time values at which to generate
            the realization
        :returns: the realization
        """
        return self.sample_at_with_initial(times, 0.0)

    def sample_at_with_initial(
        self,
        times: npt.ArrayLike,
        initial: float = 0.0,
    ) -> npt.NDArray[np.float64]:
        """Generate a realization at specified times from an initial value.

        The noise is exact. The drift is added at every step with its value
        at the end of the step, which is exact for a constant drift.

        :param times: a vector of increasing time values at which to generate
            the realization
        :param initial: the value of the process at time zero
        :returns: the realization
        """
        check_numeric(value=initial, name="Initial")
        steps, starts_at_zero = steps_from_zero(times)

        if self.__drift_value is not None:
            drifts: npt.NDArray[np.float64] | float = self.__drift_value
        else:
            drifts = np.array([float(self.__drift(time)) for time in np.cumsum(steps).tolist()])

        changes = drifts * steps + self.vol * self.rng.normal(scale=np.sqrt(steps))
        values = float(initial) + np.concatenate(([0.0], np.cumsum(changes)))
        return values if starts_at_zero else values[1:]

    def _sample_paths_from(self, n: int, paths: int, initial: float) -> npt.NDArray[np.float64]:
        """Generate several realizations from an initial value."""
        return self._accumulate(self._sample_gaussian_noise_paths(n, paths), initial)

    @override
    def _sample_paths(self, n: int, paths: int) -> npt.NDArray[np.float64]:
        return self._sample_paths_from(n, paths, 0.0)

    def sample_paths_with_initial(
        self,
        n: int,
        paths: int,
        initial: float = 0.0,
        workers: int = 1,
    ) -> npt.NDArray[np.float64]:
        """Generate several realizations starting from an initial value.

        :param n: the number of increments of each realization
        :param paths: the number of realizations
        :param initial: the initial value of the process
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
