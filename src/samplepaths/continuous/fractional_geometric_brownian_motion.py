"""Fractional geometric Brownian motion."""

import numpy as np
import numpy.typing as npt
from numpy.random import Generator

from .._compat import override
from ..base import BaseTimeProcess
from ..noise._driver import follow
from ..noise.fractional_gaussian_noise import FractionalGaussianNoise
from ..utils.validation import (
    check_nonnegative_number,
    check_numeric,
    check_positive_integer,
    check_positive_number,
)


class FractionalGeometricBrownianMotion(BaseTimeProcess):
    r"""Fractional geometric Brownian motion.

    .. image:: /_static/fractional_geometric_brownian_motion.png
        :scale: 50%

    A geometric Brownian motion driven by a fractional Brownian motion
    :math:`B^H_t` of Hurst parameter :math:`H`, with initial value
    :math:`S_0`:

    .. math::

        S_t = S_0 \exp \left( \mu t - \frac{\sigma^2}{2} t^{2H}
        + \sigma B^H_t \right)

    This is the solution of :math:`dS_t = \mu S_t dt + \sigma S_t dB^H_t`
    with the Wick product, for which the mean is :math:`S_0 e^{\mu t}`
    whatever the Hurst parameter. With :math:`H = 1/2` it is a geometric
    Brownian motion. With :math:`H > 1/2` rises tend to follow rises, and
    with :math:`H < 1/2` rises tend to follow falls.

    Realizations are exact: the fractional Brownian motion is generated with
    the Davies-Harte method.

    :param hurst: the Hurst parameter :math:`H` in :math:`(0,1)`
    :param drift: the parameter :math:`\mu`
    :param volatility: the parameter :math:`\sigma`
    :param t: the right hand endpoint of the time interval :math:`[0,t]`
        for the process
    :param rng: a custom random number generator
    """

    _parallel_paths = True

    def __init__(
        self,
        *,
        hurst: float = 0.7,
        drift: float = 0.0,
        volatility: float = 1.0,
        t: float = 1.0,
        rng: Generator | None = None,
    ) -> None:
        super().__init__(t=t, rng=rng)

        self.__noise = FractionalGaussianNoise(hurst=hurst, t=self.t, rng=self.rng)
        self.drift = drift
        self.volatility = volatility

    def __str__(self) -> str:
        return (
            f"Fractional geometric Brownian motion with Hurst {self.hurst}, drift {self.drift} "
            f"and volatility {self.volatility} on [0, {self.t}]."
        )

    def __repr__(self) -> str:
        return (
            f"FractionalGeometricBrownianMotion(hurst={self.hurst}, drift={self.drift}, "
            f"volatility={self.volatility}, t={self.t})"
        )

    @property
    def hurst(self) -> float:
        """Hurst parameter."""
        return self.__noise.hurst

    @hurst.setter
    def hurst(self, value: float) -> None:
        self.__noise.hurst = value

    @property
    def drift(self) -> float:
        r"""Drift coefficient, or :math:`\mu`."""
        return self.__drift

    @drift.setter
    def drift(self, value: float) -> None:
        check_numeric(value=value, name="Drift")

        self.__drift = float(value)

    @property
    def volatility(self) -> float:
        r"""Volatility coefficient, or :math:`\sigma`."""
        return self.__volatility

    @volatility.setter
    def volatility(self, value: float) -> None:
        check_nonnegative_number(value=value, name="Volatility")

        self.__volatility = float(value)

    def _driver(self) -> FractionalGaussianNoise:
        """Return the fractional Gaussian noise which drives the process."""
        self.__noise = follow(self.__noise, self)
        return self.__noise

    def _exponentiate(
        self, increments: npt.NDArray[np.float64], initial: float
    ) -> npt.NDArray[np.float64]:
        """Turn increments of the fractional Brownian motion into the process."""
        motion = np.zeros((*increments.shape[:-1], increments.shape[-1] + 1))
        np.cumsum(increments, axis=-1, out=motion[..., 1:])

        times = self.times
        line = self.drift * times - self.volatility**2 / 2.0 * times ** (2.0 * self.hurst)
        realizations: npt.NDArray[np.float64] = initial * np.exp(line + self.volatility * motion)
        return realizations

    @override
    def sample(self, n: int) -> npt.NDArray[np.float64]:
        return self.sample_with_initial(n, 1.0)

    def sample_with_initial(self, n: int, initial: float = 1.0) -> npt.NDArray[np.float64]:
        """Generate a realization starting from an initial value.

        :param n: the number of increments to generate
        :param initial: the initial value of the process :math:`S_0`
        :returns: the realization
        """
        check_positive_integer(n=n)
        check_positive_number(value=initial, name="Initial")
        self.set_times(n)

        return self._exponentiate(self._driver().sample(n), initial)

    @override
    def sample_at(self, times: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        raise NotImplementedError(
            f"{type(self).__name__} cannot be sampled at specified times. "
            "Use the sample method with a number of increments."
        )

    def _sample_paths_from(self, n: int, paths: int, initial: float) -> npt.NDArray[np.float64]:
        """Generate several realizations from an initial value."""
        check_positive_integer(n=n)
        check_positive_integer(n=paths, name="Number of paths")
        check_positive_number(value=initial, name="Initial")
        self.set_times(n)

        return self._exponentiate(self._driver()._sample_paths(n, paths), initial)

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
