"""Fractional Vasicek process."""

from types import SimpleNamespace
from typing import Any

import numpy as np
import numpy.typing as npt
from numpy.random import Generator

from .._compat import override
from .._recurrence import solve_linear_recurrence
from ..noise.gaussian_noise import GaussianNoise
from ..utils.validation import check_nonnegative_number, check_numeric, check_positive_number
from .fractional_euler_process import FractionalEulerProcess


class FractionalVasicekProcess(FractionalEulerProcess):
    r"""Fractional Vasicek process.

    .. image:: /_static/fractional_vasicek_process.png
        :scale: 50%

    A mean-reverting process :math:`X_t` driven by a fractional Brownian
    motion :math:`B^H_t`:

    .. math::

        dX_t = \theta (\mu - X_t) dt + \sigma dB^H_t

    With a Hurst parameter above 1/2 it models interest rates and
    volatilities with a long memory; below 1/2, rough volatility.

    Realizations are generated using the Euler method with exact increments
    of the fractional Brownian motion. :py:meth:`sample` starts from 1.

    :param hurst: the Hurst parameter :math:`H` in :math:`(0,1)`
    :param speed: the speed of reversion, or :math:`\theta` above
    :param mean: the mean of the process, or :math:`\mu` above
    :param vol: volatility coefficient of the process, or :math:`\sigma`
        above
    :param t: the right hand endpoint of the time interval :math:`[0,t]`
        for the process
    :param rng: a custom random number generator
    """

    def __init__(
        self,
        *,
        hurst: float = 0.7,
        speed: float = 1.0,
        mean: float = 1.0,
        vol: float = 1.0,
        t: float = 1.0,
        rng: Generator | None = None,
    ) -> None:
        super().__init__(hurst=hurst, t=t, rng=rng)

        self.speed = speed
        self.mean = mean
        self.vol = vol

    def __str__(self) -> str:
        return (
            f"Fractional Vasicek process with Hurst {self.hurst}, speed={self.speed}, "
            f"mean={self.mean}, vol={self.vol} on [0, {self.t}]"
        )

    def __repr__(self) -> str:
        return (
            f"FractionalVasicekProcess(hurst={self.hurst}, speed={self.speed}, "
            f"mean={self.mean}, vol={self.vol}, t={self.t})"
        )

    @property
    def speed(self) -> float:
        r"""Speed of reversion, or :math:`\theta`."""
        return self.__speed

    @speed.setter
    def speed(self, value: float) -> None:
        check_positive_number(value=value, name="Speed")

        self.__speed = float(value)

    @property
    def mean(self) -> float:
        r"""Mean, or :math:`\mu`."""
        return self.__mean

    @mean.setter
    def mean(self, value: float) -> None:
        check_numeric(value=value, name="Mean")

        self.__mean = float(value)

    @property
    def vol(self) -> float:
        r"""Volatility coefficient, or :math:`\sigma`."""
        return self.__vol

    @vol.setter
    def vol(self, value: float) -> None:
        check_nonnegative_number(value=value, name="Volatility")

        self.__vol = float(value)

    @override
    def _coefficients(self, x: Any, functions: SimpleNamespace) -> tuple[Any, Any]:
        return self.speed * (self.mean - x), self.vol

    @override
    def _confine(self, x: Any, functions: SimpleNamespace) -> Any:
        return x

    def _solve(self, noise: npt.NDArray[np.float64], initial: float) -> npt.NDArray[np.float64]:
        """Apply the Euler method to realizations of the noise, one per row.

        The equation is linear, so its steps form a linear recurrence,
        ``X[k + 1] = decay X[k] + input[k]``, which is solved for all steps
        with array operations.
        """
        check_numeric(value=initial, name="Initial")
        delta_t = self.t / noise.shape[-1]
        inputs = self.speed * self.mean * delta_t + self.vol * noise
        return solve_linear_recurrence(inputs, 1.0 - self.speed * delta_t, float(initial))

    @override
    def _useful_workers(self, n: int, paths: int) -> int:
        # No step is taken in Python, so every thread is useful.
        return GaussianNoise._useful_workers(self, n, paths)

    @override
    def _sample_from(self, n: int, initial: float) -> npt.NDArray[np.float64]:
        return self._solve(self._sample_gaussian_noise(n), initial)

    @override
    def _sample_paths_from(self, n: int, paths: int, initial: float) -> npt.NDArray[np.float64]:
        return self._solve(self._sample_gaussian_noise_paths(n, paths), initial)
