"""Fractional Cox-Ingersoll-Ross process."""

import math
from types import SimpleNamespace
from typing import Any

from numpy.random import Generator

from .._compat import override
from ..utils.validation import check_nonnegative_number, check_positive_number
from .fractional_euler_process import FractionalEulerProcess


class FractionalCoxIngersollRossProcess(FractionalEulerProcess):
    r"""Fractional Cox-Ingersoll-Ross process.

    .. image:: /_static/fractional_cox_ingersoll_ross_process.png
        :scale: 50%

    A mean-reverting process :math:`X_t` which stays nonnegative, driven by
    a fractional Brownian motion :math:`B^H_t`:

    .. math::

        dX_t = \theta (\mu - X_t) dt + \sigma \sqrt{X_t} dB^H_t

    Realizations are generated using the Euler method with exact increments
    of the fractional Brownian motion. A value which a step would make
    negative is set to zero. :py:meth:`sample` starts from 1.

    :param hurst: the Hurst parameter :math:`H` in :math:`(0,1)`
    :param speed: the speed of reversion, or :math:`\theta` above
    :param mean: the mean of the process, or :math:`\mu` above, nonnegative
    :param vol: volatility coefficient of the process, or :math:`\sigma`
        above
    :param t: the right hand endpoint of the time interval :math:`[0,t]`
        for the process
    :param rng: a custom random number generator
    """

    _initial = 1.0

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
            f"Fractional Cox-Ingersoll-Ross process with Hurst {self.hurst}, speed={self.speed}, "
            f"mean={self.mean}, vol={self.vol} on [0, {self.t}]"
        )

    def __repr__(self) -> str:
        return (
            f"FractionalCoxIngersollRossProcess(hurst={self.hurst}, speed={self.speed}, "
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
        check_nonnegative_number(value=value, name="Mean")

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
        return self.speed * (self.mean - x), self.vol * functions.sqrt(x)

    @override
    def _confine(self, x: Any, functions: SimpleNamespace) -> Any:
        return functions.maximum(x, 0.0)

    @override
    def _advance(self, gns: list[float], x: float, delta_t: float) -> list[float]:
        # The method of the base class, written out for numbers: the loop
        # then makes no call to a method and reads no attribute. It computes
        # the same values, in the same order of operations.
        speed = self.speed
        mean = self.mean
        vol = self.vol
        sqrt = math.sqrt

        values = [x]
        for gn in gns:
            x = x + speed * (mean - x) * delta_t + vol * sqrt(x) * gn
            x = max(x, 0.0)
            values.append(x)

        return values
