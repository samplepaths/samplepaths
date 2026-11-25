"""Verhulst process, or stochastic logistic growth."""

import math
from types import SimpleNamespace
from typing import Any

from numpy.random import Generator

from .._compat import override
from ..utils.validation import check_nonnegative_number, check_numeric, check_positive_number
from .euler_maruyama_process import EulerMaruyamaProcess


class VerhulstProcess(EulerMaruyamaProcess):
    r"""Verhulst process, or stochastic logistic growth.

    The size of a population which grows towards a carrying capacity,
    :math:`X_t`, which satisfies the following stochastic differential
    equation with Wiener process :math:`W_t`:

    .. math::

        dX_t = r X_t \left(1 - \frac{X_t}{K}\right) dt + \sigma X_t dW_t

    Realizations are generated using the Euler-Maruyama method. Values are kept nonnegative.
    :py:meth:`sample` starts from 1.

    :param rate: the growth rate, or :math:`r` above
    :param capacity: the carrying capacity, or :math:`K` above
    :param vol: volatility coefficient of the process, or :math:`\sigma`
        above
    :param t: the right hand endpoint of the time interval :math:`[0,t]`
        for the process
    :param rng: a custom random number generator
    """

    def __init__(
        self,
        *,
        rate: float = 1.0,
        capacity: float = 10.0,
        vol: float = 0.1,
        t: float = 1.0,
        rng: Generator | None = None,
    ) -> None:
        super().__init__(t=t, rng=rng)

        self.rate = rate
        self.capacity = capacity
        self.vol = vol

    def __str__(self) -> str:
        return (
            f"Verhulst process with rate={self.rate}, "
            f"capacity={self.capacity}, vol={self.vol} on [0, {self.t}]"
        )

    def __repr__(self) -> str:
        return (
            f"VerhulstProcess(rate={self.rate}, "
            f"capacity={self.capacity}, vol={self.vol}, t={self.t})"
        )

    @property
    def rate(self) -> float:
        r"""Growth rate, or :math:`r`."""
        return self.__rate

    @rate.setter
    def rate(self, value: float) -> None:
        check_numeric(value=value, name="Growth rate")

        self.__rate = float(value)

    @property
    def capacity(self) -> float:
        r"""Carrying capacity, or :math:`K`."""
        return self.__capacity

    @capacity.setter
    def capacity(self, value: float) -> None:
        check_positive_number(value=value, name="Capacity")

        self.__capacity = float(value)

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
        return self.rate * x * (1.0 - x / self.capacity), self.vol * x

    @override
    def _confine(self, x: Any, functions: SimpleNamespace) -> Any:
        return functions.clip(x, 0.0, float("inf"))

    @override
    def _advance(self, gns: list[float], x: float, delta_t: float) -> list[float]:
        # The method of the base class, written out for numbers: the loop
        # then makes no call to a method and reads no attribute. It computes
        # the same values, in the same order of operations.
        rate = self.rate
        capacity = self.capacity
        vol = self.vol
        infinity = math.inf

        values = [x]
        for gn in gns:
            x = x + rate * x * (1.0 - x / capacity) * delta_t + vol * x * gn
            x = min(max(x, 0.0), infinity)
            values.append(x)

        return values
