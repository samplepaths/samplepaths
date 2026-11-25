"""Gompertz process."""

from types import SimpleNamespace
from typing import Any

from numpy.random import Generator

from .._compat import override
from ..utils.validation import check_nonnegative_number, check_numeric
from .euler_maruyama_process import EulerMaruyamaProcess

#: Smallest value of the process, which keeps its logarithm finite.
_SMALLEST = 1e-12


class GompertzProcess(EulerMaruyamaProcess):
    r"""Gompertz process.

    The size of a population whose growth slows down with its logarithm,
    :math:`X_t`, which satisfies the following stochastic differential
    equation with Wiener process :math:`W_t`:

    .. math::

        dX_t = (a - b \ln X_t) X_t dt + \sigma X_t dW_t

    Its logarithm is an Ornstein-Uhlenbeck process, and the process stays
    positive.

    Realizations are generated using the Euler-Maruyama method. Values are kept positive.
    :py:meth:`sample` starts from 1.

    :param growth: the growth rate, or :math:`a` above
    :param decay: the rate at which the growth slows down, or :math:`b`
        above
    :param vol: volatility coefficient of the process, or :math:`\sigma`
        above
    :param t: the right hand endpoint of the time interval :math:`[0,t]`
        for the process
    :param rng: a custom random number generator
    """

    def __init__(
        self,
        *,
        growth: float = 1.0,
        decay: float = 1.0,
        vol: float = 0.1,
        t: float = 1.0,
        rng: Generator | None = None,
    ) -> None:
        super().__init__(t=t, rng=rng)

        self.growth = growth
        self.decay = decay
        self.vol = vol

    def __str__(self) -> str:
        return (
            f"Gompertz process with growth={self.growth}, "
            f"decay={self.decay}, vol={self.vol} on [0, {self.t}]"
        )

    def __repr__(self) -> str:
        return (
            f"GompertzProcess(growth={self.growth}, decay={self.decay}, vol={self.vol}, t={self.t})"
        )

    @property
    def growth(self) -> float:
        r"""Growth rate, or :math:`a`."""
        return self.__growth

    @growth.setter
    def growth(self, value: float) -> None:
        check_numeric(value=value, name="Growth rate")

        self.__growth = float(value)

    @property
    def decay(self) -> float:
        r"""Rate at which the growth slows down, or :math:`b`."""
        return self.__decay

    @decay.setter
    def decay(self, value: float) -> None:
        check_nonnegative_number(value=value, name="Decay")

        self.__decay = float(value)

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
        return (self.growth - self.decay * functions.log(x)) * x, self.vol * x

    @override
    def _confine(self, x: Any, functions: SimpleNamespace) -> Any:
        return functions.clip(x, _SMALLEST, float("inf"))
