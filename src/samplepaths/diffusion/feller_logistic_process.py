"""Feller logistic process."""

from types import SimpleNamespace
from typing import Any

from numpy.random import Generator

from .._compat import override
from ..utils.validation import check_nonnegative_number
from .euler_maruyama_process import EulerMaruyamaProcess


class FellerLogisticProcess(EulerMaruyamaProcess):
    r"""Feller logistic process.

    The size of a population with logistic growth and demographic noise,
    :math:`X_t`, which satisfies the following stochastic differential
    equation with Wiener process :math:`W_t`:

    .. math::

        dX_t = \kappa (\theta - X_t) X_t dt + \sigma \sqrt{X_t} dW_t

    The value zero is absorbing: the population is extinct.

    Realizations are generated using the Euler-Maruyama method. Values are kept nonnegative.
    :py:meth:`sample` starts from 1.

    :param speed: the speed of the logistic growth, or :math:`\kappa` above
    :param level: the level the population grows towards, or
        :math:`\theta` above
    :param vol: volatility coefficient of the process, or :math:`\sigma`
        above
    :param t: the right hand endpoint of the time interval :math:`[0,t]`
        for the process
    :param rng: a custom random number generator
    """

    def __init__(
        self,
        *,
        speed: float = 1.0,
        level: float = 1.0,
        vol: float = 0.1,
        t: float = 1.0,
        rng: Generator | None = None,
    ) -> None:
        super().__init__(t=t, rng=rng)

        self.speed = speed
        self.level = level
        self.vol = vol

    def __str__(self) -> str:
        return (
            f"Feller logistic process with speed={self.speed}, "
            f"level={self.level}, vol={self.vol} on [0, {self.t}]"
        )

    def __repr__(self) -> str:
        return (
            f"FellerLogisticProcess(speed={self.speed}, "
            f"level={self.level}, vol={self.vol}, t={self.t})"
        )

    @property
    def speed(self) -> float:
        r"""Speed of the logistic growth, or :math:`\kappa`."""
        return self.__speed

    @speed.setter
    def speed(self, value: float) -> None:
        check_nonnegative_number(value=value, name="Speed")

        self.__speed = float(value)

    @property
    def level(self) -> float:
        r"""Level the population grows towards, or :math:`\theta`."""
        return self.__level

    @level.setter
    def level(self, value: float) -> None:
        check_nonnegative_number(value=value, name="Level")

        self.__level = float(value)

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
        return self.speed * (self.level - x) * x, self.vol * functions.sqrt(x)

    @override
    def _confine(self, x: Any, functions: SimpleNamespace) -> Any:
        return functions.clip(x, 0.0, float("inf"))
