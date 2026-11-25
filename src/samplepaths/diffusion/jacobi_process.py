"""Jacobi process."""

from types import SimpleNamespace
from typing import Any

from numpy.random import Generator

from .._compat import override
from ..utils.validation import check_nonnegative_number, check_positive_number
from .euler_maruyama_process import EulerMaruyamaProcess


class JacobiProcess(EulerMaruyamaProcess):
    r"""Jacobi process.

    A mean-reverting process on the interval :math:`[0,1]`, :math:`X_t`, which
    satisfies the following stochastic differential equation with Wiener
    process :math:`W_t`:

    .. math::

        dX_t = \theta (\mu - X_t) dt + \sigma \sqrt{X_t (1 - X_t)} dW_t

    Its volatility vanishes at both ends of the interval. It is used for
    quantities which are proportions, such as a correlation or an allele
    frequency.

    Realizations are generated using the Euler-Maruyama method. Values are kept in :math:`[0,1]`.
    :py:meth:`sample` starts from 0.5.

    :param speed: the speed of reversion, or :math:`\theta` above
    :param mean: the mean of the process, or :math:`\mu` above, in
        :math:`[0,1]`
    :param vol: volatility coefficient of the process, or :math:`\sigma`
        above
    :param t: the right hand endpoint of the time interval :math:`[0,t]`
        for the process
    :param rng: a custom random number generator
    """

    _initial = 0.5

    def __init__(
        self,
        *,
        speed: float = 1.0,
        mean: float = 0.5,
        vol: float = 1.0,
        t: float = 1.0,
        rng: Generator | None = None,
    ) -> None:
        super().__init__(t=t, rng=rng)

        self.speed = speed
        self.mean = mean
        self.vol = vol

    def __str__(self) -> str:
        return (
            f"Jacobi process with speed={self.speed}, "
            f"mean={self.mean}, vol={self.vol} on [0, {self.t}]"
        )

    def __repr__(self) -> str:
        return f"JacobiProcess(speed={self.speed}, mean={self.mean}, vol={self.vol}, t={self.t})"

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
        if value > 1:
            raise ValueError(f"Mean value must be at most 1; got {value}.")

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
        return self.speed * (self.mean - x), self.vol * functions.sqrt(x * (1.0 - x))

    @override
    def _confine(self, x: Any, functions: SimpleNamespace) -> Any:
        return functions.clip(x, 0.0, 1.0)
