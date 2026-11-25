"""Kimura process, or Wright-Fisher diffusion."""

import math
from types import SimpleNamespace
from typing import Any

from numpy.random import Generator

from .._compat import override
from ..utils.validation import check_nonnegative_number, check_numeric
from .euler_maruyama_process import EulerMaruyamaProcess


class KimuraProcess(EulerMaruyamaProcess):
    r"""Kimura process, or Wright-Fisher diffusion.

    The frequency of an allele in a population under selection and random
    drift, :math:`X_t`, which satisfies the following stochastic differential
    equation with Wiener process :math:`W_t`:

    .. math::

        dX_t = s X_t (1 - X_t) dt + \sigma \sqrt{X_t (1 - X_t)} dW_t

    The frequencies 0 and 1 are absorbing: the allele is lost or fixed.

    Realizations are generated using the Euler-Maruyama method. Values are kept in :math:`[0,1]`.
    :py:meth:`sample` starts from 0.5.

    :param selection: the selection coefficient, or :math:`s` above
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
        selection: float = 0.0,
        vol: float = 1.0,
        t: float = 1.0,
        rng: Generator | None = None,
    ) -> None:
        super().__init__(t=t, rng=rng)

        self.selection = selection
        self.vol = vol

    def __str__(self) -> str:
        return f"Kimura process with selection={self.selection}, vol={self.vol} on [0, {self.t}]"

    def __repr__(self) -> str:
        return f"KimuraProcess(selection={self.selection}, vol={self.vol}, t={self.t})"

    @property
    def selection(self) -> float:
        r"""Selection coefficient, or :math:`s`."""
        return self.__selection

    @selection.setter
    def selection(self, value: float) -> None:
        check_numeric(value=value, name="Selection")

        self.__selection = float(value)

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
        spread = x * (1.0 - x)
        return self.selection * spread, self.vol * functions.sqrt(spread)

    @override
    def _confine(self, x: Any, functions: SimpleNamespace) -> Any:
        return functions.clip(x, 0.0, 1.0)

    @override
    def _advance(self, gns: list[float], x: float, delta_t: float) -> list[float]:
        # The method of the base class, written out for numbers: the loop
        # then makes no call to a method and reads no attribute. It computes
        # the same values, in the same order of operations.
        selection = self.selection
        vol = self.vol
        sqrt = math.sqrt

        values = [x]
        for gn in gns:
            spread = x * (1.0 - x)
            x = x + selection * spread * delta_t + vol * sqrt(spread) * gn
            x = min(max(x, 0.0), 1.0)
            values.append(x)

        return values
