"""Asymmetric GARCH process, or AGARCH(p, q)."""

import math
from collections.abc import Sequence
from types import SimpleNamespace
from typing import Any

from numpy.random import Generator

from .._compat import override
from ..utils.validation import check_coefficients, check_numeric
from .conditional_variance_process import _SMALLEST_VARIANCE, ConditionalVarianceProcess


class AGARCHProcess(ConditionalVarianceProcess):
    r"""Asymmetric GARCH process, or AGARCH(p, q).

    .. image:: /_static/agarch_process.png
        :scale: 50%

    A GARCH process in which the variance is lowest after a previous value
    equal to a shift, instead of zero (Engle, 1990):

    .. math::

        \sigma_t^2 = \omega + \sum_{i=1}^p \alpha_i (X_{t-i} - c)^2
        + \sum_{j=1}^q \beta_j \sigma_{t-j}^2, \quad X_t = \sigma_t z_t

    with :math:`z_t \sim N(0, 1)`. With a positive shift :math:`c`, a
    negative value raises the variance more than a positive value of the
    same size. The long-run variance is
    :math:`(\omega + c^2 \sum \alpha_i) / (1 - \sum \alpha_i - \sum \beta_j)`.

    :param omega: the constant term of the variance, or :math:`\omega`
        above, positive
    :param alpha: the coefficients :math:`\alpha_1, \dots, \alpha_p` of the
        squared shifted previous values, or a single number; nonnegative
    :param shift: the shift, or :math:`c` above
    :param beta: the coefficients :math:`\beta_1, \dots, \beta_q` of the
        previous variances, or a single number; nonnegative. The
        coefficients of ``alpha`` and ``beta`` add up to less than 1.
    :param rng: a custom random number generator
    """

    def __init__(
        self,
        *,
        omega: float = 0.1,
        alpha: float | Sequence[float] = 0.1,
        shift: float = 0.5,
        beta: float | Sequence[float] = 0.8,
        rng: Generator | None = None,
    ) -> None:
        super().__init__(omega=omega, rng=rng)

        self.alpha = alpha
        self.shift = shift
        self.beta = beta
        self._finish_init()

    def __str__(self) -> str:
        return (
            f"AGARCH({len(self.alpha)}, {len(self.beta)}) process with omega={self.omega}, "
            f"alpha={self.alpha}, shift={self.shift} and beta={self.beta}."
        )

    def __repr__(self) -> str:
        return (
            f"AGARCHProcess(omega={self.omega}, alpha={self.alpha}, shift={self.shift}, "
            f"beta={self.beta})"
        )

    @property
    def alpha(self) -> tuple[float, ...]:
        r"""Coefficients :math:`\\alpha_i` of the squared shifted previous values."""
        coefficients: tuple[float, ...] = self._get("alpha")
        return coefficients

    @alpha.setter
    def alpha(self, value: float | Sequence[float]) -> None:
        self._set("alpha", check_coefficients(value=value, name="Alpha"))

    @property
    def shift(self) -> float:
        r"""Shift, or :math:`c`."""
        shift: float = self._get("shift")
        return shift

    @shift.setter
    def shift(self, value: float) -> None:
        check_numeric(value=value, name="Shift")

        self._set("shift", float(value))

    @property
    def beta(self) -> tuple[float, ...]:
        r"""Coefficients :math:`\\beta_j` of the previous variances."""
        coefficients: tuple[float, ...] = self._get("beta")
        return coefficients

    @beta.setter
    def beta(self, value: float | Sequence[float]) -> None:
        self._set("beta", check_coefficients(value=value, name="Beta"))

    @override
    def _check_parameters(self) -> None:
        if self.omega <= 0:
            raise ValueError(f"Omega must be positive; got {self.omega}.")
        if min(self.alpha + self.beta, default=0.0) < 0:
            raise ValueError(
                f"Alpha and beta must hold nonnegative numbers; got {self.alpha} and {self.beta}."
            )
        if sum(self.alpha) + sum(self.beta) >= 1:
            raise ValueError(
                "The coefficients of alpha and beta must add up to less than 1 for the "
                f"variance to settle; got {self.alpha} and {self.beta}."
            )

    @property
    @override
    def _lags(self) -> int:
        return max(len(self.alpha), len(self.beta))

    @override
    def _stationary_state(self) -> float:
        alpha = sum(self.alpha)
        return (self.omega + self.shift**2 * alpha) / (1.0 - alpha - sum(self.beta))

    @override
    def _next_state(
        self,
        values: Sequence[Any],
        innovations: Sequence[Any],
        states: Sequence[Any],
        functions: SimpleNamespace,
    ) -> Any:
        shift = self.shift
        variance = self.omega
        for coefficient, value in zip(self.alpha, values, strict=False):
            variance = variance + coefficient * (value - shift) * (value - shift)
        for coefficient, state in zip(self.beta, states, strict=False):
            variance = variance + coefficient * state
        return variance

    @override
    def _advance_floats(
        self,
        presample: list[float],
        innovations: list[float],
    ) -> tuple[list[float], list[float]]:
        if len(self.alpha) != 1 or len(self.beta) != 1:
            return super()._advance_floats(presample, innovations)

        # The method of the base class, written out for one previous value:
        # the loop then makes no call to a method and reads no attribute. It
        # computes the same values, in the same order of operations.
        omega = self.omega
        alpha = self.alpha[0]
        shift = self.shift
        beta = self.beta[0]
        sqrt = math.sqrt
        state = self._stationary_state()
        value = sqrt(state) * presample[0]

        values = []
        deviations = []
        for z in innovations:
            state = omega + alpha * (value - shift) * (value - shift) + beta * state
            deviation = sqrt(max(state, _SMALLEST_VARIANCE))
            value = deviation * z
            values.append(value)
            deviations.append(deviation)

        return values, deviations
