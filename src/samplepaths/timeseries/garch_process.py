"""Generalized ARCH process, or GARCH(p, q)."""

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Any

from numpy.random import Generator

from .._compat import override
from ..utils.validation import check_coefficients
from .conditional_variance_process import ConditionalVarianceProcess


class GARCHProcess(ConditionalVarianceProcess):
    r"""Generalized ARCH process, or GARCH(p, q).

    .. image:: /_static/garch_process.png
        :scale: 50%

    A series with a zero mean whose variance depends on the squares of the
    :math:`p` previous values and on the :math:`q` previous variances
    (Bollerslev, 1986):

    .. math::

        X_t = \sigma_t z_t, \quad
        \sigma_t^2 = \omega + \sum_{i=1}^p \alpha_i X_{t-i}^2
        + \sum_{j=1}^q \beta_j \sigma_{t-j}^2, \quad z_t \sim N(0, 1)

    The previous variances make agitated periods last. The long-run variance
    is :math:`\omega / (1 - \sum \alpha_i - \sum \beta_j)`.

    :param omega: the constant term of the variance, or :math:`\omega`
        above, positive
    :param alpha: the coefficients :math:`\alpha_1, \dots, \alpha_p` of the
        squared previous values, or a single number; nonnegative
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
        beta: float | Sequence[float] = 0.8,
        rng: Generator | None = None,
    ) -> None:
        super().__init__(omega=omega, rng=rng)

        self.alpha = alpha
        self.beta = beta
        self._finish_init()

    def __str__(self) -> str:
        return (
            f"GARCH({len(self.alpha)}, {len(self.beta)}) process with omega={self.omega}, "
            f"alpha={self.alpha} and beta={self.beta}."
        )

    def __repr__(self) -> str:
        return f"GARCHProcess(omega={self.omega}, alpha={self.alpha}, beta={self.beta})"

    @property
    def alpha(self) -> tuple[float, ...]:
        r"""Coefficients :math:`\\alpha_i` of the squared previous values."""
        coefficients: tuple[float, ...] = self._get("alpha")
        return coefficients

    @alpha.setter
    def alpha(self, value: float | Sequence[float]) -> None:
        self._set("alpha", check_coefficients(value=value, name="Alpha"))

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
        return self.omega / (1.0 - sum(self.alpha) - sum(self.beta))

    @override
    def _next_state(
        self,
        values: Sequence[Any],
        innovations: Sequence[Any],
        states: Sequence[Any],
        functions: SimpleNamespace,
    ) -> Any:
        variance = self.omega
        for coefficient, value in zip(self.alpha, values, strict=False):
            variance = variance + coefficient * value * value
        for coefficient, state in zip(self.beta, states, strict=False):
            variance = variance + coefficient * state
        return variance
