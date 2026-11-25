"""Threshold GARCH process, or GJR-GARCH(p, q)."""

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Any

from numpy.random import Generator

from .._compat import override
from ..utils.validation import check_coefficients
from .conditional_variance_process import ConditionalVarianceProcess


class TGARCHProcess(ConditionalVarianceProcess):
    r"""Threshold GARCH process, or GJR-GARCH(p, q).

    .. image:: /_static/tgarch_process.png
        :scale: 50%

    A GARCH process in which a negative value raises the variance more than a
    positive value of the same size (Glosten, Jagannathan and Runkle, 1993):

    .. math::

        \sigma_t^2 = \omega + \sum_{i=1}^p \left( \alpha_i
        + \gamma_i 1_{\{X_{t-i} < 0\}} \right) X_{t-i}^2
        + \sum_{j=1}^q \beta_j \sigma_{t-j}^2, \quad X_t = \sigma_t z_t

    with :math:`z_t \sim N(0, 1)`. This reproduces the leverage effect of
    stock returns, where falls are followed by more volatility than rises.
    The long-run variance is
    :math:`\omega / (1 - \sum \alpha_i - \sum \gamma_i / 2 - \sum \beta_j)`.

    :param omega: the constant term of the variance, or :math:`\omega`
        above, positive
    :param alpha: the coefficients :math:`\alpha_1, \dots, \alpha_p` of the
        squared previous values, or a single number; nonnegative
    :param gamma: the additional coefficients
        :math:`\gamma_1, \dots, \gamma_p` of the squared negative previous
        values, or a single number; as many as ``alpha``
    :param beta: the coefficients :math:`\beta_1, \dots, \beta_q` of the
        previous variances, or a single number; nonnegative
    :param rng: a custom random number generator
    """

    def __init__(
        self,
        *,
        omega: float = 0.1,
        alpha: float | Sequence[float] = 0.05,
        gamma: float | Sequence[float] = 0.1,
        beta: float | Sequence[float] = 0.8,
        rng: Generator | None = None,
    ) -> None:
        super().__init__(omega=omega, rng=rng)

        self.alpha = alpha
        self.gamma = gamma
        self.beta = beta
        self._finish_init()

    def __str__(self) -> str:
        return (
            f"TGARCH({len(self.alpha)}, {len(self.beta)}) process with omega={self.omega}, "
            f"alpha={self.alpha}, gamma={self.gamma} and beta={self.beta}."
        )

    def __repr__(self) -> str:
        return (
            f"TGARCHProcess(omega={self.omega}, alpha={self.alpha}, gamma={self.gamma}, "
            f"beta={self.beta})"
        )

    @property
    def alpha(self) -> tuple[float, ...]:
        r"""Coefficients :math:`\\alpha_i` of the squared previous values."""
        coefficients: tuple[float, ...] = self._get("alpha")
        return coefficients

    @alpha.setter
    def alpha(self, value: float | Sequence[float]) -> None:
        self._set("alpha", check_coefficients(value=value, name="Alpha"))

    @property
    def gamma(self) -> tuple[float, ...]:
        r"""Additional coefficients :math:`\\gamma_i` of the squared negative values."""
        coefficients: tuple[float, ...] = self._get("gamma")
        return coefficients

    @gamma.setter
    def gamma(self, value: float | Sequence[float]) -> None:
        self._set("gamma", check_coefficients(value=value, name="Gamma"))

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
        if len(self.alpha) != len(self.gamma):
            raise ValueError(
                "Alpha and gamma must hold the same number of coefficients; "
                f"got {len(self.alpha)} and {len(self.gamma)}."
            )
        if min(self.alpha + self.beta, default=0.0) < 0:
            raise ValueError(
                f"Alpha and beta must hold nonnegative numbers; got {self.alpha} and {self.beta}."
            )
        if any(a + g < 0 for a, g in zip(self.alpha, self.gamma, strict=True)):
            raise ValueError(
                "The sum of a coefficient of alpha and its coefficient of gamma must not "
                f"be negative; got {self.alpha} and {self.gamma}."
            )
        if sum(self.alpha) + sum(self.gamma) / 2.0 + sum(self.beta) >= 1:
            raise ValueError(
                "The coefficients of alpha and beta and half of those of gamma must add up "
                f"to less than 1; got {self.alpha}, {self.gamma} and {self.beta}."
            )

    @property
    @override
    def _lags(self) -> int:
        return max(len(self.alpha), len(self.beta))

    @override
    def _stationary_state(self) -> float:
        persistence = sum(self.alpha) + sum(self.gamma) / 2.0 + sum(self.beta)
        return self.omega / (1.0 - persistence)

    @override
    def _next_state(
        self,
        values: Sequence[Any],
        innovations: Sequence[Any],
        states: Sequence[Any],
        functions: SimpleNamespace,
    ) -> Any:
        variance = self.omega
        for alpha, gamma, value in zip(self.alpha, self.gamma, values, strict=False):
            variance = variance + (alpha + gamma * (value < 0)) * value * value
        for coefficient, state in zip(self.beta, states, strict=False):
            variance = variance + coefficient * state
        return variance
