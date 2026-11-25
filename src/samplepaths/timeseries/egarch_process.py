"""Exponential GARCH process, or EGARCH(p, q)."""

import math
from collections.abc import Sequence
from types import SimpleNamespace
from typing import Any

from numpy.random import Generator

from .._compat import override
from ..utils.validation import check_coefficients
from .conditional_variance_process import ConditionalVarianceProcess

#: Mean of the absolute value of a standard normal variate.
_MEAN_ABSOLUTE_NORMAL = math.sqrt(2.0 / math.pi)

#: Largest logarithm of a variance, in absolute value.
_LARGEST_LOG = 700.0


class EGARCHProcess(ConditionalVarianceProcess):
    r"""Exponential GARCH process, or EGARCH(p, q).

    .. image:: /_static/egarch_process.png
        :scale: 50%

    A GARCH process for the logarithm of the variance (Nelson, 1991):

    .. math::

        \ln \sigma_t^2 = \omega + \sum_{i=1}^p \left[ \alpha_i
        \left( |z_{t-i}| - E|z| \right) + \gamma_i z_{t-i} \right]
        + \sum_{j=1}^q \beta_j \ln \sigma_{t-j}^2, \quad X_t = \sigma_t z_t

    with :math:`z_t \sim N(0, 1)` and :math:`E|z| = \sqrt{2 / \pi}`. The
    variance is positive whatever the signs of the coefficients. The
    coefficients :math:`\alpha_i` set the effect of the size of a value, and
    the :math:`\gamma_i` the effect of its sign: a negative
    :math:`\gamma_i` makes negative values raise the variance more. The
    long-run level of :math:`\ln \sigma_t^2` is
    :math:`\omega / (1 - \sum \beta_j)`.

    :param omega: the constant term of the logarithm of the variance, or
        :math:`\omega` above
    :param alpha: the coefficients :math:`\alpha_1, \dots, \alpha_p` of the
        sizes of the previous values, or a single number
    :param gamma: the coefficients :math:`\gamma_1, \dots, \gamma_p` of the
        signs of the previous values, or a single number; as many as
        ``alpha``
    :param beta: the coefficients :math:`\beta_1, \dots, \beta_q` of the
        previous logarithms of the variance, or a single number, with a sum
        below 1 in absolute value
    :param rng: a custom random number generator
    """

    def __init__(
        self,
        *,
        omega: float = -0.1,
        alpha: float | Sequence[float] = 0.2,
        gamma: float | Sequence[float] = -0.1,
        beta: float | Sequence[float] = 0.9,
        rng: Generator | None = None,
    ) -> None:
        super().__init__(omega=omega, rng=rng)

        self.alpha = alpha
        self.gamma = gamma
        self.beta = beta
        self._finish_init()

    def __str__(self) -> str:
        return (
            f"EGARCH({len(self.alpha)}, {len(self.beta)}) process with omega={self.omega}, "
            f"alpha={self.alpha}, gamma={self.gamma} and beta={self.beta}."
        )

    def __repr__(self) -> str:
        return (
            f"EGARCHProcess(omega={self.omega}, alpha={self.alpha}, gamma={self.gamma}, "
            f"beta={self.beta})"
        )

    @property
    def alpha(self) -> tuple[float, ...]:
        r"""Coefficients :math:`\\alpha_i` of the sizes of the previous values."""
        coefficients: tuple[float, ...] = self._get("alpha")
        return coefficients

    @alpha.setter
    def alpha(self, value: float | Sequence[float]) -> None:
        self._set("alpha", check_coefficients(value=value, name="Alpha"))

    @property
    def gamma(self) -> tuple[float, ...]:
        r"""Coefficients :math:`\\gamma_i` of the signs of the previous values."""
        coefficients: tuple[float, ...] = self._get("gamma")
        return coefficients

    @gamma.setter
    def gamma(self, value: float | Sequence[float]) -> None:
        self._set("gamma", check_coefficients(value=value, name="Gamma"))

    @property
    def beta(self) -> tuple[float, ...]:
        r"""Coefficients :math:`\\beta_j` of the previous logarithms of the variance."""
        coefficients: tuple[float, ...] = self._get("beta")
        return coefficients

    @beta.setter
    def beta(self, value: float | Sequence[float]) -> None:
        self._set("beta", check_coefficients(value=value, name="Beta"))

    @override
    def _check_parameters(self) -> None:
        if len(self.alpha) != len(self.gamma):
            raise ValueError(
                "Alpha and gamma must hold the same number of coefficients; "
                f"got {len(self.alpha)} and {len(self.gamma)}."
            )
        if abs(sum(self.beta)) >= 1:
            raise ValueError(
                "The coefficients of beta must add up to less than 1 in absolute value "
                f"for the variance to settle; got {self.beta}."
            )

    @property
    @override
    def _lags(self) -> int:
        return max(len(self.alpha), len(self.beta))

    @override
    def _stationary_state(self) -> float:
        return self.omega / (1.0 - sum(self.beta))

    @override
    def _next_state(
        self,
        values: Sequence[Any],
        innovations: Sequence[Any],
        states: Sequence[Any],
        functions: SimpleNamespace,
    ) -> Any:
        log_variance = self.omega
        for alpha, gamma, z in zip(self.alpha, self.gamma, innovations, strict=False):
            log_variance = log_variance + alpha * (abs(z) - _MEAN_ABSOLUTE_NORMAL) + gamma * z
        for coefficient, state in zip(self.beta, states, strict=False):
            log_variance = log_variance + coefficient * state
        return log_variance

    @override
    def _variance(self, state: Any, functions: SimpleNamespace) -> Any:
        # Keep the exponential within the range of floats.
        return functions.exp(functions.clip(state, -_LARGEST_LOG, _LARGEST_LOG))
