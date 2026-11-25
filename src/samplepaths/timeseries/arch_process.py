"""Autoregressive conditional heteroskedasticity process, or ARCH(p)."""

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Any

from numpy.random import Generator

from .._compat import override
from ..utils.validation import check_coefficients
from .conditional_variance_process import ConditionalVarianceProcess


class ARCHProcess(ConditionalVarianceProcess):
    r"""Autoregressive conditional heteroskedasticity process, or ARCH(p).

    .. image:: /_static/arch_process.png
        :scale: 50%

    A series with a zero mean whose variance depends on the squares of the
    :math:`p` previous values (Engle, 1982):

    .. math::

        X_t = \sigma_t z_t, \quad
        \sigma_t^2 = \omega + \sum_{i=1}^p \alpha_i X_{t-i}^2,
        \quad z_t \sim N(0, 1)

    A large value is followed by a large variance, so large values cluster.
    The long-run variance is :math:`\omega / (1 - \sum \alpha_i)`.

    :param omega: the constant term of the variance, or :math:`\omega`
        above, positive
    :param alpha: the coefficients :math:`\alpha_1, \dots, \alpha_p` of the
        squared previous values, or a single number; nonnegative, with a sum
        below 1
    :param rng: a custom random number generator
    """

    def __init__(
        self,
        *,
        omega: float = 0.5,
        alpha: float | Sequence[float] = 0.5,
        rng: Generator | None = None,
    ) -> None:
        super().__init__(omega=omega, rng=rng)

        self.alpha = alpha
        self._finish_init()

    def __str__(self) -> str:
        return f"ARCH({len(self.alpha)}) process with omega={self.omega} and alpha={self.alpha}."

    def __repr__(self) -> str:
        return f"ARCHProcess(omega={self.omega}, alpha={self.alpha})"

    @property
    def alpha(self) -> tuple[float, ...]:
        r"""Coefficients :math:`\\alpha_i` of the squared previous values."""
        coefficients: tuple[float, ...] = self._get("alpha")
        return coefficients

    @alpha.setter
    def alpha(self, value: float | Sequence[float]) -> None:
        self._set("alpha", check_coefficients(value=value, name="Alpha"))

    @override
    def _check_parameters(self) -> None:
        if self.omega <= 0:
            raise ValueError(f"Omega must be positive; got {self.omega}.")
        if min(self.alpha, default=0.0) < 0:
            raise ValueError(f"Alpha must hold nonnegative numbers; got {self.alpha}.")
        if sum(self.alpha) >= 1:
            raise ValueError(
                "The coefficients of alpha must add up to less than 1 for the variance "
                f"to settle; got {self.alpha}."
            )

    @property
    @override
    def _lags(self) -> int:
        return len(self.alpha)

    @override
    def _stationary_state(self) -> float:
        return self.omega / (1.0 - sum(self.alpha))

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
        return variance
