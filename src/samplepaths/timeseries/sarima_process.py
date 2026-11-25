"""Seasonal autoregressive integrated moving average process."""

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
from numpy.random import Generator

from .._compat import override
from ..utils.validation import (
    check_coefficients,
    check_nonnegative_integer,
    check_positive_integer,
)
from ._filters import differencing_polynomial, lag_polynomial
from .linear_process import LinearProcess


class SARIMAProcess(LinearProcess):
    r"""Seasonal ARIMA process, or SARIMA(p, d, q)(P, D, Q) with period s.

    .. image:: /_static/sarima_process.png
        :scale: 50%

    An ARIMA process with a second set of terms at multiples of a season
    length :math:`s`. With the lag operator :math:`B`, where
    :math:`B X_t = X_{t-1}`, and Gaussian white noise :math:`\varepsilon_t`
    with standard deviation :math:`\sigma`:

    .. math::

        \Phi(B^s) \phi(B) (1 - B)^d (1 - B^s)^D X_t
        = \Theta(B^s) \theta(B) \varepsilon_t

    where :math:`\phi` and :math:`\theta` are the autoregressive and the
    moving average polynomials of an ARIMA process, and :math:`\Phi` and
    :math:`\Theta` the seasonal ones. The series and the noise are zero
    before the first value.

    :param ar: the autoregressive coefficients
        :math:`\phi_1, \dots, \phi_p`, or a single number
    :param ma: the moving average coefficients
        :math:`\theta_1, \dots, \theta_q`, or a single number
    :param seasonal_ar: the seasonal autoregressive coefficients
        :math:`\Phi_1, \dots, \Phi_P`, or a single number
    :param seasonal_ma: the seasonal moving average coefficients
        :math:`\Theta_1, \dots, \Theta_Q`, or a single number
    :param d: the order of differencing, or :math:`d` above
    :param seasonal_d: the order of seasonal differencing, or :math:`D` above
    :param period: the season length, or :math:`s` above
    :param scale: the standard deviation of the noise, or :math:`\sigma`
        above
    :param rng: a custom random number generator
    """

    def __init__(
        self,
        *,
        ar: float | Sequence[float] = (),
        ma: float | Sequence[float] = (),
        seasonal_ar: float | Sequence[float] = (),
        seasonal_ma: float | Sequence[float] = (),
        d: int = 0,
        seasonal_d: int = 0,
        period: int = 12,
        scale: float = 1.0,
        rng: Generator | None = None,
    ) -> None:
        super().__init__(scale=scale, rng=rng)

        self.ar = ar
        self.ma = ma
        self.seasonal_ar = seasonal_ar
        self.seasonal_ma = seasonal_ma
        self.d = d
        self.seasonal_d = seasonal_d
        self.period = period

    def __str__(self) -> str:
        return (
            f"SARIMA({len(self.ar)}, {self.d}, {len(self.ma)})"
            f"({len(self.seasonal_ar)}, {self.seasonal_d}, {len(self.seasonal_ma)}) process "
            f"with period {self.period} and scale {self.scale}."
        )

    def __repr__(self) -> str:
        return (
            f"SARIMAProcess(ar={self.ar}, ma={self.ma}, seasonal_ar={self.seasonal_ar}, "
            f"seasonal_ma={self.seasonal_ma}, d={self.d}, seasonal_d={self.seasonal_d}, "
            f"period={self.period}, scale={self.scale})"
        )

    @property
    def ar(self) -> tuple[float, ...]:
        r"""Autoregressive coefficients :math:`\phi_1, \dots, \phi_p`."""
        return self.__ar

    @ar.setter
    def ar(self, value: float | Sequence[float]) -> None:
        self.__ar = check_coefficients(value=value, name="Autoregressive coefficients")

    @property
    def ma(self) -> tuple[float, ...]:
        r"""Moving average coefficients :math:`\theta_1, \dots, \theta_q`."""
        return self.__ma

    @ma.setter
    def ma(self, value: float | Sequence[float]) -> None:
        self.__ma = check_coefficients(value=value, name="Moving average coefficients")

    @property
    def seasonal_ar(self) -> tuple[float, ...]:
        r"""Seasonal autoregressive coefficients :math:`\Phi_1, \dots, \Phi_P`."""
        return self.__seasonal_ar

    @seasonal_ar.setter
    def seasonal_ar(self, value: float | Sequence[float]) -> None:
        self.__seasonal_ar = check_coefficients(
            value=value, name="Seasonal autoregressive coefficients"
        )

    @property
    def seasonal_ma(self) -> tuple[float, ...]:
        r"""Seasonal moving average coefficients :math:`\Theta_1, \dots, \Theta_Q`."""
        return self.__seasonal_ma

    @seasonal_ma.setter
    def seasonal_ma(self, value: float | Sequence[float]) -> None:
        self.__seasonal_ma = check_coefficients(
            value=value, name="Seasonal moving average coefficients"
        )

    @property
    def d(self) -> int:
        """Order of differencing."""
        return self.__d

    @d.setter
    def d(self, value: int) -> None:
        check_nonnegative_integer(n=value, name="Order of differencing")

        self.__d = int(value)

    @property
    def seasonal_d(self) -> int:
        """Order of seasonal differencing."""
        return self.__seasonal_d

    @seasonal_d.setter
    def seasonal_d(self, value: int) -> None:
        check_nonnegative_integer(n=value, name="Order of seasonal differencing")

        self.__seasonal_d = int(value)

    @property
    def period(self) -> int:
        """Season length."""
        return self.__period

    @period.setter
    def period(self, value: int) -> None:
        check_positive_integer(n=value, name="Period")

        self.__period = int(value)

    @override
    def _polynomials(self) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        period = self.period

        autoregressive = lag_polynomial(self.ar, sign=-1.0)
        for factor in (
            lag_polynomial(self.seasonal_ar, sign=-1.0, step=period),
            differencing_polynomial(self.d),
            differencing_polynomial(self.seasonal_d, step=period),
        ):
            autoregressive = np.convolve(autoregressive, factor)

        moving_average = np.convolve(
            lag_polynomial(self.ma, sign=1.0),
            lag_polynomial(self.seasonal_ma, sign=1.0, step=period),
        )
        return autoregressive, moving_average
