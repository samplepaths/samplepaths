"""Autoregressive integrated moving average process."""

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
from numpy.random import Generator

from .._compat import override
from ..utils.validation import check_coefficients, check_nonnegative_integer
from ._filters import differencing_polynomial, lag_polynomial
from .linear_process import LinearProcess


class ARIMAProcess(LinearProcess):
    r"""Autoregressive integrated moving average process, or ARIMA(p, d, q).

    .. image:: /_static/arima_process.png
        :scale: 50%

    A series whose :math:`d`-th differences follow an autoregressive moving
    average model. With the lag operator :math:`B`, where
    :math:`B X_t = X_{t-1}`, and Gaussian white noise :math:`\varepsilon_t`
    with standard deviation :math:`\sigma`:

    .. math::

        \phi(B) (1 - B)^d X_t = \theta(B) \varepsilon_t

    where :math:`\phi(B) = 1 - \phi_1 B - \dots - \phi_p B^p` and
    :math:`\theta(B) = 1 + \theta_1 B + \dots + \theta_q B^q`. The series
    and the noise are zero before the first value. Without differencing,
    :math:`d = 0`, this is an ARMA(p, q) process; with :math:`d = 1` the
    series wanders like a random walk.

    :param ar: the autoregressive coefficients
        :math:`\phi_1, \dots, \phi_p` above, or a single number
    :param ma: the moving average coefficients
        :math:`\theta_1, \dots, \theta_q` above, or a single number
    :param d: the order of differencing, or :math:`d` above
    :param scale: the standard deviation of the noise, or :math:`\sigma`
        above
    :param rng: a custom random number generator
    """

    def __init__(
        self,
        *,
        ar: float | Sequence[float] = (),
        ma: float | Sequence[float] = (),
        d: int = 0,
        scale: float = 1.0,
        rng: Generator | None = None,
    ) -> None:
        super().__init__(scale=scale, rng=rng)

        self.ar = ar
        self.ma = ma
        self.d = d

    def __str__(self) -> str:
        return (
            f"ARIMA({len(self.ar)}, {self.d}, {len(self.ma)}) process with ar={self.ar}, "
            f"ma={self.ma} and scale {self.scale}."
        )

    def __repr__(self) -> str:
        return f"ARIMAProcess(ar={self.ar}, ma={self.ma}, d={self.d}, scale={self.scale})"

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
    def d(self) -> int:
        """Order of differencing."""
        return self.__d

    @d.setter
    def d(self, value: int) -> None:
        check_nonnegative_integer(n=value, name="Order of differencing")

        self.__d = int(value)

    @override
    def _polynomials(self) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        autoregressive = np.convolve(
            lag_polynomial(self.ar, sign=-1.0),
            differencing_polynomial(self.d),
        )
        return autoregressive, lag_polynomial(self.ma, sign=1.0)
