"""Autoregressive process."""

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
from numpy.random import Generator

from .._compat import override
from ..utils.validation import check_coefficients
from ._filters import lag_polynomial
from .linear_process import LinearProcess


class AutoregressiveProcess(LinearProcess):
    r"""Autoregressive process, or AR(p).

    .. image:: /_static/autoregressive_process.png
        :scale: 50%

    A series in which every value is a linear combination of the :math:`p`
    previous values, plus Gaussian white noise :math:`\varepsilon_t` with
    standard deviation :math:`\sigma`:

    .. math::

        X_t = \sum_{k=1}^p \phi_k X_{t-k} + \varepsilon_t

    The values before the first one are zero. The series is stationary when
    the roots of :math:`1 - \phi_1 z - \dots - \phi_p z^p` lie outside the
    unit circle, which for one coefficient means :math:`|\phi_1| < 1`; this
    is not required.

    :param coefficients: the coefficients :math:`\phi_1, \dots, \phi_p`
        above, or a single number for an AR(1)
    :param scale: the standard deviation of the noise, or :math:`\sigma`
        above
    :param rng: a custom random number generator
    """

    def __init__(
        self,
        *,
        coefficients: float | Sequence[float] = 0.5,
        scale: float = 1.0,
        rng: Generator | None = None,
    ) -> None:
        super().__init__(scale=scale, rng=rng)

        self.coefficients = coefficients

    def __str__(self) -> str:
        return (
            f"Autoregressive process of order {len(self.coefficients)} "
            f"with coefficients {self.coefficients} and scale {self.scale}."
        )

    def __repr__(self) -> str:
        return f"AutoregressiveProcess(coefficients={self.coefficients}, scale={self.scale})"

    @property
    def coefficients(self) -> tuple[float, ...]:
        r"""Coefficients :math:`\phi_1, \dots, \phi_p` of the previous values."""
        return self.__coefficients

    @coefficients.setter
    def coefficients(self, value: float | Sequence[float]) -> None:
        self.__coefficients = check_coefficients(value=value, name="Coefficients")

    @override
    def _polynomials(self) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        return lag_polynomial(self.coefficients, sign=-1.0), np.ones(1)
