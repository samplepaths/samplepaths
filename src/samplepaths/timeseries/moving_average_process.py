"""Moving average process."""

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
from numpy.random import Generator

from .._compat import override
from ..utils.validation import check_coefficients
from ._filters import lag_polynomial
from .linear_process import LinearProcess


class MovingAverageProcess(LinearProcess):
    r"""Moving average process, or MA(q).

    .. image:: /_static/moving_average_process.png
        :scale: 50%

    A series in which every value is a linear combination of the current
    and the :math:`q` previous values of Gaussian white noise
    :math:`\varepsilon_t` with standard deviation :math:`\sigma`:

    .. math::

        X_t = \varepsilon_t + \sum_{k=1}^q \theta_k \varepsilon_{t-k}

    The noise before the first value is zero. Values more than :math:`q`
    steps apart are independent.

    :param coefficients: the coefficients :math:`\theta_1, \dots, \theta_q`
        above, or a single number for an MA(1)
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
            f"Moving average process of order {len(self.coefficients)} "
            f"with coefficients {self.coefficients} and scale {self.scale}."
        )

    def __repr__(self) -> str:
        return f"MovingAverageProcess(coefficients={self.coefficients}, scale={self.scale})"

    @property
    def coefficients(self) -> tuple[float, ...]:
        r"""Coefficients :math:`\theta_1, \dots, \theta_q` of the previous noise."""
        return self.__coefficients

    @coefficients.setter
    def coefficients(self, value: float | Sequence[float]) -> None:
        self.__coefficients = check_coefficients(value=value, name="Coefficients")

    @override
    def _polynomials(self) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        return np.ones(1), lag_polynomial(self.coefficients, sign=1.0)
