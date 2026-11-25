"""Correlated Brownian motions."""

from collections.abc import Sequence
from typing import cast

import numpy as np
import numpy.typing as npt
from numpy.random import Generator

from .._compat import override
from ..utils.validation import is_number
from .vector_process import VectorProcess


class CorrelatedBrownianMotions(VectorProcess):
    r"""Correlated Brownian motions.

    .. image:: /_static/correlated_brownian_motions.png
        :scale: 50%

    Several standard Brownian motions :math:`W^1_t, \dots, W^k_t` whose
    increments over the same time step are correlated:

    .. math::

        \operatorname{corr}(W^i_t, W^j_t) = \rho_{ij}

    Each of them alone is a standard Brownian motion. They drive the models
    with several sources of randomness, such as a price and its volatility,
    or the prices of several assets.

    :param correlation: a number between -1 and 1 for two motions, or the
        correlation matrix :math:`\rho` of any number of motions: symmetric,
        with ones on the diagonal and no negative eigenvalue
    :param t: the right hand endpoint of the time interval :math:`[0,t]`
        for the process
    :param rng: a custom random number generator
    """

    def __init__(
        self,
        *,
        correlation: float | Sequence[Sequence[float]] | npt.NDArray[np.float64] = 0.5,
        t: float = 1.0,
        rng: Generator | None = None,
    ) -> None:
        super().__init__(t=t, rng=rng)

        self.correlation = correlation

    def __str__(self) -> str:
        return f"{len(self.components)} correlated Brownian motions on [0, {self.t}]"

    def __repr__(self) -> str:
        return f"CorrelatedBrownianMotions(correlation={self.correlation.tolist()}, t={self.t})"

    @property
    def correlation(self) -> npt.NDArray[np.float64]:
        r"""Correlation matrix of the motions, or :math:`\rho`."""
        return self.__correlation

    @correlation.setter
    def correlation(
        self, value: float | Sequence[Sequence[float]] | npt.NDArray[np.float64]
    ) -> None:
        if is_number(value):
            rho = self._check_correlation(cast(float, value))
            matrix = np.array([[1.0, rho], [rho, 1.0]])
        else:
            try:
                matrix = np.array(value, dtype=np.float64)
            except (TypeError, ValueError) as error:
                raise TypeError(
                    "Correlation must be a number or a square matrix of numbers."
                ) from error
            if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or len(matrix) == 0:
                raise ValueError(
                    f"Correlation matrix must be square; got the shape {matrix.shape}."
                )
            if not np.allclose(matrix, matrix.T) or not np.allclose(np.diag(matrix), 1.0):
                raise ValueError("Correlation matrix must be symmetric with ones on the diagonal.")

        # A factor F with F F' equal to the matrix turns independent motions
        # into correlated ones. The eigenvalues give one even when the matrix
        # is singular, as it is for perfectly correlated motions.
        eigenvalues, eigenvectors = np.linalg.eigh(matrix)
        if eigenvalues.min() < -1e-10:
            raise ValueError("Correlation matrix must not have a negative eigenvalue.")

        self.__correlation = matrix
        self.__factor = eigenvectors * np.sqrt(np.clip(eigenvalues, 0.0, None))
        self.components = tuple(f"motion {index + 1}" for index in range(len(matrix)))

    @override
    def _sample_paths(self, n: int, paths: int) -> npt.NDArray[np.float64]:
        self._check_paths(n, paths)

        count = len(self.components)
        noise = self.rng.normal(scale=np.sqrt(self.t / n), size=(paths, count, n))
        realizations = np.zeros((paths, count, n + 1))
        np.cumsum(self.__factor @ noise, axis=-1, out=realizations[..., 1:])
        return realizations
