"""Squared Bessel process."""

import numpy as np
import numpy.typing as npt

from .._compat import override
from ..utils.validation import check_positive_integer
from .bessel_process import BesselProcess


class SquaredBesselProcess(BesselProcess):
    r"""Squared Bessel process.

    .. image:: /_static/squared_bessel_process.png
        :scale: 50%

    The square of a Bessel process: :math:`\|\mathbf{W}_t\|^2`.

    The Bessel process is the Euclidean norm of an :math:`n`-dimensional
    Wiener process, e.g. :math:`\|\mathbf{W}_t\|`

    :param dim: the number of underlying independent Brownian motions to
        use
    :param t: the right hand endpoint of the time interval :math:`[0,t]`
        for the process
    """

    def __str__(self) -> str:
        return f"Squared Bessel process of {self.dim} Wiener processes on [0, {self.t}]"

    def __repr__(self) -> str:
        return f"SquaredBesselProcess(dim={self.dim}, t={self.t})"

    def _sample_squared_bessel_process(
        self,
        n: int,
    ) -> npt.NDArray[np.float64]:
        """Generate a realization of a squared Bessel process."""
        check_positive_integer(n=n)

        samples = np.array([self._sample_brownian_motion(n) for _ in range(self.dim)])

        squares: npt.NDArray[np.float64] = np.sum(np.square(samples), axis=0)
        return squares

    def _sample_squared_bessel_process_at(
        self,
        times: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Generate a realization of a squared Bessel process."""
        samples = np.array([self._sample_brownian_motion_at(times) for _ in range(self.dim)])

        squares: npt.NDArray[np.float64] = np.sum(np.square(samples), axis=0)
        return squares

    @override
    def sample(self, n: int) -> npt.NDArray[np.float64]:
        return self._sample_squared_bessel_process(n)

    @override
    def sample_at(self, times: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return self._sample_squared_bessel_process_at(times)

    @override
    def _sample_paths(self, n: int, paths: int) -> npt.NDArray[np.float64]:
        samples = np.array([self._sample_brownian_motion_paths(n, paths) for _ in range(self.dim)])

        squares: npt.NDArray[np.float64] = np.sum(np.square(samples), axis=0)
        return squares
