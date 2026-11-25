"""Fractional Brownian motion process."""

import numpy as np
import numpy.typing as npt

from .._compat import override
from ..noise import FractionalGaussianNoise


class FractionalBrownianMotion(FractionalGaussianNoise):
    """Fractional Brownian motion process.

    .. image:: /_static/fractional_brownian_motion.png
        :scale: 50%

    A fractional Brownian motion (discretely sampled) has correlated Gaussian
    increments defined by Hurst parameter :math:`H`. When :math:`H = 1/2`,
    the process is a standard Brownian motion. When :math:`H > 1/2`, the
    increments are positively correlated. When :math:`H < 1/2`, the
    increments are negatively correlated.

    Hosking's method:

    * Hosking, Jonathan RM. "Modeling persistence in hydrological time series
      using fractional differencing." Water resources research 20, no. 12
      (1984): 1898-1908.

    Davies Harte method:

    * Davies, Robert B., and D. S. Harte. "Tests for Hurst effect." Biometrika
      74, no. 1 (1987): 95-101.

    :param hurst: the Hurst parameter on the interval (0, 1)
    :param t: the right hand endpoint of the time interval :math:`[0,t]`
        for the process
    :param rng: a custom random number generator
    """

    def __str__(self) -> str:
        return f"Fractional Brownian motion with Hurst {self.hurst} on [0, {self.t}]."

    def __repr__(self) -> str:
        return f"FractionalBrownianMotion(hurst={self.hurst}, t={self.t})"

    @staticmethod
    def _accumulate(fgn: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Turn the increments into the motion, which starts at zero."""
        return np.insert(fgn.cumsum(), [0], 0)

    @override
    def sample(self, n: int) -> npt.NDArray[np.float64]:
        return self.sample_davies_harte(n)

    @override
    def sample_davies_harte(self, n: int) -> npt.NDArray[np.float64]:
        return self._accumulate(self._sample_fractional_gaussian_noise(n, "davies-harte"))

    @override
    def sample_hosking(self, n: int) -> npt.NDArray[np.float64]:
        return self._accumulate(self._sample_fractional_gaussian_noise(n, "hosking"))

    @override
    def sample_paths(self, n: int, paths: int) -> npt.NDArray[np.float64]:
        fgn = self._davies_harte_paths(n, paths)
        return np.concatenate((np.zeros((paths, 1)), np.cumsum(fgn, axis=1)), axis=1)
