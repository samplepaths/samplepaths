"""Base class of the processes built on Brownian motion."""

import numpy as np
import numpy.typing as npt

from .._compat import override
from ..noise.gaussian_noise import GaussianNoise


class BrownianBase(GaussianNoise):
    """Base class of the processes built on Brownian motion.

    Generates Brownian motion with a drift and a scale which are fixed to
    zero and one. :py:class:`BrownianMotion` makes them parameters.

    :param t: the right hand endpoint of the time interval :math:`[0,t]`
        for the process
    :param rng: a custom random number generator
    """

    _drift: float = 0.0
    _scale: float = 1.0

    def _sample_brownian_motion(self, n: int) -> npt.NDArray[np.float64]:
        """Generate a realization of Brownian Motion.

        Generate a Brownian motion realization with n increments. If zero is
        True then include W_0 = 0.
        """
        bm = np.cumsum(self._scale * self._sample_gaussian_noise(n))
        bm = np.insert(bm, [0], 0)

        if self._drift != 0:
            return self._drift * self.times + bm

        return bm

    @override
    def sample(self, n: int) -> npt.NDArray[np.float64]:
        return self._sample_brownian_motion(n)

    def _sample_brownian_motion_at(
        self,
        times: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Generate a Brownian motion at specified times."""
        bm = np.cumsum(self._scale * self._sample_gaussian_noise_at(times))

        if times[0] == 0:
            bm = np.insert(bm, 0, [0])

        if self._drift != 0:
            bm += self._drift * np.asarray(times, dtype=np.float64)

        return bm

    @override
    def sample_at(
        self,
        times: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        return self._sample_brownian_motion_at(times)

    def _sample_brownian_motion_paths(self, n: int, paths: int) -> npt.NDArray[np.float64]:
        """Generate several realizations of Brownian motion, one per row."""
        noise = self._sample_gaussian_noise_paths(n, paths)

        bm = np.concatenate(
            (np.zeros((paths, 1)), np.cumsum(self._scale * noise, axis=1)),
            axis=1,
        )
        if self._drift != 0:
            bm += self._drift * self.times

        return bm

    @override
    def _sample_paths(self, n: int, paths: int) -> npt.NDArray[np.float64]:
        return self._sample_brownian_motion_paths(n, paths)
