"""Multifractional Brownian motion process."""

import inspect
from collections.abc import Callable
from typing import override

import numpy as np
import numpy.typing as npt
from numpy.random import Generator
from scipy.special import gamma

from ..base import BaseTimeProcess
from ..utils.validation import check_positive_integer

#: Largest number of increments for which the weights are kept between
#: realizations. The weights take 4 n squared bytes: 17 MB at this limit.
_CACHE_MAX_INCREMENTS = 2048


class MultifractionalBrownianMotion(BaseTimeProcess):
    r"""Multifractional Brownian motion process.

    .. image:: /_static/multifractional_brownian_motion.png
        :scale: 50%

    A multifractional Brownian motion generalizes a fractional Brownian
    motion with a Hurst parameter which is a function of time,
    :math:`h(t)`. If the Hurst is constant, the process is a fractional
    Brownian motion. If Hurst is constant equal to 0.5, the process is a
    Brownian motion.

    Approximate method originally proposed for fBm in

    * Rambaldi, Sandro, and Ombretta Pinazza. "An accurate fractional Brownian
      motion generator." Physica A: Statistical Mechanics and its Applications
      208, no. 1 (1994): 21-30.

    Adapted to approximate mBm in

    * Muniandy, S. V., and S. C. Lim. "Modeling of locally self-similar
      processes using multifractional Brownian motion of Riemann-Liouville
      type." Physical Review E 63, no. 4 (2001): 046104.

    :param Callable[[float], float] hurst: a callable with one argument :math:`h(t)` such that
        :math:`h(t') \in (0, 1) \forall t' \in [0, t]`. Default is
        :math:`h(t) = 0.5`.
    :param t: the right hand endpoint of the time interval :math:`[0,t]`
        for the process
    :param rng: a custom random number generator
    """

    def __init__(
        self,
        *,
        hurst: Callable[[float], float] | None = None,
        t: float = 1.0,
        rng: Generator | None = None,
    ) -> None:
        super().__init__(t=t, rng=rng)

        self.__weights_key: tuple[int, float, Callable[[float], float]] | None = None
        self.__weights: list[npt.NDArray[np.float64]] = []

        self.hurst = hurst if hurst is not None else lambda x: 0.5

    def __str__(self) -> str:
        return (
            f"Multifractional Brownian motion with Hurst function {self.hurst.__name__} "
            f"on [0, {self.t}]."
        )

    def __repr__(self) -> str:
        return f"MultifractionalBrownianMotion(hurst={self.hurst.__name__}, t={self.t})"

    @property
    def hurst(self) -> Callable[[float], float]:
        """Hurst function."""
        return self._hurst

    @hurst.setter
    def hurst(self, value: Callable[[float], float]) -> None:
        try:
            num_args = len(inspect.signature(value).parameters)
        except Exception as e:
            raise TypeError("Hurst parameter must be an inspectable function.") from e

        if not callable(value) or num_args != 1:
            raise TypeError("Hurst parameter must be a function of one argument.")

        self._hurst = value

    def _get_hurst(self, value: Callable[[float], float]) -> list[float]:
        hs = [0.0] * len(self.times)
        for i, t in enumerate(self.times):
            h = value(t)
            if h <= 0 or h >= 1:
                raise ValueError(f"Hurst range must be on interval (0, 1). Got {h} for t={t}.")
            hs[i] = h

        return hs

    def _sample_multifractional_brownian_motion(self, n: int) -> npt.NDArray[np.float64]:
        """Generate Riemann-Liouville mBm."""
        gn = self.rng.normal(0.0, 1.0, n)
        self.set_times(n)

        dt = 1.0 * self.t / self.n
        coefs = (gn / np.sqrt(dt)) * dt

        weights = self._weights(n, dt)

        mbm = np.zeros(self.n + 1)
        for k in range(1, self.n + 1):
            # The weights of time k apply to the increments in reverse order.
            mbm[k] = coefs[:k] @ weights[k][::-1]

        return mbm

    def _weights(self, n: int, dt: float) -> list[npt.NDArray[np.float64]]:
        """Return the weights of the Riemann-Liouville method for every time.

        Entry k holds the weights of the times 1 to k for the Hurst value at
        time k. The weights depend on the number of increments, the end time
        and the Hurst function only, so they are kept for the next realization
        unless they would take too much memory.
        """
        key = (n, self.t, self._hurst)
        cached = self.__weights_key
        if cached is not None and cached[:2] == key[:2] and cached[2] is key[2]:
            return self.__weights

        hs = self._get_hurst(self.hurst)
        times = self.times

        weights = [np.zeros(0)]
        for k in range(1, n + 1):
            hurst = hs[k]
            elapsed = times[1 : k + 1]
            weights.append(
                np.sqrt((elapsed ** (2 * hurst) - (elapsed - dt) ** (2 * hurst)) / (2 * hurst * dt))
                / gamma(hurst + 0.5)
            )

        if n <= _CACHE_MAX_INCREMENTS:
            self.__weights_key = key
            self.__weights = weights

        return weights

    @override
    def sample(self, n: int) -> npt.NDArray[np.float64]:
        return self._sample_multifractional_brownian_motion(n)

    @override
    def sample_paths(self, n: int, paths: int) -> npt.NDArray[np.float64]:
        check_positive_integer(n=n, name="Number of increments")
        check_positive_integer(n=paths, name="Number of paths")

        gn = self.rng.normal(0.0, 1.0, (paths, n))
        self.set_times(n)

        dt = 1.0 * self.t / n
        coefs = (gn / np.sqrt(dt)) * dt
        weights = self._weights(n, dt)

        mbm = np.zeros((paths, n + 1))
        for k in range(1, n + 1):
            mbm[:, k] = coefs[:, :k] @ weights[k][::-1]

        return mbm
