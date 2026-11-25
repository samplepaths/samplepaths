"""Brownian excursion."""

import numpy as np
import numpy.typing as npt
from numpy.random import Generator

from .._compat import override
from .brownian_bridge import BrownianBridge


class BrownianExcursion(BrownianBridge):
    """Brownian excursion.

    .. image:: /_static/brownian_excursion.png
        :scale: 50%

    A Brownian excursion is a Brownian bridge from (0, 0) to (t, 0) which is
    conditioned to be nonnegative on the interval [0, t].

    Generated using method by

    * Biane, Philippe. "Relations entre pont et excursion du mouvement
      Brownien reel." Ann. Inst. Henri Poincare 22, no. 1 (1986): 1-7.

    * Vervaat, Wim. "A relation between Brownian bridge and Brownian
      excursion." The Annals of Probability (1979): 143-149.

    :param t: the right hand endpoint of the time interval :math:`[0,t]`
        for the process
    :param rng: a custom random number generator
    """

    def __init__(
        self,
        *,
        t: float = 1.0,
        rng: Generator | None = None,
    ) -> None:
        super().__init__(
            b=0.0,
            t=t,
            rng=rng,
        )

    def __str__(self) -> str:
        return f"Brownian excursion on [0, {self.t}]"

    def __repr__(self) -> str:
        return f"BrownianExcursion(t={self.t})"

    def _sample_brownian_excursion(self, n: int) -> npt.NDArray[np.float64]:
        """Generate a Brownian excursion."""
        brownian_bridge = self._sample_brownian_bridge(n)
        idx_min = np.argmin(brownian_bridge)
        indices = (idx_min + np.arange(n + 1)) % n
        s: npt.NDArray[np.float64] = brownian_bridge[indices] - brownian_bridge[idx_min]
        return s

    def _sample_brownian_excursion_at(
        self,
        times: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Generate a Brownian excursion."""
        if times[0] != 0:
            zero = False
            times = np.array([0] + list(times))
        else:
            zero = True

        brownian_bridge = self._sample_brownian_bridge_at(times)
        idx_min = np.argmin(brownian_bridge)
        n = len(brownian_bridge)
        indices = (idx_min + np.arange(n)) % (n - 1)
        s: npt.NDArray[np.float64] = brownian_bridge[indices] - brownian_bridge[idx_min]

        if zero:
            return s

        return s[1:]

    @override
    def sample(self, n: int) -> npt.NDArray[np.float64]:
        return self._sample_brownian_excursion(n)

    @override
    def sample_at(
        self,
        times: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        return self._sample_brownian_excursion_at(times)

    @override
    def sample_paths(self, n: int, paths: int) -> npt.NDArray[np.float64]:
        bridges = self._sample_brownian_bridge_paths(n, paths)

        # Rotate every bridge to start at its minimum.
        idx_min = np.argmin(bridges, axis=1)
        indices = (idx_min[:, np.newaxis] + np.arange(n + 1)) % n
        minimums = bridges[np.arange(paths), idx_min][:, np.newaxis]

        excursions: npt.NDArray[np.float64] = (
            np.take_along_axis(bridges, indices, axis=1) - minimums
        )
        return excursions
