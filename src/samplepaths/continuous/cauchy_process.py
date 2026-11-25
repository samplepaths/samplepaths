"""Symmetric Cauchy process."""

import numpy as np
import numpy.typing as npt
from numpy.random import Generator
from scipy.stats import levy

from .._compat import override
from ..utils.validation import check_positive_integer, times_to_increments
from .brownian_base import BrownianBase


class CauchyProcess(BrownianBase):
    """Symmetric Cauchy process.

    .. image:: /_static/cauchy_process.png
        :scale: 50%

    The symmetric Cauchy process is a Brownian motion with a Levy subordinator
    using location parameter 0 and scale parameter :math:`t^2/2`.

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
        super().__init__(t=t, rng=rng)

    def __str__(self) -> str:
        return f"Cauchy process on [0, {self.t}]"

    def __repr__(self) -> str:
        return f"CauchyProcess(t={self.t})"

    def _sample_cauchy_process(self, n: int) -> npt.NDArray[np.float64]:
        """Generate a realization of a Cauchy process."""
        check_positive_integer(n=n, name="Number of increments")
        self.set_times(n)

        delta_t = 1.0 * self.t / n

        levys = np.asarray(
            levy.rvs(
                loc=0,
                scale=delta_t**2 / 2,
                size=n,
                random_state=self.rng,
            ),
            dtype=np.float64,
        )

        return np.insert(self._subordinate(levys), 0, 0.0)

    def _subordinate(self, levys: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Evaluate the Brownian motion over the increments of the subordinator.

        The Brownian increments are generated from the increments of the Levy
        subordinator directly. Accumulating the subordinator first and taking
        differences loses the small increments next to a large one.
        """
        samples: npt.NDArray[np.float64] = np.cumsum(self.rng.normal(scale=np.sqrt(levys)))
        return samples

    def _sample_cauchy_process_at(
        self,
        times: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Generate a realization of a Cauchy process."""
        if times[0] != 0:
            zero = False
            times = np.insert(times, 0, [0])
        else:
            zero = True

        deltas = times_to_increments(times=times)
        # Standard Levy variates, scaled per increment.
        standard = np.asarray(
            levy.rvs(
                loc=0,
                scale=1.0,
                size=len(deltas),
                random_state=self.rng,
            ),
            dtype=np.float64,
        )
        levys = standard * (deltas**2 / 2)

        samples = self._subordinate(levys)
        if zero:
            samples = np.insert(samples, 0, 0.0)

        return samples

    @override
    def sample(self, n: int) -> npt.NDArray[np.float64]:
        return self._sample_cauchy_process(n)

    @override
    def sample_at(
        self,
        times: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        return self._sample_cauchy_process_at(times)

    @override
    def _sample_paths(self, n: int, paths: int) -> npt.NDArray[np.float64]:
        check_positive_integer(n=n, name="Number of increments")
        check_positive_integer(n=paths, name="Number of paths")
        self.set_times(n)

        delta_t = 1.0 * self.t / n
        levys = np.asarray(
            levy.rvs(
                loc=0,
                scale=delta_t**2 / 2,
                size=(paths, n),
                random_state=self.rng,
            ),
            dtype=np.float64,
        )

        samples = np.cumsum(self.rng.normal(scale=np.sqrt(levys)), axis=1)

        return np.concatenate((np.zeros((paths, 1)), samples), axis=1)
