"""Symmetric Cauchy process."""

from typing import override

import numpy as np
import numpy.typing as npt
from numpy.random import Generator
from scipy.stats import levy

from ..utils.validation import check_positive_integer, times_to_increments
from .brownian_motion import BrownianMotion


class CauchyProcess(BrownianMotion):
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
        super().__init__(
            drift=0.0,
            scale=1.0,
            t=t,
            rng=rng,
        )

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
        samples = np.cumsum(self.scale * self.rng.normal(scale=np.sqrt(levys)))
        if self.drift != 0:
            samples += self.drift * np.cumsum(levys)

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
