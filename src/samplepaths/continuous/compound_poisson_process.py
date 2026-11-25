"""Compound Poisson process."""

import numpy as np
import numpy.typing as npt
from numpy.random import Generator

from .._compat import override
from ..base import BaseTimeProcess
from ..utils.validation import check_positive_integer, check_positive_number, times_to_increments
from ._jumps import JumpSizes, compound_increments, standard_normal_jumps


class CompoundPoissonProcess(BaseTimeProcess):
    r"""Compound Poisson process.

    The sum of the jumps of a Poisson process with rate :math:`\lambda`,

    .. math::

        X_t = \sum_{k=1}^{N_t} Y_k

    where :math:`N_t` is the number of jumps up to time :math:`t` and the
    jump sizes :math:`Y_k` are independent with a common distribution.

    The jump sizes are drawn by a function which takes a generator and the
    number of jumps, for example
    ``lambda rng, size: rng.exponential(2.0, size)``. Drawing from the
    generator it is given keeps the realizations reproducible.

    :param rate: the rate of the jumps, or :math:`\lambda` above
    :param jumps: a function which draws the jump sizes. The default draws
        from the standard normal distribution.
    :param t: the right hand endpoint of the time interval :math:`[0,t]`
        for the process
    :param rng: a custom random number generator
    """

    _parallel_paths = True

    def __init__(
        self,
        *,
        rate: float = 1.0,
        jumps: JumpSizes | None = None,
        t: float = 1.0,
        rng: Generator | None = None,
    ) -> None:
        super().__init__(t=t, rng=rng)

        self.rate = rate
        self.jumps = jumps if jumps is not None else standard_normal_jumps

    def __str__(self) -> str:
        return f"Compound Poisson process with rate {self.rate} on [0, {self.t}]."

    def __repr__(self) -> str:
        return f"CompoundPoissonProcess(rate={self.rate}, t={self.t})"

    @property
    def rate(self) -> float:
        r"""Rate of the jumps, or :math:`\lambda`."""
        return self.__rate

    @rate.setter
    def rate(self, value: float) -> None:
        check_positive_number(value=value, name="Rate")

        self.__rate = float(value)

    @property
    def jumps(self) -> JumpSizes:
        """Function which draws the jump sizes."""
        return self.__jumps

    @jumps.setter
    def jumps(self, value: JumpSizes) -> None:
        if not callable(value):
            raise TypeError("Jump sizes must be drawn by a function of a generator and a size.")

        self.__jumps = value

    @override
    def sample(self, n: int) -> npt.NDArray[np.float64]:
        check_positive_integer(n=n, name="Number of increments")
        self.set_times(n)

        increments = compound_increments(self.rng, self.rate * self.t / n, (n,), self.jumps)
        return np.concatenate(([0.0], np.cumsum(increments)))

    @override
    def sample_at(self, times: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        times = np.asarray(times, dtype=np.float64)
        zero = times[0] == 0
        if not zero:
            times = np.insert(times, 0, 0.0)
        lengths = times_to_increments(times=times)

        increments = compound_increments(self.rng, self.rate * lengths, lengths.shape, self.jumps)
        samples = np.cumsum(increments)
        if zero:
            samples = np.insert(samples, 0, 0.0)

        return samples

    @override
    def _sample_paths(self, n: int, paths: int) -> npt.NDArray[np.float64]:
        check_positive_integer(n=n, name="Number of increments")
        self.set_times(n)

        increments = compound_increments(self.rng, self.rate * self.t / n, (paths, n), self.jumps)
        return np.concatenate((np.zeros((paths, 1)), np.cumsum(increments, axis=1)), axis=1)
