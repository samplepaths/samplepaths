"""Poisson process."""

import numpy as np
import numpy.typing as npt
from numpy.random import Generator

from .._compat import override
from ..base import BaseProcess
from ..utils.validation import (
    check_nonnegative_number,
    check_positive_integer,
    check_positive_number,
)


class PoissonProcess(BaseProcess):
    r"""Poisson process.

    .. image:: /_static/poisson_process.png
        :scale: 50%

    A Poisson process with rate :math:`\lambda` is a count of occurrences of
    i.i.d. exponential random variables with mean :math:`1/\lambda`. This class
    generates samples of times for which cumulative exponential random
    variables occur.

    :param rate: the parameter :math:`\lambda` which defines the rate of
        occurrences of the process
    :param rng: a custom random number generator
    """

    def __init__(
        self,
        *,
        rate: float = 1.0,
        rng: Generator | None = None,
    ) -> None:
        super().__init__(rng=rng)

        self.rate = rate

    def __str__(self) -> str:
        return f"Poisson process with rate {self.rate}."

    def __repr__(self) -> str:
        return f"PoissonProcess(rate={self.rate})"

    @property
    def rate(self) -> float:
        """Rate parameter."""
        return self.__rate

    @rate.setter
    def rate(self, value: float) -> None:
        check_nonnegative_number(value=value, name="Arrival rate")

        self.__rate = value

    def _sample_poisson_process(
        self,
        n: int | None = None,
        length: float | None = None,
    ) -> npt.NDArray[np.float64]:
        """Generate a realization of a Poisson process.

        Generate a poisson process sample up to count of length if time=False,
        otherwise generate a sample up to time t=length if time=True
        """
        if n is not None:
            check_positive_integer(n=n)

            exponentials = self.rng.exponential(
                scale=1.0 / self.rate,
                size=n,
            )

            return np.concatenate(([0.0], np.cumsum(exponentials)))

        if length is not None:
            check_positive_number(value=length, name="Sample length")

            t = 0.0
            times = [0.0]
            exp_rate = 1.0 / self.rate

            while t < length:
                t += self.rng.exponential(scale=exp_rate)
                times.append(t)

            return np.array(times)

        raise ValueError("Must provide either argument n or length.")

    @override
    def sample(
        self,
        n: int,
    ) -> npt.NDArray[np.float64]:
        return self._sample_poisson_process(n=n)

    def sample_with_length(
        self,
        length: float,
    ) -> npt.NDArray[np.float64]:
        """Generate the arrival times over a length of time.

        :param length: the length of time to simulate
        :returns: the arrival times, starting at zero and ending with the first
            arrival after ``length``
        """
        return self._sample_poisson_process(length=length)

    @override
    def sample_paths(self, n: int, paths: int) -> npt.NDArray[np.float64]:
        check_positive_integer(n=n)
        check_positive_integer(n=paths, name="Number of paths")

        exponentials = self.rng.exponential(scale=1.0 / self.rate, size=(paths, n))
        return np.concatenate((np.zeros((paths, 1)), np.cumsum(exponentials, axis=1)), axis=1)
