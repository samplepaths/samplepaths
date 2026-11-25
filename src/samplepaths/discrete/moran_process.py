"""Moran process."""

from bisect import bisect_right
from typing import override

import numpy as np
import numpy.typing as npt
from numpy.random import Generator

from ..base import BaseSequenceProcess
from ..utils.validation import check_positive_integer, is_integer


def _probabilities(n: int) -> npt.NDArray[np.float64]:
    """Generate the transition probabilities for state :math:`n`.

    :param n: the current state for which to generate transition
        probabilities.
    """
    probabilities = []
    for k in range(1, n):
        p_down = 1.0 * (n - k) / n * k / n
        p_up = 1.0 * k / n * (n - k) / n
        p_same = 1.0 - p_down - p_up
        probabilities.append([p_down, p_same, p_up])

    return np.asarray(probabilities)


class MoranProcess(BaseSequenceProcess):
    """Moran process.

    .. image:: /_static/moran_process.png
        :scale: 50%

    A neutral drift Moran process, typically used to model populations. At
    each step this process will increase by one, decrease by one, or remain
    at the same value between values of zero and the number of
    states, :math:`n`. The process ends when its value reaches zero or the
    maximum valued state.

    :param maximum: the maximum possible value for the process.
    :param rng: a custom random number generator
    """

    def __init__(
        self,
        maximum: int,
        rng: Generator | None = None,
    ) -> None:
        super().__init__(rng=rng)

        self.maximum = maximum

    def __str__(self) -> str:
        return f"Moran process with {self.maximum} states"

    def __repr__(self) -> str:
        return f"MoranProcess(maximum={self.maximum})"

    @property
    def maximum(self) -> int:
        """Maximum value."""
        return self.__maximum

    @maximum.setter
    def maximum(self, n: int) -> None:
        check_positive_integer(n=n, name="Maximum value")

        if n < 3:
            raise ValueError(f"Number of states must be at least 3; got {n}.")

        self.__maximum = n

        # Cumulative distributions of the increment (-1, 0 or 1) per state.
        cumulative = np.cumsum(_probabilities(n), axis=1)
        cumulative /= cumulative[:, -1:]
        self.__cumulative: list[list[float]] = cumulative.tolist()

    def _sample_moran_process(
        self,
        n: int,
        start: int,
    ) -> npt.NDArray[np.int64]:
        """Generate a realization of the Moran process.

        Generate a Moran process until absorption occurs (state 0 or n) or
        length of process reaches length :math:`maximum`.
        """
        if not is_integer(start):
            raise TypeError("Initial state must be a positive integer.")

        if start < 0 or start > self.maximum:
            raise ValueError("Initial state must be between 0 and " + str(self.maximum))

        if not is_integer(n):
            raise TypeError("Sample length must be positive integer.")

        if n < 1:
            raise ValueError("Sample length must be at least 1.")

        start = int(start)
        s = [start]
        # Each step inverts the cumulative distribution of the increment at a
        # uniform variate, which is what Generator.choice does, without its
        # overhead per call. The variates are drawn one at a time because the
        # realization may stop early.
        cumulative = self.__cumulative
        random = self.rng.random
        maximum = self.maximum
        for _ in range(n - 1):
            if start in (0, maximum):
                break
            start += bisect_right(cumulative[start - 1], random()) - 1
            s.append(start)

        return np.array(s, dtype=np.int64)

    @override
    def sample(self, n: int) -> npt.NDArray[np.int64]:
        return self._sample_moran_process(n, 1)

    def sample_with_start(self, n: int, start: int) -> npt.NDArray[np.int64]:
        """Generate a realization starting from a specified state.

        The realization ends early if the process reaches zero or the maximum
        value.

        :param n: the maximum number of values to generate
        :param start: the initial state of the process
        :returns: the realization
        """
        return self._sample_moran_process(n, start)
