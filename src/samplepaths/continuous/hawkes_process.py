"""Hawkes process."""

import math

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


class HawkesProcess(BaseProcess):
    r"""Hawkes process.

    A self-exciting point process: every arrival raises the rate of the
    arrivals which follow, and the effect fades exponentially. The rate at
    time :math:`t` is

    .. math::

        \lambda(t) = \mu + \sum_{t_i < t} \alpha e^{-\beta (t - t_i)}

    where the :math:`t_i` are the earlier arrivals. Arrivals come in
    clusters. On average an arrival triggers :math:`\alpha / \beta` further
    arrivals, which must be less than one, and the long-run rate of the
    arrivals is :math:`\mu / (1 - \alpha / \beta)`.

    This class generates samples of the times of the arrivals, using Ogata's
    thinning method.

    :param baseline: the rate without excitation, or :math:`\mu` above
    :param excitation: the increase of the rate at an arrival, or
        :math:`\alpha` above
    :param decay: the rate at which the excitation fades, or :math:`\beta`
        above, larger than ``excitation``
    :param rng: a custom random number generator
    """

    def __init__(
        self,
        *,
        baseline: float = 1.0,
        excitation: float = 0.5,
        decay: float = 1.0,
        rng: Generator | None = None,
    ) -> None:
        super().__init__(rng=rng)

        check_nonnegative_number(value=excitation, name="Excitation")
        check_positive_number(value=decay, name="Decay")
        self._check_stationary(excitation, decay)
        self.__excitation = float(excitation)
        self.__decay = float(decay)

        self.baseline = baseline

    def __str__(self) -> str:
        return (
            f"Hawkes process with baseline {self.baseline}, excitation {self.excitation} "
            f"and decay {self.decay}."
        )

    def __repr__(self) -> str:
        return (
            f"HawkesProcess(baseline={self.baseline}, excitation={self.excitation}, "
            f"decay={self.decay})"
        )

    @staticmethod
    def _check_stationary(excitation: float, decay: float) -> None:
        """Ensure that an arrival triggers less than one arrival on average."""
        if excitation >= decay:
            raise ValueError(
                "Excitation must be smaller than decay, or the arrivals multiply without "
                f"bound; got excitation={excitation}, decay={decay}."
            )

    @property
    def baseline(self) -> float:
        r"""Rate without excitation, or :math:`\mu`."""
        return self.__baseline

    @baseline.setter
    def baseline(self, value: float) -> None:
        check_positive_number(value=value, name="Baseline")

        self.__baseline = float(value)

    @property
    def excitation(self) -> float:
        r"""Increase of the rate at an arrival, or :math:`\alpha`."""
        return self.__excitation

    @excitation.setter
    def excitation(self, value: float) -> None:
        check_nonnegative_number(value=value, name="Excitation")
        self._check_stationary(value, self.decay)

        self.__excitation = float(value)

    @property
    def decay(self) -> float:
        r"""Rate at which the excitation fades, or :math:`\beta`."""
        return self.__decay

    @decay.setter
    def decay(self, value: float) -> None:
        check_positive_number(value=value, name="Decay")
        self._check_stationary(self.excitation, value)

        self.__decay = float(value)

    def _sample_hawkes_process(
        self,
        n: int | None = None,
        length: float | None = None,
    ) -> npt.NDArray[np.float64]:
        """Generate the arrival times, by number of arrivals or by length of time."""
        baseline = self.baseline
        excitation = self.excitation
        decay = self.decay
        exponential = self.rng.exponential
        random = self.rng.random

        # Ogata's thinning method. Between arrivals the rate only decreases,
        # so the rate just after the last candidate bounds it from above. A
        # candidate arrival drawn at that bound is kept with the probability
        # that the true rate has not faded below it.
        times = [0.0]
        time = 0.0
        excited = 0.0  # the part of the rate which comes from past arrivals
        while True:
            bound = baseline + excited
            wait = exponential(1.0 / bound)
            time += wait
            excited *= math.exp(-decay * wait)
            if random() * bound <= baseline + excited:
                times.append(time)
                excited += excitation
                if n is not None and len(times) > n:
                    break
                if length is not None and time > length:
                    break

        return np.array(times)

    @override
    def sample(self, n: int) -> npt.NDArray[np.float64]:
        check_positive_integer(n=n)
        return self._sample_hawkes_process(n=n)

    def sample_with_length(self, length: float) -> npt.NDArray[np.float64]:
        """Generate the arrival times over a length of time.

        :param length: the length of time to simulate
        :returns: the arrival times, starting at zero and ending with the first
            arrival after ``length``
        """
        check_positive_number(value=length, name="Sample length")
        return self._sample_hawkes_process(length=length)
