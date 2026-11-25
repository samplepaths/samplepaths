"""Base class of the processes defined on a time interval."""

from abc import abstractmethod

import numpy as np
import numpy.typing as npt
from numpy.random import Generator

from .._compat import override
from ..utils import generate_times
from ..utils.validation import check_positive_integer, check_positive_number
from .base_process import BaseProcess

_NOT_SAMPLED = "No realization has been generated yet; call sample() first."


class BaseTimeProcess(BaseProcess):
    """Base class of the processes defined on a time interval.

    Handles the end time of the process and the time values of its
    realizations.

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
        super().__init__(rng=rng)

        self.t = t

        self.__n: int | None = None
        self.__times: npt.NDArray[np.float64] | None = None

    @abstractmethod
    def sample(self, n: int) -> npt.NDArray[np.float64]:
        """Generate a realization.

        :param n: the number of increments to generate
        :returns: the realization
        """

    @property
    def t(self) -> float:
        """End time of the process."""
        return self._t

    @t.setter
    def t(self, value: float) -> None:
        check_positive_number(value=value, name="Time end")

        self._t = float(value)

    @property
    def n(self) -> int:
        """Number of increments of the most recent realization.

        :raises RuntimeError: if no realization has been generated yet
        """
        if self.__n is None:
            raise RuntimeError(_NOT_SAMPLED)
        return self.__n

    @property
    def times(self) -> npt.NDArray[np.float64]:
        """Time values of the most recent realization.

        Holds ``n + 1`` equally spaced values on the interval :math:`[0,t]`.

        :raises RuntimeError: if no realization has been generated yet
        """
        if self.__times is None:
            raise RuntimeError(_NOT_SAMPLED)
        return self.__times

    def set_times_with_t(self, end_t: float, n: int) -> bool:
        """Generate the time values of n increments on [0, end_t].

        :param end_t: the right hand endpoint of the time interval
        :param n: the number of increments
        :returns: ``True`` if the time values changed, ``False`` if they were
            already generated for the same arguments
        """
        check_positive_integer(n=n, name="Number of increments")
        # check_positive_number(end_t, "End time")

        if self.__n == n and self.__times is not None and self.__times[-1] == end_t:
            return False

        self.__n = n
        self.__times = generate_times(end=end_t, n=n)

        return True

    def set_times(self, n: int) -> bool:
        """Generate the time values of n increments on [0, t].

        :param n: the number of increments
        :returns: ``True`` if the time values changed, ``False`` if they were
            already generated for the same arguments
        """
        return self.set_times_with_t(self.t, n)

    def sample_at(
        self,
        times: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Generate a realization at specified times.

        :param times: a vector of increasing time values at which to generate
            the realization
        :returns: the realization
        :raises NotImplementedError: if the process does not support
            sampling at specified times
        """
        raise NotImplementedError(
            f"{type(self).__name__} cannot be sampled at specified times. "
            "Use the sample method with a number of increments."
        )

    @override
    def sample_paths(self, n: int, paths: int) -> npt.NDArray[np.float64]:
        return self._stack_paths(n, paths)

    def _stack_paths(self, n: int, paths: int) -> npt.NDArray[np.float64]:
        """Generate several realizations with one call to sample for each."""
        check_positive_integer(n=paths, name="Number of paths")
        return np.array([self.sample(n) for _ in range(paths)], dtype=np.float64)
