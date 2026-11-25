"""Base class of all processes."""

from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt
from numpy.random import Generator

from .. import random


class BaseProcess(ABC):
    """Base class of all processes.

    Handles the random number generator and defines the sampling interface.

    :param rng: a custom random number generator
    """

    def __init__(
        self,
        *,
        rng: Generator | None = None,
    ) -> None:
        self.rng = rng

    @property
    def rng(self) -> Generator:
        """Random number generator used by the process.

        This is the generator passed at instantiation, or the package default
        :py:data:`samplepaths.random.generator` when none was given.
        """
        return self.__rng or random.generator

    @rng.setter
    def rng(self, value: Generator | None) -> None:
        if value is None:
            self.__rng = None
        elif isinstance(value, Generator):
            self.__rng = value
        else:
            raise TypeError("rng must be of type `numpy.random.Generator`")

    @abstractmethod
    def sample(self, n: int) -> npt.NDArray[np.float64] | npt.NDArray[np.int64]:
        """Generate a realization.

        :param n: the number of increments to generate
        :returns: the realization. Processes with integer values, such as
            the state of a Markov chain, return an array of integers.
        """
