"""Base class of all processes."""

import copy
from abc import ABC, abstractmethod
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Self

import numpy as np
import numpy.typing as npt
from numpy.random import Generator

from .. import random
from ..utils.validation import check_positive_integer

#: Smallest number of values for which a thread is started by sample_paths.
_VALUES_PER_WORKER = 50_000


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

    #: Whether ``_sample_paths`` generates the realizations with NumPy
    #: operations, which run in parallel on several threads.
    _parallel_paths: bool = False

    def sample_paths(
        self,
        n: int,
        paths: int,
        workers: int = 1,
    ) -> npt.NDArray[np.float64] | npt.NDArray[np.int64]:
        """Generate several independent realizations at once.

        The realizations have the distribution of those of :py:meth:`sample`.
        Many processes generate them together, which is much faster than
        calling :py:meth:`sample` repeatedly. For a given seed, the values
        differ from those of repeated calls to :py:meth:`sample`, because the
        random numbers are drawn in a different order.

        With ``workers`` greater than one, the realizations are generated on
        that many threads, each with its own generator derived from the
        generator of the process. The result is reproducible for a given
        seed and number of workers, and differs between numbers of workers.
        Processes which generate their realizations one after the other
        ignore ``workers``.

        :param n: the number of increments of each realization
        :param paths: the number of realizations
        :param workers: the number of threads to use
        :returns: a two-dimensional array with one realization per row
        """
        return self._run_paths(
            lambda process, count: process._sample_paths(n, count),
            n,
            paths,
            workers,
        )

    def _sample_paths(
        self,
        n: int,
        paths: int,
    ) -> npt.NDArray[np.float64] | npt.NDArray[np.int64]:
        """Generate several realizations, with one call to sample for each.

        Processes which can generate the realizations together override this
        method and set ``_parallel_paths``.
        """
        stacked: npt.NDArray[np.float64] | npt.NDArray[np.int64] = np.array(
            [self.sample(n) for _ in range(paths)]
        )
        return stacked

    def _useful_workers(self, n: int, paths: int) -> int:
        """Return the largest number of threads worth using.

        A thread needs enough values to generate to make up for the cost of
        starting it and of sharing the interpreter with the other threads.
        """
        return max(1, min(paths, paths * n // _VALUES_PER_WORKER))

    def _prepare_paths(self, n: int) -> None:  # noqa: B027
        """Compute what the realizations share, before the work is split.

        The default does nothing.
        """

    def _run_paths(
        self,
        generate: Callable[[Self, int], npt.NDArray[Any]],
        n: int,
        paths: int,
        workers: int,
    ) -> npt.NDArray[Any]:
        """Generate realizations, on several threads when asked to.

        :param generate: generates a number of realizations from a process
        :param n: the number of increments of each realization
        :param paths: the number of realizations
        :param workers: the number of threads to use
        :returns: a two-dimensional array with one realization per row
        """
        check_positive_integer(n=paths, name="Number of paths")
        check_positive_integer(n=workers, name="Number of workers")

        workers = min(workers, self._useful_workers(n, paths))
        if workers == 1 or not self._parallel_paths:
            return generate(self, paths)

        self._prepare_paths(n)

        # Every thread works on a copy of the process with its own generator,
        # spawned from the generator of the process. NumPy releases the
        # interpreter lock while it generates and transforms the numbers.
        try:
            generators = self.rng.spawn(workers)
        except TypeError as error:
            raise TypeError(
                "The generator of the process cannot spawn independent generators. "
                "Create it with numpy.random.default_rng."
            ) from error

        clones = []
        for generator in generators:
            clone = copy.copy(self)
            clone.rng = generator
            clones.append(clone)
        counts = [paths // workers + (index < paths % workers) for index in range(workers)]

        with ThreadPoolExecutor(max_workers=workers) as pool:
            parts = list(pool.map(generate, clones, counts))

        return np.concatenate(parts)
