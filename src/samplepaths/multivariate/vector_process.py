"""Base class of the processes with several components."""

import numpy as np
import numpy.typing as npt

from .._compat import override
from ..base import BaseTimeProcess
from ..utils.validation import check_numeric, check_positive_integer


class VectorProcess(BaseTimeProcess):
    r"""Base class of the processes with several components.

    A realization is a two-dimensional array with one component per row and
    ``n + 1`` values per component, at the times given by ``times``. The
    names of the components are in ``components``, in the order of the rows.
    ``sample_paths`` returns a three-dimensional array of shape
    ``(paths, components, n + 1)``.

    :param t: the right hand endpoint of the time interval :math:`[0,t]`
        for the process
    :param rng: a custom random number generator
    """

    _parallel_paths = True

    #: Names of the components, in the order of the rows of a realization.
    components: tuple[str, ...] = ()

    #: Smallest number of realizations worth a thread, for the models which
    #: advance step by step: with fewer, the threads compete for the
    #: interpreter. ``None`` for the models computed with array operations.
    _paths_per_worker: int | None = None

    #: Largest number of threads of the models which advance step by step.
    _max_stepping_workers = 6

    @override
    def _useful_workers(self, n: int, paths: int) -> int:
        workers = super()._useful_workers(n, paths)
        if self._paths_per_worker is None:
            return workers
        return max(1, min(workers, self._max_stepping_workers, paths // self._paths_per_worker))

    @staticmethod
    def _check_correlation(value: float) -> float:
        """Check a correlation coefficient and return it as a float."""
        check_numeric(value=value, name="Correlation")
        if not -1 <= value <= 1:
            raise ValueError(f"Correlation must be between -1 and 1; got {value}.")
        return float(value)

    def _wiener_increments(
        self, n: int, paths: int, correlation: float
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Generate the increments of two correlated Wiener processes.

        :param n: the number of increments
        :param paths: the number of realizations
        :param correlation: the correlation of the two processes
        :returns: two arrays of shape ``(n, paths)``, stored step by step
        """
        scale = np.sqrt(self.t / n)
        first = self.rng.normal(scale=scale, size=(n, paths))
        second = self.rng.normal(scale=scale * np.sqrt(1.0 - correlation**2), size=(n, paths))
        second += correlation * first
        return first, second

    def _check_paths(self, n: int, paths: int) -> None:
        """Check the arguments of a batch and set the time values."""
        check_positive_integer(n=n)
        check_positive_integer(n=paths, name="Number of paths")
        self.set_times(n)

    @override
    def sample(self, n: int) -> npt.NDArray[np.float64]:
        realization: npt.NDArray[np.float64] = self._sample_paths(n, 1)[0]
        return realization

    @override
    def sample_at(self, times: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        raise NotImplementedError(
            f"{type(self).__name__} cannot be sampled at specified times. "
            "Use the sample method with a number of increments."
        )

    @override
    def sample_paths(self, n: int, paths: int, workers: int = 1) -> npt.NDArray[np.float64]:
        """Generate several independent realizations.

        :param n: the number of increments of each realization
        :param paths: the number of realizations
        :param workers: the number of threads to use
        :returns: a three-dimensional array of shape
            ``(paths, components, n + 1)``
        """
        return super().sample_paths(n, paths, workers)
