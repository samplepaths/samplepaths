"""Base class of the linear time series models."""

from abc import abstractmethod

import numpy as np
import numpy.typing as npt
from numpy.random import Generator

from .._compat import override
from ..base import BaseSequenceProcess
from ..utils.validation import check_positive_integer, check_positive_number
from ._filters import apply_autoregression, apply_moving_average, filter_noise

#: Estimated costs in nanoseconds, used by sample_paths to choose between
#: filtering the realizations one after the other in compiled code and
#: advancing all of them together, one step at a time: of a value filtered
#: in compiled code, for a polynomial of degree zero and for each further
#: degree; of a call to numpy at a step; and of a value in that call.
_FILTERED_VALUE_COST = 4.0
_FILTERED_DEGREE_COST = 0.5
_STEP_CALL_COST = 1000.0
_STEP_VALUE_COST = 0.5


class LinearProcess(BaseSequenceProcess):
    r"""Base class of the linear time series models.

    A series :math:`X_t` driven by Gaussian white noise
    :math:`\varepsilon_t` with standard deviation :math:`\sigma` through two
    polynomials of the lag operator :math:`B`, where
    :math:`B X_t = X_{t-1}`:

    .. math::

        a(B) X_t = b(B) \varepsilon_t

    The series and the noise are zero before the first value. A subclass
    defines the two polynomials.

    :param scale: the standard deviation of the noise, or :math:`\sigma`
        above
    :param rng: a custom random number generator
    """

    _parallel_paths = True

    def __init__(
        self,
        *,
        scale: float = 1.0,
        rng: Generator | None = None,
    ) -> None:
        super().__init__(rng=rng)

        self.scale = scale

    @property
    def scale(self) -> float:
        r"""Standard deviation of the noise, or :math:`\sigma`."""
        return self.__scale

    @scale.setter
    def scale(self, value: float) -> None:
        check_positive_number(value=value, name="Scale")

        self.__scale = float(value)

    @abstractmethod
    def _polynomials(self) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Return the autoregressive and the moving average polynomials.

        :returns: the coefficients of :math:`a(B)` and of :math:`b(B)`, by
            increasing power of :math:`B`
        """

    def _filter(self, shape: tuple[int, ...]) -> npt.NDArray[np.float64]:
        """Generate white noise of a shape and turn it into series."""
        autoregressive, moving_average = self._polynomials()
        noise = self.rng.normal(scale=self.scale, size=shape)
        return filter_noise(noise, autoregressive, moving_average)

    @override
    def sample(self, n: int) -> npt.NDArray[np.float64]:
        check_positive_integer(n=n)
        return self._filter((n,))

    def _advance_together(self, paths: int) -> bool:
        """Tell whether to advance the realizations together, step by step.

        The alternative filters them one after the other in compiled code,
        which is faster for few realizations or many coefficients.
        """
        autoregressive, _ = self._polynomials()
        calls = 2 * np.count_nonzero(autoregressive[1:])
        together = calls * (_STEP_CALL_COST + _STEP_VALUE_COST * paths)
        in_turn = (_FILTERED_VALUE_COST + _FILTERED_DEGREE_COST * len(autoregressive)) * paths
        return bool(together < in_turn)

    def _moving_average_paths(self, n: int, paths: int, together: bool) -> npt.NDArray[np.float64]:
        """Generate white noise with the moving average polynomial applied.

        Releases the interpreter lock, so it can run on several threads.

        :param n: the number of values of each realization
        :param paths: the number of realizations
        :param together: whether the realizations will be advanced together,
            which asks for an array stored step by step
        :returns: an array with one realization per row
        """
        _, moving_average = self._polynomials()
        if together:
            noise = self.rng.normal(scale=self.scale, size=(n, paths))
            return apply_moving_average(noise, moving_average).T

        noise = self.rng.normal(scale=self.scale, size=(paths, n))
        return apply_moving_average(noise.T, moving_average).T

    def _autoregress_paths(
        self, series: npt.NDArray[np.float64], together: bool
    ) -> npt.NDArray[np.float64]:
        """Apply the autoregressive polynomial to realizations.

        :param series: the result of ``_moving_average_paths``
        :param together: whether to advance the realizations together
        :returns: an array with one realization per row
        """
        autoregressive, _ = self._polynomials()
        if len(autoregressive) == 1:
            return series

        if together:
            steps = np.ascontiguousarray(series.T)
            apply_autoregression(steps, autoregressive)
            return steps.T

        return filter_noise(series, autoregressive, np.ones(1))

    @override
    def _sample_paths(self, n: int, paths: int) -> npt.NDArray[np.float64]:
        check_positive_integer(n=n)
        check_positive_integer(n=paths, name="Number of paths")

        together = self._advance_together(paths)
        return self._autoregress_paths(self._moving_average_paths(n, paths, together), together)

    @override
    def sample_paths(self, n: int, paths: int, workers: int = 1) -> npt.NDArray[np.float64]:
        check_positive_integer(n=n)
        check_positive_integer(n=paths, name="Number of paths")

        # The threads generate the noise and apply the moving average, which
        # releases the interpreter lock. The autoregression then runs on one
        # thread: it does not release the lock.
        together = self._advance_together(paths)
        series = self._run_paths(
            lambda process, count: process._moving_average_paths(n, count, together),
            n,
            paths,
            workers,
        )
        return self._autoregress_paths(series, together)
