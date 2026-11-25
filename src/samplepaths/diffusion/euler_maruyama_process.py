"""Base class of the diffusions sampled step by step."""

from abc import abstractmethod
from types import SimpleNamespace
from typing import Any

import numpy as np
import numpy.typing as npt
from numpy.random import Generator

from .._compat import override
from .._functions import ARRAY, SCALAR
from ..noise.gaussian_noise import GaussianNoise
from ..utils.validation import check_numeric, check_positive_integer

#: Number of realizations from which sample_paths advances all of them
#: together instead of one after the other.
_VECTORIZE_FROM_PATHS = 32

#: Limits on the threads used by sample_paths.
_MAX_WORKERS = 4
_PATHS_PER_WORKER = 20_000


class EulerMaruyamaProcess(GaussianNoise):
    r"""Base class of the diffusions sampled step by step.

    A process :math:`X_t` which satisfies the stochastic differential equation

    .. math::

        dX_t = a(X_t) dt + b(X_t) dW_t

    with Wiener process :math:`W_t`. Realizations are generated using the
    Euler-Maruyama method, and every value is confined to the domain of the
    process, so that a step which leaves the domain does not end the
    realization.

    A subclass defines the drift :math:`a` and the volatility :math:`b` in
    ``_coefficients``, the domain in ``_confine`` and the initial value used
    by :py:meth:`sample` in ``_initial``.

    :param t: the right hand endpoint of the time interval :math:`[0,t]`
        for the process
    :param rng: a custom random number generator
    """

    #: Initial value of the realizations of sample and sample_paths.
    _initial: float = 1.0

    def __init__(
        self,
        *,
        t: float = 1.0,
        rng: Generator | None = None,
    ) -> None:
        super().__init__(t=t, rng=rng)

    @abstractmethod
    def _coefficients(self, x: Any, functions: SimpleNamespace) -> tuple[Any, Any]:
        """Return the drift and the volatility at a value of the process.

        :param x: a value of the process, or an array of values
        :param functions: the functions ``sqrt``, ``log`` and ``clip`` for
            the type of ``x``
        :returns: the drift and the volatility
        """

    @abstractmethod
    def _confine(self, x: Any, functions: SimpleNamespace) -> Any:
        """Confine a value, or an array of values, to the domain."""

    def _check_initial(self, initial: float) -> float:
        """Check an initial value and return it as a float."""
        check_numeric(value=initial, name="Initial")
        return float(self._confine(float(initial), SCALAR))

    def _advance(self, gns: list[float], x: float, delta_t: float) -> list[float]:
        """Apply the Euler-Maruyama method to one realization of the noise."""
        coefficients = self._coefficients
        confine = self._confine

        values = [x]
        for gn in gns:
            drift, volatility = coefficients(x, SCALAR)
            x = confine(x + drift * delta_t + volatility * gn, SCALAR)
            values.append(x)

        return values

    def _sample_from(self, n: int, initial: float) -> npt.NDArray[np.float64]:
        """Generate a realization from an initial value."""
        check_positive_integer(n=n)
        x = self._check_initial(initial)

        gns: list[float] = self._sample_gaussian_noise(n).tolist()
        return np.array(self._advance(gns, x, 1.0 * self.t / n))

    def _sample_paths_from(
        self,
        n: int,
        paths: int,
        initial: float,
    ) -> npt.NDArray[np.float64]:
        """Generate several realizations from an initial value."""
        x0 = self._check_initial(initial)

        delta_t = 1.0 * self.t / n
        noise = self._sample_gaussian_noise_paths(n, paths)

        if paths < _VECTORIZE_FROM_PATHS:
            # Few realizations: step through each of them in turn.
            return np.array([self._advance(gns, x0, delta_t) for gns in noise.tolist()])

        # Many realizations: advance all of them together, one step at a time.
        x = np.full(paths, x0)
        realizations = np.empty((paths, n + 1))
        realizations[:, 0] = x
        for k, gn in enumerate(np.ascontiguousarray(noise.T)):
            drift, volatility = self._coefficients(x, ARRAY)
            x = self._confine(x + drift * delta_t + volatility * gn, ARRAY)
            realizations[:, k + 1] = x

        return realizations

    @override
    def _useful_workers(self, n: int, paths: int) -> int:
        # The realizations advance step by step in Python. Threads compete
        # for the interpreter unless each handles many realizations.
        return max(1, min(_MAX_WORKERS, paths // _PATHS_PER_WORKER))

    @override
    def sample(self, n: int) -> npt.NDArray[np.float64]:
        return self._sample_from(n, self._initial)

    def sample_with_initial(self, n: int, initial: float) -> npt.NDArray[np.float64]:
        """Generate a realization starting from an initial value.

        :param n: the number of increments to generate
        :param initial: the initial value of the process
        :returns: the realization
        """
        return self._sample_from(n, initial)

    @override
    def sample_at(self, times: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        raise NotImplementedError(
            f"{type(self).__name__} cannot be sampled at specified times. "
            "Use the sample method with a number of increments."
        )

    @override
    def _sample_paths(self, n: int, paths: int) -> npt.NDArray[np.float64]:
        return self._sample_paths_from(n, paths, self._initial)

    def sample_paths_with_initial(
        self,
        n: int,
        paths: int,
        initial: float,
        workers: int = 1,
    ) -> npt.NDArray[np.float64]:
        """Generate several realizations starting from an initial value.

        :param n: the number of increments of each realization
        :param paths: the number of realizations
        :param initial: the initial value of the process
        :param workers: the number of threads to use
        :returns: a two-dimensional array with one realization per row
        """
        return np.asarray(
            self._run_paths(
                lambda process, count: process._sample_paths_from(n, count, initial),
                n,
                paths,
                workers,
            ),
            dtype=np.float64,
        )
