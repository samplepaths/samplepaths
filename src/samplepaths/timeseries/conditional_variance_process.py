"""Base class of the time series models with a changing variance."""

from abc import abstractmethod
from collections import deque
from collections.abc import Sequence
from types import SimpleNamespace
from typing import Any

import numpy as np
import numpy.typing as npt
from numpy.random import Generator

from .._compat import override
from .._functions import ARRAY, SCALAR
from ..base import BaseSequenceProcess
from ..utils.validation import check_numeric, check_positive_integer

#: Number of realizations from which sample_paths advances all of them
#: together instead of one after the other.
_VECTORIZE_FROM_PATHS = 32

#: Smallest variance of a value, which keeps the series defined.
_SMALLEST_VARIANCE = 1e-12

_NOT_SAMPLED = "No realization has been generated yet; call sample() first."


class ConditionalVarianceProcess(BaseSequenceProcess):
    r"""Base class of the time series models with a changing variance.

    A series of values with a zero mean whose variance depends on the past,

    .. math::

        X_t = \sigma_t z_t, \quad z_t \sim N(0, 1)

    where the variance :math:`\sigma_t^2` is a function of the previous
    values and of their variances. Calm and agitated periods alternate, as
    in the returns of financial assets.

    A subclass defines that function. A realization starts from the
    stationary state of the model: the variances before the first value are
    at their long-run level, and the values before it are drawn at that
    level.

    After a call to :py:meth:`sample` or :py:meth:`sample_paths`, the
    attribute ``volatility`` holds the standard deviations :math:`\sigma_t`
    of the values.

    :param omega: the constant term of the variance
    :param rng: a custom random number generator
    """

    _parallel_paths = True

    def __init__(
        self,
        *,
        omega: float,
        rng: Generator | None = None,
    ) -> None:
        super().__init__(rng=rng)

        self.__ready = False
        self.__parameters: dict[str, Any] = {}
        self.__volatility: npt.NDArray[np.float64] | None = None
        self.omega = omega

    def _finish_init(self) -> None:
        """Check the parameters together, once all of them are set."""
        self.__ready = True
        self._check_parameters()

    def _get(self, name: str) -> Any:
        """Return the value of a parameter."""
        return self.__parameters[name]

    def _set(self, name: str, value: Any) -> None:
        """Set a parameter and check all of them together.

        The parameters are only valid together, so a change which makes the
        model invalid is undone before the error is raised.
        """
        missing = object()
        previous = self.__parameters.get(name, missing)
        self.__parameters[name] = value
        if not self.__ready:
            return

        try:
            self._check_parameters()
        except ValueError:
            if previous is missing:
                del self.__parameters[name]
            else:
                self.__parameters[name] = previous
            raise

    @property
    def omega(self) -> float:
        r"""Constant term of the variance, or :math:`\omega`."""
        omega: float = self._get("omega")
        return omega

    @omega.setter
    def omega(self, value: float) -> None:
        check_numeric(value=value, name="Omega")

        self._set("omega", float(value))

    @property
    def volatility(self) -> npt.NDArray[np.float64]:
        r"""Standard deviations :math:`\sigma_t` of the most recent realization.

        An array of the shape of the result of :py:meth:`sample` or
        :py:meth:`sample_paths`, whichever was called last.

        :raises RuntimeError: if no realization has been generated yet
        """
        if self.__volatility is None:
            raise RuntimeError(_NOT_SAMPLED)
        return self.__volatility

    @abstractmethod
    def _check_parameters(self) -> None:
        """Ensure that the parameters define a stationary model.

        :raises ValueError: if they do not
        """

    @property
    @abstractmethod
    def _lags(self) -> int:
        """Number of previous values on which the variance depends."""

    @abstractmethod
    def _stationary_state(self) -> float:
        """Return the long-run level of the state of the variance."""

    @abstractmethod
    def _next_state(
        self,
        values: Sequence[Any],
        innovations: Sequence[Any],
        states: Sequence[Any],
        functions: SimpleNamespace,
    ) -> Any:
        r"""Return the state of the variance of the next value.

        The sequences hold the previous values, most recent first, as numbers
        or as arrays with one entry per realization.

        :param values: the previous values :math:`X_{t-1}, X_{t-2}, \dots`
        :param innovations: the previous :math:`z_{t-1}, z_{t-2}, \dots`
        :param states: the previous states of the variance
        :param functions: the functions for the type of the entries
        :returns: the state of the variance of :math:`X_t`
        """

    def _variance(self, state: Any, functions: SimpleNamespace) -> Any:
        """Return the variance for a state. The state is the variance by default."""
        return state

    def _advance(
        self,
        presample: Any,
        innovations: Any,
        functions: SimpleNamespace,
    ) -> tuple[list[Any], list[Any]]:
        """Generate values and their standard deviations from innovations.

        :param presample: the innovations before the first value, most recent
            first, one per lag
        :param innovations: the innovations of the values, in order
        :param functions: the functions for the type of the innovations
        :returns: the values and their standard deviations
        """
        lags = self._lags
        state = self._stationary_state()
        level = functions.sqrt(self._variance(state, functions))

        past_innovations = deque(presample, maxlen=lags)
        past_values = deque((level * z for z in presample), maxlen=lags)
        past_states = deque((state for _ in range(lags)), maxlen=lags)

        next_state = self._next_state
        variance = self._variance

        values = []
        deviations = []
        for z in innovations:
            state = next_state(past_values, past_innovations, past_states, functions)
            deviation = functions.sqrt(
                functions.maximum(variance(state, functions), _SMALLEST_VARIANCE)
            )
            value = deviation * z

            past_values.appendleft(value)
            past_innovations.appendleft(z)
            past_states.appendleft(state)
            values.append(value)
            deviations.append(deviation)

        return values, deviations

    @override
    def sample(self, n: int) -> npt.NDArray[np.float64]:
        check_positive_integer(n=n)

        lags = self._lags
        innovations: list[float] = self.rng.standard_normal(lags + n).tolist()
        values, deviations = self._advance(innovations[:lags], innovations[lags:], SCALAR)

        self.__volatility = np.array(deviations)
        return np.array(values)

    def _innovations(self, n: int, paths: int) -> npt.NDArray[np.float64]:
        """Generate the standard normal variates of several realizations.

        Releases the interpreter lock, so it can run on several threads.

        :returns: an array with one realization per row, stored step by step,
            which starts with the variates before the first value
        """
        innovations: npt.NDArray[np.float64] = self.rng.standard_normal((self._lags + n, paths)).T
        return innovations

    def _paths_from(self, innovations: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Generate one realization from each row of innovations."""
        lags = self._lags

        if len(innovations) < _VECTORIZE_FROM_PATHS:
            # Few realizations: step through each of them in turn.
            rows = [self._advance(row[:lags], row[lags:], SCALAR) for row in innovations.tolist()]
            self.__volatility = np.array([deviations for _, deviations in rows])
            return np.array([values for values, _ in rows])

        # Many realizations: advance all of them together, one step at a time.
        columns = np.ascontiguousarray(innovations.T)
        values, deviations = self._advance(list(columns[:lags]), columns[lags:], ARRAY)

        self.__volatility = np.array(deviations).T
        return np.array(values).T

    @override
    def _sample_paths(self, n: int, paths: int) -> npt.NDArray[np.float64]:
        check_positive_integer(n=n)
        check_positive_integer(n=paths, name="Number of paths")

        return self._paths_from(self._innovations(n, paths))

    @override
    def sample_paths(self, n: int, paths: int, workers: int = 1) -> npt.NDArray[np.float64]:
        check_positive_integer(n=n)

        # The threads generate the random numbers, which releases the
        # interpreter lock. The values then advance on one thread: a step is
        # a few short calls to numpy, which threads would only compete for.
        innovations = self._run_paths(
            lambda process, count: process._innovations(n, count),
            n,
            paths,
            workers,
        )
        return self._paths_from(innovations)
