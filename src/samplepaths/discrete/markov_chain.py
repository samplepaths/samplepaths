"""Finite state Markov chain."""

from bisect import bisect_right
from collections.abc import Iterable
from typing import override

import numpy as np
import numpy.typing as npt
from numpy.random import Generator

from ..base import BaseSequenceProcess
from ..utils.validation import check_positive_integer

DEFAULT_TRANSITION: npt.NDArray[np.float64] = np.array(
    [[0.5, 0.5], [0.5, 0.5]],
    dtype=np.float64,
)

#: Number of realizations from which sample_paths advances all of them
#: together instead of one after the other.
_VECTORIZE_FROM_PATHS = 32


def _cumulative(probabilities: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Compute a normalized cumulative distribution, as Generator.choice does."""
    cumulative = np.cumsum(probabilities, dtype=np.float64)
    cumulative /= cumulative[-1]
    return cumulative


class MarkovChain(BaseSequenceProcess):
    """Finite state Markov chain.

    .. image:: /_static/markov_chain.png
        :scale: 50%

    A Markov Chain which changes between states according to the transition
    matrix.

    :param transition: a square matrix representing the transition
        probabilities between states.
    :param initial: a vector representing the initial state probabilities. If
        not provided, each state has equal initial probability.
    :param rng: a custom random number generator
    """

    def __init__(
        self,
        transition: Iterable[Iterable[float]] | npt.NDArray[np.float64] = DEFAULT_TRANSITION,
        initial: Iterable[float] | npt.NDArray[np.float64] | None = None,
        rng: Generator | None = None,
    ) -> None:
        super().__init__(rng=rng)

        self.transition = transition

        if initial is None:
            initial = [float(1.0 / len(self.transition)) for _ in self.transition]

        self.initial = initial

        self.__num_states = len(self.__initial)

    def __str__(self) -> str:
        return (
            f"Markov chain with transition matrix = {self.transition} "
            f"and initial state probabilities = {self.initial}"
        )

    def __repr__(self) -> str:
        return f"MarkovChain(transition={self.transition}, initial={self.initial})"

    @property
    def transition(self) -> npt.NDArray[np.float64]:
        """Transition probability matrix."""
        return self.__transition

    @transition.setter
    def transition(self, values: Iterable[Iterable[float]] | npt.NDArray[np.float64]) -> None:
        values = np.asarray(values, dtype=np.float64)

        if values.ndim != 2:
            raise ValueError(
                f"Transition matrix must be a two-dimensional array; got {values.ndim} dimensions."
            )

        if values.shape[0] != values.shape[1]:
            raise ValueError(
                "Transition matrix must be a square matrix; "
                f"got {values.shape[0]}x{values.shape[1]}."
            )

        for row in values:
            if not np.isclose(np.sum(row), 1.0):
                raise ValueError(
                    "Transition matrix is not a proper stochastic matrix; "
                    f"row sums to {np.sum(row)}."
                )

        self.__transition = values

    @property
    def initial(self) -> npt.NDArray[np.float64]:
        """Vector of initial state probabilities."""
        return self.__initial

    @initial.setter
    def initial(self, values: Iterable[float] | npt.NDArray[np.float64]) -> None:
        values = np.asarray(values, dtype=np.float64)

        if values.ndim != 1:
            raise ValueError(
                "Initial state probabilities must be a one-dimensional vector; "
                f"got {values.ndim} dimensions."
            )

        if len(values) != len(self.transition):
            raise ValueError(
                "Length of initial state probabilities must match the number of states in the "
                f"transition matrix; got {len(values)} values for {len(self.transition)} states."
            )

        if not np.isclose(np.sum(values), 1.0):
            raise ValueError(f"Initial state probabilities must sum to 1; got {np.sum(values)}.")

        self.__initial = values

    @property
    def num_states(self) -> int:
        """Number of states in the Markov chain."""
        return self.__num_states

    @override
    def sample(self, n: int) -> npt.NDArray[np.int64]:
        check_positive_integer(n=n)

        # Each step inverts the cumulative distribution of the current state
        # at a uniform variate, which is what Generator.choice does, without
        # its overhead per call.
        initial = _cumulative(self.initial).tolist()
        transition = [_cumulative(row).tolist() for row in self.transition]
        uniforms = self.rng.random(n).tolist()

        state = bisect_right(initial, uniforms[0])
        markov_chain = [state]
        for uniform in uniforms[1:]:
            state = bisect_right(transition[state], uniform)
            markov_chain.append(state)

        return np.array(markov_chain, dtype=np.int64)

    @override
    def sample_paths(self, n: int, paths: int) -> npt.NDArray[np.int64]:
        check_positive_integer(n=n)
        check_positive_integer(n=paths, name="Number of paths")

        initial = _cumulative(self.initial)
        transition = np.array([_cumulative(row) for row in self.transition])
        uniforms = self.rng.random((paths, n))

        realizations = np.empty((paths, n), dtype=np.int64)
        if paths < _VECTORIZE_FROM_PATHS:
            # Few realizations: step through each of them in turn.
            initial_list = initial.tolist()
            transition_list = transition.tolist()
            for row, row_uniforms in zip(realizations, uniforms.tolist(), strict=True):
                state = bisect_right(initial_list, row_uniforms[0])
                states = [state]
                for uniform in row_uniforms[1:]:
                    state = bisect_right(transition_list[state], uniform)
                    states.append(state)
                row[:] = states
            return realizations

        # Many realizations: advance all of them together, one step at a time.
        # The next state is the number of cumulative probabilities which do
        # not exceed the uniform variate.
        current = np.searchsorted(initial, uniforms[:, 0], side="right")
        realizations[:, 0] = current
        for k in range(1, n):
            current = np.sum(uniforms[:, k, np.newaxis] >= transition[current], axis=1)
            realizations[:, k] = current

        return realizations
