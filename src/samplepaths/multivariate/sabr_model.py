"""SABR model."""

import numpy as np
import numpy.typing as npt
from numpy.random import Generator

from .._compat import override
from ..utils.validation import check_nonnegative_number, check_positive_number
from .vector_process import VectorProcess

#: Number of realizations from which the forwards of all realizations advance
#: together at each step, instead of one realization after the other.
_VECTORIZE_FROM_PATHS = 32

#: Smallest number of realizations worth a thread when the forward advances
#: step by step: with fewer, the threads compete for the interpreter.
_PATHS_PER_WORKER = 5000


class SABRModel(VectorProcess):
    r"""SABR model.

    .. image:: /_static/sabr_model.png
        :scale: 50%

    A forward price or rate :math:`F_t` with a random volatility
    :math:`\sigma_t` (Hagan, Kumar, Lesniewski and Woodward, 2002):

    .. math::

        dF_t &= \sigma_t F_t^\beta dW^F_t \\
        d\sigma_t &= \nu \sigma_t dW^\sigma_t

    with Wiener processes :math:`W^F_t` and :math:`W^\sigma_t` of
    correlation :math:`\rho`. The exponent :math:`\beta` goes from 0, a
    normal model in which the forward can become negative, to 1, a lognormal
    model.

    A realization has two components, the forward and the volatility. The
    volatility is exact. The forward is exact given the volatility for
    :math:`\beta = 0`, uses the Euler method on its logarithm for
    :math:`\beta = 1`, and the Euler method otherwise, where a forward which
    reaches zero stays there.

    :param volvol: the volatility of the volatility, or :math:`\nu` above
    :param beta: the exponent :math:`\beta` in :math:`[0,1]`
    :param correlation: the correlation of the two Wiener processes, or
        :math:`\rho`
    :param initial_volatility: the volatility at time zero, often called
        :math:`\alpha`
    :param t: the right hand endpoint of the time interval :math:`[0,t]`
        for the process
    :param rng: a custom random number generator
    """

    components = ("forward", "volatility")

    def __init__(
        self,
        *,
        volvol: float = 0.4,
        beta: float = 0.5,
        correlation: float = -0.3,
        initial_volatility: float = 0.2,
        t: float = 1.0,
        rng: Generator | None = None,
    ) -> None:
        super().__init__(t=t, rng=rng)

        self.volvol = volvol
        self.beta = beta
        self.correlation = correlation
        self.initial_volatility = initial_volatility

    def __str__(self) -> str:
        return (
            f"SABR model with volvol={self.volvol}, beta={self.beta}, "
            f"correlation={self.correlation} on [0, {self.t}]"
        )

    def __repr__(self) -> str:
        return (
            f"SABRModel(volvol={self.volvol}, beta={self.beta}, correlation={self.correlation}, "
            f"initial_volatility={self.initial_volatility}, t={self.t})"
        )

    @property
    def volvol(self) -> float:
        r"""Volatility of the volatility, or :math:`\nu`."""
        return self.__volvol

    @volvol.setter
    def volvol(self, value: float) -> None:
        check_nonnegative_number(value=value, name="Volatility of the volatility")

        self.__volvol = float(value)

    @property
    def beta(self) -> float:
        r"""Exponent of the forward in its volatility, or :math:`\beta`."""
        return self.__beta

    @beta.setter
    def beta(self, value: float) -> None:
        check_nonnegative_number(value=value, name="Beta")
        if value > 1:
            raise ValueError(f"Beta value must be at most 1; got {value}.")

        self.__beta = float(value)

    @property
    def correlation(self) -> float:
        r"""Correlation of the two Wiener processes, or :math:`\rho`."""
        return self.__correlation

    @correlation.setter
    def correlation(self, value: float) -> None:
        self.__correlation = self._check_correlation(value)

    @property
    def initial_volatility(self) -> float:
        """Volatility at time zero."""
        return self.__initial_volatility

    @initial_volatility.setter
    def initial_volatility(self, value: float) -> None:
        check_positive_number(value=value, name="Initial volatility")

        self.__initial_volatility = float(value)

    @override
    def _useful_workers(self, n: int, paths: int) -> int:
        workers = super()._useful_workers(n, paths)
        if self.beta in (0.0, 1.0):
            # No step is taken in Python, so every thread is useful.
            return workers
        return max(1, min(workers, paths // _PATHS_PER_WORKER))

    def _forwards(
        self,
        initial: float,
        volatilities: npt.NDArray[np.float64],
        increments: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Apply the Euler method to the forward, for an exponent in (0, 1).

        :param initial: the forward at time zero
        :param volatilities: the volatility at the start of each step
        :param increments: the increments of the Wiener process of the forward
        :returns: the forwards after each step, in an array of the shape of
            the arguments, ``(n, paths)``
        """
        beta = self.beta
        forwards = np.empty_like(increments)

        if increments.shape[1] < _VECTORIZE_FROM_PATHS:
            # Few realizations: numbers are faster than short arrays.
            columns = zip(volatilities.T.tolist(), increments.T.tolist(), strict=True)
            for column, (sigmas, changes) in enumerate(columns):
                forward = initial
                values = []
                for sigma, change in zip(sigmas, changes, strict=True):
                    forward = max(forward + sigma * forward**beta * change, 0.0)
                    values.append(forward)
                forwards[:, column] = values
            return forwards

        current = np.full(increments.shape[1], initial)
        for step, (sigma, change) in enumerate(zip(volatilities, increments, strict=True)):
            current = np.maximum(current + sigma * current**beta * change, 0.0)
            forwards[step] = current
        return forwards

    def _sample_paths_from(self, n: int, paths: int, initial: float) -> npt.NDArray[np.float64]:
        """Generate several realizations from an initial forward."""
        if self.beta == 0:
            initial = float(initial)
        else:
            check_positive_number(value=initial, name="Initial")
        self._check_paths(n, paths)

        delta_t = self.t / n
        for_volatility, for_forward = self._wiener_increments(n, paths, self.correlation)

        # The volatility is a geometric Brownian motion without drift.
        logarithms = np.zeros((n + 1, paths))
        np.cumsum(
            self.volvol * for_volatility - self.volvol**2 * delta_t / 2.0,
            axis=0,
            out=logarithms[1:],
        )
        volatilities = self.initial_volatility * np.exp(logarithms)
        before = volatilities[:-1]

        if self.beta == 0:
            forwards = initial + np.cumsum(before * for_forward, axis=0)
        elif self.beta == 1:
            forwards = initial * np.exp(
                np.cumsum(before * for_forward - before**2 * delta_t / 2.0, axis=0)
            )
        else:
            forwards = self._forwards(float(initial), before, for_forward)

        realizations = np.empty((paths, 2, n + 1))
        realizations[:, 0, 0] = initial
        realizations[:, 0, 1:] = forwards.T
        realizations[:, 1, :] = volatilities.T
        return realizations

    @override
    def _sample_paths(self, n: int, paths: int) -> npt.NDArray[np.float64]:
        return self._sample_paths_from(n, paths, 1.0)

    def sample_with_initial(self, n: int, initial: float = 1.0) -> npt.NDArray[np.float64]:
        """Generate a realization starting from an initial forward.

        :param n: the number of increments to generate
        :param initial: the forward at time zero
        :returns: the realization, with the forward in the first row and the
            volatility in the second
        """
        realization: npt.NDArray[np.float64] = self._sample_paths_from(n, 1, initial)[0]
        return realization

    def sample_paths_with_initial(
        self,
        n: int,
        paths: int,
        initial: float = 1.0,
        workers: int = 1,
    ) -> npt.NDArray[np.float64]:
        """Generate several realizations starting from an initial forward.

        :param n: the number of increments of each realization
        :param paths: the number of realizations
        :param initial: the forward at time zero
        :param workers: the number of threads to use
        :returns: a three-dimensional array of shape ``(paths, 2, n + 1)``
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
