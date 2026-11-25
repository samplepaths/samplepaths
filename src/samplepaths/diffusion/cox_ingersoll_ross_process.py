"""Cox-Ingersoll-Ross process."""

import copy

import numpy as np
import numpy.typing as npt
from numpy.random import Generator

from .._compat import override
from .._square_root import square_root_paths, square_root_values
from .._times import steps_from_zero
from ..base import BaseProcess
from ..utils.validation import check_nonnegative_number, check_positive_integer
from .diffusion_process import DiffusionProcess

#: Number of realizations from which the exact method draws the values of
#: all realizations together at each step, instead of one after the other.
_VECTORIZE_FROM_PATHS = 8

#: Smallest number of realizations worth a thread for the exact method: with
#: fewer, the threads compete for the interpreter.
_PATHS_PER_WORKER = 5000

#: Largest number of threads of the exact method.
_MAX_WORKERS = 6


class CoxIngersollRossProcess(DiffusionProcess):
    r"""Cox-Ingersoll-Ross process.

    .. image:: /_static/cox_ingersoll_ross_process.png
        :scale: 50%

    A model for instantaneous interest rate.

    The process :math:`X_t` that satisfies the following stochastic
    differential equation with Wiener process :math:`W_t`:

    .. math::

        dX_t = \theta (\mu - X_t) dt + \sigma \sqrt{X_t} dW_t

    :py:meth:`sample` and :py:meth:`sample_paths` generate realizations
    using the Euler-Maruyama method. :py:meth:`sample_exact`,
    :py:meth:`sample_paths_exact` and :py:meth:`sample_at` draw every value from its exact
    distribution given the previous one, a noncentral chi-square
    distribution. Their realizations are never negative and do not depend on
    the number of steps: a single step gives the exact distribution of the
    final value. A step costs about twice as much as a step of the
    Euler-Maruyama method.

    .. note::

        Since the family of diffusion processes have parameters which
        generalize to functions of ``t``, parameter attributes will be returned
        as callables, even if they are initialized as constants. e.g. a
        ``speed`` parameter of 1 accessed from an instance attribute will return
        a function which accepts a single argument and always returns 1.

    :param speed: the speed of reversion, or :math:`\theta` above
    :param mean: the mean of the process, or :math:`\mu` above
    :param vol: volatility coefficient of the process, or :math:`\sigma`
        above
    :param t: the right hand endpoint of the time interval :math:`[0,t]`
        for the process
    :param rng: a custom random number generator
    """

    #: Whether this copy of the process is generating exact realizations.
    _exact = False

    def __init__(
        self,
        *,
        speed: float = 1.0,
        mean: float = 0.0,
        vol: float = 1.0,
        t: float = 1.0,
        rng: Generator | None = None,
    ) -> None:
        super().__init__(
            speed=speed,
            mean=mean,
            vol=vol,
            volexp=0.5,
            t=t,
            rng=rng,
        )

    def __str__(self) -> str:
        return (
            f"Cox-Ingersoll-Ross process with speed={self.speed}, mean={self.mean}, "
            f"vol={self.vol} on [0, {self.t}]"
        )

    def __repr__(self) -> str:
        return (
            f"CoxIngersollRossProcess(speed={self.speed}, mean={self.mean}, "
            f"vol={self.vol}, t={self.t})"
        )

    def _exact_parameters(self) -> tuple[float, float, float]:
        """Return the parameters, checked for the exact method."""
        speed, mean, vol = self._constant_parameters()
        if speed is None or mean is None or vol is None:
            raise ValueError(
                "The exact method needs constant parameters; got a function of time. "
                "Use sample() or sample_paths()."
            )
        if speed <= 0 or mean < 0 or vol < 0:
            raise ValueError(
                "The exact method needs a positive speed and a nonnegative mean and "
                f"volatility; got speed={speed}, mean={mean}, vol={vol}."
            )
        return speed, mean, vol

    def _sample_paths_exact(self, n: int, paths: int, initial: float) -> npt.NDArray[np.float64]:
        """Generate several realizations from the exact distribution."""
        check_positive_integer(n=n)
        check_positive_integer(n=paths, name="Number of paths")
        check_nonnegative_number(value=initial, name="Initial")
        speed, mean, vol = self._exact_parameters()
        self.set_times(n)

        values = square_root_paths(
            self.rng,
            n,
            paths,
            speed=speed,
            mean=mean,
            vol=vol,
            initial=float(initial),
            t=self.t,
            vectorize_from=_VECTORIZE_FROM_PATHS,
        )
        return np.ascontiguousarray(values.T)

    def sample_exact(self, n: int, initial: float = 1.0) -> npt.NDArray[np.float64]:
        """Generate a realization from the exact distribution of the process.

        The speed, the mean and the volatility must be numbers, with a
        positive speed and a nonnegative mean and volatility.

        :param n: the number of increments to generate
        :param initial: the initial value of the process, nonnegative
        :returns: the realization
        :raises ValueError: if a parameter is a function of time or is out
            of range
        """
        realization: npt.NDArray[np.float64] = self._sample_paths_exact(n, 1, initial)[0]
        return realization

    def sample_paths_exact(
        self,
        n: int,
        paths: int,
        initial: float = 1.0,
        workers: int = 1,
    ) -> npt.NDArray[np.float64]:
        """Generate several realizations from the exact distribution.

        The speed, the mean and the volatility must be numbers, with a
        positive speed and a nonnegative mean and volatility.

        :param n: the number of increments of each realization
        :param paths: the number of realizations
        :param initial: the initial value of the process, nonnegative
        :param workers: the number of threads to use; at most six are used,
            and one for every 5000 realizations
        :returns: a two-dimensional array with one realization per row
        :raises ValueError: if a parameter is a function of time or is out
            of range
        """
        check_positive_integer(n=n)
        self._exact_parameters()

        # The limit on the threads of the Euler-Maruyama method does not
        # apply: a step of the exact method spends its time drawing random
        # numbers, which releases the interpreter lock.
        process = copy.copy(self)
        process._exact = True
        realizations = np.asarray(
            process._run_paths(
                lambda clone, count: clone._sample_paths_exact(n, count, initial),
                n,
                paths,
                workers,
            ),
            dtype=np.float64,
        )
        self.set_times(n)
        return realizations

    @override
    def sample_at(self, times: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Generate a realization at specified times, starting from 1.

        Every value is drawn from its exact distribution given the previous
        one. The speed, the mean and the volatility must be numbers, with a
        positive speed and a nonnegative mean and volatility.

        :param times: a vector of increasing time values at which to generate
            the realization
        :returns: the realization
        :raises NotImplementedError: if a parameter is a function of time
        :raises ValueError: if a parameter is out of range
        """
        return self.sample_at_with_initial(times, 1.0)

    def sample_at_with_initial(
        self,
        times: npt.ArrayLike,
        initial: float = 1.0,
    ) -> npt.NDArray[np.float64]:
        """Generate a realization at specified times from an initial value.

        :param times: a vector of increasing time values at which to generate
            the realization
        :param initial: the value of the process at time zero, nonnegative
        :returns: the realization
        :raises NotImplementedError: if a parameter is a function of time
        :raises ValueError: if a parameter is out of range
        """
        check_nonnegative_number(value=initial, name="Initial")
        if None in self._constant_parameters():
            raise NotImplementedError(
                "Sampling at specified times needs constant parameters; got a function "
                "of time. Use the sample method with a number of increments."
            )
        speed, mean, vol = self._exact_parameters()
        steps, starts_at_zero = steps_from_zero(times)

        values = square_root_values(
            self.rng,
            steps,
            1,
            speed=speed,
            mean=mean,
            vol=vol,
            initial=float(initial),
            vectorize_from=_VECTORIZE_FROM_PATHS,
        )[:, 0]
        return values if starts_at_zero else values[1:]

    @override
    def _useful_workers(self, n: int, paths: int) -> int:
        if not self._exact:
            return super()._useful_workers(n, paths)
        workers = BaseProcess._useful_workers(self, n, paths)
        return max(1, min(workers, _MAX_WORKERS, paths // _PATHS_PER_WORKER))
