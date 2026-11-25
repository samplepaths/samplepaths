"""Extended Vasicek process."""

import math
from collections.abc import Callable

import numpy as np
import numpy.typing as npt
from numpy.random import Generator

from .._compat import override
from .._recurrence import solve_linear_recurrence
from .._times import steps_from_zero
from ..utils.validation import check_numeric, check_positive_integer
from .diffusion_process import DiffusionProcess


class ExtendedVasicekProcess(DiffusionProcess):
    r"""Extended Vasicek process.

    A model for future interest rates.

    .. image:: /_static/extended_vasicek_process.png
        :scale: 50%

    The ExtendedVasicek process :math:`X_t` that satisfies the following
    stochastic differential equation with Wiener process :math:`W_t`:

    .. math::

        dX_t = \theta_t (\mu_t - X_t) dt + \sigma_t dW_t

    :py:meth:`sample` and :py:meth:`sample_paths` generate realizations
    using the Euler-Maruyama method. When the parameters are numbers, the
    value after a time step is normal given the value before, and
    :py:meth:`sample_exact`, :py:meth:`sample_paths_exact` and
    :py:meth:`sample_at` draw every value from that exact distribution. Their
    realizations do not depend on the number of steps.

    .. note::

        Since the family of diffusion processes have parameters which
        generalize to functions of ``t``, parameter attributes will be returned
        as callables, even if they are initialized as constants. e.g. a
        ``speed`` parameter of 1 accessed from an instance attribute will return
        a function which accepts a single argument and always returns 1.

    :param speed: the speed of reversion, or :math:`\theta_t` above
    :param mean: the mean of the process, or :math:`\mu_t` above
    :param vol: volatility coefficient of the process, or :math:`\sigma_t`
        above
    :param t: the right hand endpoint of the time interval :math:`[0,t]`
        for the process
    :param rng: a custom random number generator
    """

    def __init__(
        self,
        *,
        speed: Callable[[float], float] | float | int = 1.0,
        mean: Callable[[float], float] | float | int = 0.0,
        vol: Callable[[float], float] | float | int = 1.0,
        t: float = 1.0,
        rng: Generator | None = None,
    ) -> None:
        super().__init__(
            speed=speed,
            mean=mean,
            vol=vol,
            volexp=0.0,
            t=t,
            rng=rng,
        )

    def __str__(self) -> str:
        return (
            f"Extended Vasicek process with speed={self.speed}, "
            f"mean={self.mean}, vol={self.vol} on [0, {self.t}]"
        )

    def __repr__(self) -> str:
        return (
            f"ExtendedVasicekProcess(speed={self.speed}, "
            f"mean={self.mean}, vol={self.vol}, t={self.t})"
        )

    def _exact_parameters(self) -> tuple[float, float, float]:
        """Return the parameters, checked for the exact method."""
        speed, mean, vol = self._constant_parameters()
        if speed is None or mean is None or vol is None:
            raise ValueError(
                "The exact method needs constant parameters; got a function of time. "
                "Use sample() or sample_paths()."
            )
        if speed <= 0 or vol < 0:
            raise ValueError(
                "The exact method needs a positive speed and a nonnegative volatility; "
                f"got speed={speed}, vol={vol}."
            )
        return speed, mean, vol

    def _sample_paths_exact(self, n: int, paths: int, initial: float) -> npt.NDArray[np.float64]:
        """Generate several realizations from the exact distribution."""
        check_numeric(value=initial, name="Initial")
        speed, mean, vol = self._exact_parameters()
        noise = self._sample_gaussian_noise_paths(n, paths)

        # Over a step the value is the mean plus the decayed distance to it
        # plus normal noise: a linear recurrence, solved with array
        # operations. The noise already has the variance of the time step.
        delta_t = self.t / n
        decay = math.exp(-speed * delta_t)
        scale = vol * math.sqrt((1.0 - decay**2) / (2.0 * speed * delta_t))
        inputs = mean * (1.0 - decay) + scale * noise
        return solve_linear_recurrence(inputs, decay, float(initial))

    def sample_exact(self, n: int, initial: float = 1.0) -> npt.NDArray[np.float64]:
        """Generate a realization from the exact distribution of the process.

        The speed, the mean and the volatility must be numbers, with a
        positive speed.

        :param n: the number of increments to generate
        :param initial: the initial value of the process
        :returns: the realization
        :raises ValueError: if a parameter is a function of time or is out
            of range
        """
        check_positive_integer(n=n)
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
        positive speed.

        :param n: the number of increments of each realization
        :param paths: the number of realizations
        :param initial: the initial value of the process
        :param workers: the number of threads to use
        :returns: a two-dimensional array with one realization per row
        :raises ValueError: if a parameter is a function of time or is out
            of range
        """
        check_positive_integer(n=n)
        self._exact_parameters()

        return np.asarray(
            self._run_paths(
                lambda process, count: process._sample_paths_exact(n, count, initial),
                n,
                paths,
                workers,
            ),
            dtype=np.float64,
        )

    @override
    def sample_at(self, times: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Generate a realization at specified times, starting from 1.

        Every value is drawn from its exact distribution given the previous
        one. The parameters must be numbers, with a positive speed.

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
        :param initial: the value of the process at time zero
        :returns: the realization
        :raises NotImplementedError: if a parameter is a function of time
        :raises ValueError: if a parameter is out of range
        """
        check_numeric(value=initial, name="Initial")
        if None in self._constant_parameters():
            raise NotImplementedError(
                "Sampling at specified times needs constant parameters; got a function "
                "of time. Use the sample method with a number of increments."
            )
        speed, mean, vol = self._exact_parameters()
        steps, starts_at_zero = steps_from_zero(times)

        decays = np.exp(-speed * steps)
        noises = (
            vol * np.sqrt((1.0 - decays**2) / (2.0 * speed)) * self.rng.standard_normal(len(steps))
        )

        value = float(initial)
        values = [value]
        for decay, noise in zip(decays.tolist(), noises.tolist(), strict=True):
            value = mean + (value - mean) * decay + noise
            values.append(value)

        return np.array(values if starts_at_zero else values[1:])
