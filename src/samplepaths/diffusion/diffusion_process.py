"""Generalized diffusion process."""

import math
from collections.abc import Callable

import numpy as np
import numpy.typing as npt
from numpy.random import Generator
from scipy.signal import lfilter

from .._compat import override
from ..noise import GaussianNoise
from ..utils import ensure_single_arg_constant_function
from ..utils.validation import (
    check_numeric,
    check_numeric_or_single_arg_callable,
    check_positive_integer,
)

#: Number of realizations from which sample_paths advances all of them
#: together instead of one after the other.
_VECTORIZE_FROM_PATHS = 32

#: Largest logarithm of a power of the decay used in the linear recurrence.
_LOG_POWER_RANGE = 60.0

#: Limits on the threads used by the diffusions which advance step by step.
_STEPPING_MAX_WORKERS = 4
_STEPPING_PATHS_PER_WORKER = 20_000


def _constant(value: Callable[[float], float] | float | int) -> float | None:
    """Return the value of a constant parameter, or None for a function."""
    return None if callable(value) else float(value)


def _power(base: float, exponent: float) -> float:
    """Raise a float to a power with the semantics of NumPy scalars.

    A negative base with a fractional exponent gives ``nan`` with a warning
    and an overflow gives ``inf``, where Python floats would return a complex
    number or raise an exception.
    """
    if exponent == 0:
        return 1.0
    if exponent == 1:
        return base
    if base > 0.0:
        try:
            return float(base**exponent)
        except OverflowError:
            pass
    return float(np.float64(base) ** exponent)


class DiffusionProcess(GaussianNoise):
    r"""Generalized diffusion process.

    A base process for more specific diffusion processes.

    The process :math:`X_t` that satisfies the following
    stochastic differential equation with Wiener process :math:`W_t`:

    .. math::

        dX_t = \theta_t (\mu_t - X_t) dt + \sigma_t X_t^{\gamma_t} dW_t

    Realizations are generated using the Euler-Maruyama method.

    .. note::

        Since the family of diffusion processes have parameters which
        generalize to functions of ``t``, parameter attributes will be returned
        as callables, even if they are initialized as constants. e.g. a
        ``speed`` parameter of 1 accessed from an instance attribute will
        return a function which accepts a single argument and always returns 1.

    :param speed: the speed of reversion, or :math:`\theta_t` above
    :param mean: the mean of the process, or :math:`\mu_t` above
    :param vol: volatility coefficient of the process, or :math:`\sigma_t`
        above
    :param volexp: volatility exponent of the process, or :math:`\gamma_t`
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
        volexp: Callable[[float], float] | float | int = 0.0,
        t: float = 1.0,
        rng: Generator | None = None,
    ) -> None:
        super().__init__(t=t, rng=rng)

        self.speed = speed
        self.mean = mean
        self.vol = vol
        self.volexp = volexp

    def __str__(self) -> str:
        return (
            f"Diffusion process with speed={self.speed}, mean={self.mean}, "
            f"vol={self.vol}, volexp={self.volexp} on [0, {self.t}]"
        )

    def __repr__(self) -> str:
        return (
            f"DiffusionProcess(speed={self.speed}, mean={self.mean}, "
            f"vol={self.vol}, volexp={self.volexp}, t={self.t})"
        )

    @property
    def speed(self) -> Callable[[float], float]:
        r"""Speed, or :math:`\theta_t`."""
        return self.__speed

    @speed.setter
    def speed(self, value: Callable[[float], float] | float | int) -> None:
        check_numeric_or_single_arg_callable(value=value, name="speed")
        self.__speed = ensure_single_arg_constant_function(value=value)
        self.__speed_value = _constant(value)

    @property
    def mean(self) -> Callable[[float], float]:
        r"""Mean, or :math:`\mu_t`."""
        return self.__mean

    @mean.setter
    def mean(self, value: Callable[[float], float] | float | int) -> None:
        check_numeric_or_single_arg_callable(value=value, name="mean")
        self.__mean = ensure_single_arg_constant_function(value=value)
        self.__mean_value = _constant(value)

    @property
    def vol(self) -> Callable[[float], float]:
        r"""Volatility, or :math:`\sigma_t`."""
        return self.__vol

    @vol.setter
    def vol(self, value: Callable[[float], float] | float | int) -> None:
        check_numeric_or_single_arg_callable(value=value, name="vol")
        self.__vol = ensure_single_arg_constant_function(value=value)
        self.__vol_value = _constant(value)

    @property
    def volexp(self) -> Callable[[float], float]:
        r"""Volatility exponent, or :math:`\gamma_t`."""
        return self.__volexp

    @volexp.setter
    def volexp(
        self,
        value: Callable[[float], float] | float | int,
    ) -> None:
        check_numeric_or_single_arg_callable(value=value, name="volexp")
        self.__volexp = ensure_single_arg_constant_function(value=value)
        self.__volexp_value = _constant(value)

    def _constant_parameters(self) -> tuple[float | None, float | None, float | None]:
        """Return the speed, the mean and the volatility which are numbers.

        A parameter given as a function of time is returned as ``None``.
        """
        return self.__speed_value, self.__mean_value, self.__vol_value

    def _sample(self, n: int, initial: float = 1.0) -> npt.NDArray[np.float64]:
        """Generate a realization of a diffusion process using Euler-Maruyama."""
        check_positive_integer(n=n)
        check_numeric(value=initial, name="Initial")

        delta_t = 1.0 * self.t / n
        # Plain Python floats make the loop several times faster than NumPy
        # scalars, with the same arithmetic.
        gns: list[float] = self._sample_gaussian_noise(n).tolist()

        return np.array(self._euler_maruyama(gns, float(initial), delta_t))

    def _euler_maruyama(self, gns: list[float], x: float, delta_t: float) -> list[float]:
        """Apply the Euler-Maruyama method to one realization of the noise."""
        speed = self.__speed_value
        mean = self.__mean_value
        vol = self.__vol_value
        volexp = self.__volexp_value

        s = [x]

        if speed is not None and mean is not None and vol is not None and volexp is not None:
            # Constant parameters: no function calls inside the loop.
            if volexp == 0:
                for gn in gns:
                    x += speed * (mean - x) * delta_t + vol * gn
                    s.append(x)
            elif volexp == 1:
                for gn in gns:
                    x += speed * (mean - x) * delta_t + vol * x * gn
                    s.append(x)
            else:
                for gn in gns:
                    x += speed * (mean - x) * delta_t + vol * _power(x, volexp) * gn
                    s.append(x)

            return s

        # At least one parameter is a function. The functions are called at
        # every step, in the order speed, mean, vol and volexp.
        speed_function = self.__speed
        mean_function = self.__mean
        vol_function = self.__vol
        volexp_function = self.__volexp

        t = 0.0
        for gn in gns:
            t += delta_t
            a = speed if speed is not None else float(speed_function(t))
            m = mean if mean is not None else float(mean_function(t))
            v = vol if vol is not None else float(vol_function(t))
            g = volexp if volexp is not None else float(volexp_function(t))
            x += a * (m - x) * delta_t + v * _power(x, g) * gn
            s.append(x)

        return s

    @override
    def _useful_workers(self, n: int, paths: int) -> int:
        if self.__speed_value is not None and self.__volexp_value == 0:
            return super()._useful_workers(n, paths)

        # The other diffusions advance step by step in Python. Threads then
        # compete for the interpreter unless each handles many realizations.
        return max(1, min(_STEPPING_MAX_WORKERS, paths // _STEPPING_PATHS_PER_WORKER))

    def _sample_paths_from(self, n: int, paths: int, initial: float) -> npt.NDArray[np.float64]:
        """Generate several realizations of a diffusion process."""
        check_numeric(value=initial, name="Initial")

        delta_t = 1.0 * self.t / n
        noise = self._sample_gaussian_noise_paths(n, paths)

        speed = self.__speed_value
        mean = self.__mean_value
        vol = self.__vol_value
        volexp = self.__volexp_value

        if paths < _VECTORIZE_FROM_PATHS:
            # Few realizations: step through each of them in turn.
            return np.array(
                [self._euler_maruyama(gns, float(initial), delta_t) for gns in noise.tolist()]
            )

        if speed is not None and volexp == 0:
            # A constant speed without a volatility exponent gives a linear
            # recurrence, X[k + 1] = decay X[k] + c[k], whose solution is a
            # weighted cumulative sum. It is evaluated for all realizations
            # and all steps with array operations.
            times = delta_t * np.arange(1, n + 1)
            means = mean if mean is not None else np.array([self.__mean(t) for t in times])
            vols = vol if vol is not None else np.array([self.__vol(t) for t in times])
            decay = 1.0 - speed * delta_t
            inputs = speed * means * delta_t + vols * noise

            realizations = np.empty((paths, n + 1))
            realizations[:, 0] = float(initial)

            if decay > 0:
                # The powers of the decay are kept within the range of floats
                # by solving the recurrence over blocks of steps.
                block = n if decay == 1 else max(1, int(_LOG_POWER_RANGE / abs(math.log(decay))))
                for start in range(0, n, block):
                    stop = min(n, start + block)
                    powers = decay ** np.arange(1, stop - start + 1)
                    previous = realizations[:, start : start + 1]
                    realizations[:, start + 1 : stop + 1] = powers * (
                        previous + np.cumsum(inputs[:, start:stop] / powers, axis=1)
                    )
            else:
                # A time step this large makes the method unstable. A compiled
                # filter evaluates the recurrence as it is written.
                state = np.full((paths, 1), decay * float(initial))
                realizations[:, 1:], _ = lfilter([1.0], [1.0, -decay], inputs, axis=1, zi=state)

            return realizations

        # Many realizations: advance all of them together, one step at a time.
        speed_function = self.__speed
        mean_function = self.__mean
        vol_function = self.__vol
        volexp_function = self.__volexp

        x = np.full(paths, float(initial))
        realizations = np.empty((paths, n + 1))
        realizations[:, 0] = x

        t = 0.0
        for k, gn in enumerate(np.ascontiguousarray(noise.T)):
            t += delta_t
            a = speed if speed is not None else float(speed_function(t))
            m = mean if mean is not None else float(mean_function(t))
            v = vol if vol is not None else float(vol_function(t))
            g = volexp if volexp is not None else float(volexp_function(t))
            if g == 0:
                x = x + a * (m - x) * delta_t + v * gn
            elif g == 1:
                x = x + a * (m - x) * delta_t + v * x * gn
            else:
                x = x + a * (m - x) * delta_t + v * np.power(x, g) * gn
            realizations[:, k + 1] = x

        return realizations

    @override
    def _sample_paths(self, n: int, paths: int) -> npt.NDArray[np.float64]:
        return self._sample_paths_from(n, paths, 1.0)

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

    @override
    def sample(self, n: int) -> npt.NDArray[np.float64]:
        """Generate a realization starting from an initial value of 1.

        :param n: the number of increments to generate
        :returns: the realization
        """
        return self._sample(n, 1.0)

    def sample_with_initial(
        self,
        n: int,
        initial: float,
    ) -> npt.NDArray[np.float64]:
        """Generate a realization starting from an initial value.

        :param n: the number of increments to generate
        :param initial: the initial value of the process
        :returns: the realization
        """
        return self._sample(n, initial)

    @override
    def sample_at(
        self,
        times: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        raise NotImplementedError(
            f"{type(self).__name__} cannot be sampled at specified times. "
            "Use the sample method with a number of increments."
        )
