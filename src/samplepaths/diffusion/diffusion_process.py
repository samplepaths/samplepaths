"""Generalized diffusion process."""

from collections.abc import Callable
from typing import override

import numpy as np
import numpy.typing as npt
from numpy.random import Generator

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
            g = volexp if volexp is not None else float(volexp_function(x))
            x += a * (m - x) * delta_t + v * _power(x, g) * gn
            s.append(x)

        return s

    def _sample_paths(self, n: int, paths: int, initial: float) -> npt.NDArray[np.float64]:
        """Generate several realizations of a diffusion process."""
        check_numeric(value=initial, name="Initial")

        delta_t = 1.0 * self.t / n
        noise = self._sample_gaussian_noise_paths(n, paths)

        speed = self.__speed_value
        mean = self.__mean_value
        vol = self.__vol_value
        volexp = self.__volexp_value

        if paths < _VECTORIZE_FROM_PATHS or volexp is None:
            # Few realizations, or a volatility exponent which is a function
            # of the value of each realization: step through each of them in
            # turn.
            return np.array(
                [self._euler_maruyama(gns, float(initial), delta_t) for gns in noise.tolist()]
            )

        # Many realizations: advance all of them together, one step at a time.
        speed_function = self.__speed
        mean_function = self.__mean
        vol_function = self.__vol

        x = np.full(paths, float(initial))
        realizations = np.empty((paths, n + 1))
        realizations[:, 0] = x

        t = 0.0
        for k, gn in enumerate(np.ascontiguousarray(noise.T)):
            t += delta_t
            a = speed if speed is not None else float(speed_function(t))
            m = mean if mean is not None else float(mean_function(t))
            v = vol if vol is not None else float(vol_function(t))
            if volexp == 0:
                x = x + a * (m - x) * delta_t + v * gn
            elif volexp == 1:
                x = x + a * (m - x) * delta_t + v * x * gn
            else:
                x = x + a * (m - x) * delta_t + v * np.power(x, volexp) * gn
            realizations[:, k + 1] = x

        return realizations

    @override
    def sample_paths(self, n: int, paths: int) -> npt.NDArray[np.float64]:
        return self._sample_paths(n, paths, 1.0)

    def sample_paths_with_initial(
        self,
        n: int,
        paths: int,
        initial: float,
    ) -> npt.NDArray[np.float64]:
        """Generate several realizations starting from an initial value.

        :param n: the number of increments of each realization
        :param paths: the number of realizations
        :param initial: the initial value of the process
        :returns: a two-dimensional array with one realization per row
        """
        return self._sample_paths(n, paths, initial)

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
