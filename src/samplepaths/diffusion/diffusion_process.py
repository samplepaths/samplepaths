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

    @property
    def mean(self) -> Callable[[float], float]:
        r"""Mean, or :math:`\mu_t`."""
        return self.__mean

    @mean.setter
    def mean(self, value: Callable[[float], float] | float | int) -> None:
        check_numeric_or_single_arg_callable(value=value, name="mean")
        self.__mean = ensure_single_arg_constant_function(value=value)

    @property
    def vol(self) -> Callable[[float], float]:
        r"""Volatility, or :math:`\sigma_t`."""
        return self.__vol

    @vol.setter
    def vol(self, value: Callable[[float], float] | float | int) -> None:
        check_numeric_or_single_arg_callable(value=value, name="vol")
        self.__vol = ensure_single_arg_constant_function(value=value)

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

    def _sample(self, n: int, initial: float = 1.0) -> npt.NDArray[np.float64]:
        """Generate a realization of a diffusion process using Euler-Maruyama."""
        check_positive_integer(n=n)
        check_numeric(value=initial, name="Initial")

        delta_t = 1.0 * self.t / n
        gns = self._sample_gaussian_noise(n)

        s = [initial]
        t = 0.0
        for k in range(n):
            t += delta_t
            initial += (
                self.__speed(t) * (self.__mean(t) - initial) * delta_t
                + self.__vol(t) * initial ** self.__volexp(initial) * gns[k]
            )
            s.append(initial)

        return np.array(s)

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
