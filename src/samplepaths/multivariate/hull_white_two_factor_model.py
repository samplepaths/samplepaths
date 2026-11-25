"""Two-factor Hull-White model."""

from collections.abc import Callable

import numpy as np
import numpy.typing as npt
from numpy.random import Generator

from .._compat import override
from .._recurrence import solve_linear_recurrence
from ..utils import ensure_single_arg_constant_function
from ..utils.validation import (
    check_nonnegative_number,
    check_numeric,
    check_numeric_or_single_arg_callable,
    check_positive_number,
)
from .vector_process import VectorProcess


class HullWhiteTwoFactorModel(VectorProcess):
    r"""Two-factor Hull-White model.

    .. image:: /_static/hull_white_two_factor_model.png
        :scale: 50%

    An interest rate :math:`r_t` which reverts to a level that is itself
    random (Hull and White, 1994):

    .. math::

        dr_t &= (\theta_t + u_t - a r_t) dt + \sigma_1 dW^1_t \\
        du_t &= -b u_t dt + \sigma_2 dW^2_t

    with Wiener processes :math:`W^1_t` and :math:`W^2_t` of correlation
    :math:`\rho`. The factor :math:`u_t` starts at zero and moves the level
    to which the rate reverts, which lets rates of different maturities move
    differently. The drift :math:`\theta_t` is a number or a function of
    time.

    A realization has two components, the rate and the factor. Both are
    generated using the Euler-Maruyama method. The equations are linear, so
    all steps are computed with array operations.

    :param speed: the speed of reversion of the rate, or :math:`a` above
    :param drift: the drift, or :math:`\theta_t`: a number or a function of
        the time
    :param vol: the volatility coefficient of the rate, or :math:`\sigma_1`
    :param factor_speed: the speed of reversion of the factor, or :math:`b`
    :param factor_vol: the volatility coefficient of the factor, or
        :math:`\sigma_2`
    :param correlation: the correlation of the two Wiener processes, or
        :math:`\rho`
    :param t: the right hand endpoint of the time interval :math:`[0,t]`
        for the process
    :param rng: a custom random number generator
    """

    components = ("rate", "factor")

    def __init__(
        self,
        *,
        speed: float = 1.0,
        drift: Callable[[float], float] | float = 0.0,
        vol: float = 1.0,
        factor_speed: float = 1.0,
        factor_vol: float = 1.0,
        correlation: float = 0.0,
        t: float = 1.0,
        rng: Generator | None = None,
    ) -> None:
        super().__init__(t=t, rng=rng)

        self.speed = speed
        self.drift = drift
        self.vol = vol
        self.factor_speed = factor_speed
        self.factor_vol = factor_vol
        self.correlation = correlation

    def __str__(self) -> str:
        return (
            f"Two-factor Hull-White model with speed={self.speed}, drift={self.drift}, "
            f"vol={self.vol}, factor_speed={self.factor_speed}, factor_vol={self.factor_vol}, "
            f"correlation={self.correlation} on [0, {self.t}]"
        )

    def __repr__(self) -> str:
        return (
            f"HullWhiteTwoFactorModel(speed={self.speed}, drift={self.drift}, vol={self.vol}, "
            f"factor_speed={self.factor_speed}, factor_vol={self.factor_vol}, "
            f"correlation={self.correlation}, t={self.t})"
        )

    @property
    def speed(self) -> float:
        r"""Speed of reversion of the rate, or :math:`a`."""
        return self.__speed

    @speed.setter
    def speed(self, value: float) -> None:
        check_positive_number(value=value, name="Speed")

        self.__speed = float(value)

    @property
    def drift(self) -> Callable[[float], float]:
        r"""Drift, or :math:`\theta_t`, as a function of the time."""
        return self.__drift

    @drift.setter
    def drift(self, value: Callable[[float], float] | float) -> None:
        check_numeric_or_single_arg_callable(value=value, name="drift")

        self.__drift = ensure_single_arg_constant_function(value=value)
        self.__drift_value = None if callable(value) else float(value)

    @property
    def vol(self) -> float:
        r"""Volatility coefficient of the rate, or :math:`\sigma_1`."""
        return self.__vol

    @vol.setter
    def vol(self, value: float) -> None:
        check_nonnegative_number(value=value, name="Volatility")

        self.__vol = float(value)

    @property
    def factor_speed(self) -> float:
        r"""Speed of reversion of the factor, or :math:`b`."""
        return self.__factor_speed

    @factor_speed.setter
    def factor_speed(self, value: float) -> None:
        check_positive_number(value=value, name="Factor speed")

        self.__factor_speed = float(value)

    @property
    def factor_vol(self) -> float:
        r"""Volatility coefficient of the factor, or :math:`\sigma_2`."""
        return self.__factor_vol

    @factor_vol.setter
    def factor_vol(self, value: float) -> None:
        check_nonnegative_number(value=value, name="Factor volatility")

        self.__factor_vol = float(value)

    @property
    def correlation(self) -> float:
        r"""Correlation of the two Wiener processes, or :math:`\rho`."""
        return self.__correlation

    @correlation.setter
    def correlation(self, value: float) -> None:
        self.__correlation = self._check_correlation(value)

    def _sample_paths_from(self, n: int, paths: int, initial: float) -> npt.NDArray[np.float64]:
        """Generate several realizations from an initial rate."""
        check_numeric(value=initial, name="Initial")
        self._check_paths(n, paths)

        delta_t = self.t / n
        for_rate, for_factor = self._wiener_increments(n, paths, self.correlation)

        if self.__drift_value is not None:
            drifts: npt.NDArray[np.float64] | float = self.__drift_value
        else:
            drifts = np.array([float(self.__drift(time)) for time in self.times[1:].tolist()])

        realizations = np.empty((paths, 2, n + 1))
        factor = solve_linear_recurrence(
            self.factor_vol * for_factor.T, 1.0 - self.factor_speed * delta_t, 0.0
        )
        realizations[:, 1, :] = factor
        realizations[:, 0, :] = solve_linear_recurrence(
            (drifts + factor[:, :-1]) * delta_t + self.vol * for_rate.T,
            1.0 - self.speed * delta_t,
            float(initial),
        )
        return realizations

    @override
    def _sample_paths(self, n: int, paths: int) -> npt.NDArray[np.float64]:
        return self._sample_paths_from(n, paths, 0.0)

    def sample_with_initial(self, n: int, initial: float = 0.0) -> npt.NDArray[np.float64]:
        """Generate a realization starting from an initial rate.

        :param n: the number of increments to generate
        :param initial: the rate at time zero
        :returns: the realization, with the rate in the first row and the
            factor in the second
        """
        realization: npt.NDArray[np.float64] = self._sample_paths_from(n, 1, initial)[0]
        return realization

    def sample_paths_with_initial(
        self,
        n: int,
        paths: int,
        initial: float = 0.0,
        workers: int = 1,
    ) -> npt.NDArray[np.float64]:
        """Generate several realizations starting from an initial rate.

        :param n: the number of increments of each realization
        :param paths: the number of realizations
        :param initial: the rate at time zero
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
