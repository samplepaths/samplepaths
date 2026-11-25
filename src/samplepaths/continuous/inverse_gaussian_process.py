"""Inverse Gaussian process."""

import inspect
from collections.abc import Callable
from typing import override

import numpy as np
import numpy.typing as npt
from numpy.random import Generator

from samplepaths.utils.validation import check_positive_number

from ..base import BaseTimeProcess


class InverseGaussianProcess(BaseTimeProcess):
    r"""Inverse Gaussian process.

    .. image:: /_static/inverse_gaussian.png
        :scale: 50%

    An inverse Gaussian process has independent increments which follow an
    inverse Gaussian distribution with parameters defined by a monotonically
    increasing function, :math:`\Gamma(t)`. E.g. for increment :math:`[s, t]`:

    :math:`\mathcal{IG}(\Gamma(t) - \Gamma(s), \eta(\Gamma(t) - \Gamma(s))^2)`

    Uses a method for generating inverse Gaussian variates from:

    * Michael, John R., William R. Schucany, and Roy W. Haas. "Generating
      random variates using transformations with multiple roots." The
      American Statistician 30, no. 2 (1976): 88-90.

    :param mean: a callable with one argument :math:`\Gamma(t)` such
        that :math:`\Gamma(t') > \Gamma(t) \forall t' > t`. Default is the
        identity function.
    :param scale: scale factor of the shape parameter of the inverse
        gaussian, or :math:`\eta` from the above equation.
    :param t: the right hand endpoint of the time interval :math:`[0,t]`
        for the process
    :param rng: a custom random number generator
    """

    def __init__(
        self,
        *,
        mean: Callable[[float], float] = lambda x: x,
        scale: float = 1.0,
        t: float = 1.0,
        rng: Generator | None = None,
    ) -> None:
        super().__init__(t=t, rng=rng)

        self.mean = mean
        self.scale = scale

        self.__ms: npt.NDArray[np.float64] | None = None

    def __str__(self) -> str:
        return (
            f"Inverse Gaussian process with mean {self.mean.__name__} "
            f"and scale {self.scale} on interval [0, {self.t}]."
        )

    def __repr__(self) -> str:
        return f"InverseGaussianProcess(mean={self.mean.__name__}, scale={self.scale}, t={self.t})"

    @property
    def mean(self) -> Callable[[float], float]:
        """Mean function."""
        return self.__mean

    @mean.setter
    def mean(self, value: Callable[[float], float]) -> None:
        if not callable(value):
            raise TypeError(f"Mean must be a callable function; got {type(value)}.")

        try:
            num_args = len(inspect.signature(value).parameters)
        except Exception as e:
            raise TypeError("Mean must be an inspectable callable.") from e

        if num_args != 1:
            raise ValueError("Mean must be a function of one argument.")

        self.__mean = value

    @property
    def scale(self) -> float:
        """Scale parameter."""
        return self.__scale

    @scale.setter
    def scale(self, value: float) -> None:
        check_positive_number(value=value, name="Scale")

        self.__scale = value

    def _check_mean(self, left: float, right: float) -> float:
        """Check the validity of the mean function."""
        delta = self.mean(right) - self.mean(left)

        if delta <= 0:
            raise ValueError("Mean must be monotonically increasing.")

        return delta

    def _sample_inverse_gaussian_process(
        self,
        n: int,
    ) -> npt.NDArray[np.float64]:
        """Generate a realization of the inverse Gaussian process.

        Generate an inverse Gaussian process realization with n increments.
        """
        if self.set_times(n):
            self.__ms = np.zeros(n)
            for k in range(n):
                self.__ms[k] = self._check_mean(
                    self.times[k],
                    self.times[k + 1],
                )

        assert self.__ms is not None
        ls = self.scale * self.__ms**2

        gn = self.rng.normal(size=n)
        ys = gn**2

        xs = (
            self.__ms
            + self.__ms**2 * ys / 2 / ls
            - self.__ms / 2 / ys * np.sqrt(4 * self.__ms * ls * ys + self.__ms**2 * ys**2)
        )

        zs = self.rng.uniform(size=n)

        ign = np.where(zs <= self.__ms / (self.__ms + xs), xs, self.__ms**2 / xs)

        return np.concatenate(([0.0], np.cumsum(ign)))

    @override
    def sample(self, n: int) -> npt.NDArray[np.float64]:
        return self._sample_inverse_gaussian_process(n)

    def _sample_inverse_gaussian_process_at(
        self,
        times: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Generate an inverse Gaussian process at specified times."""
        n = len(times) - 1
        if times[0] != 0:
            times = np.concatenate(([0], times))

        ms = np.zeros(n)
        for k in range(n):
            ms[k] = self._check_mean(times[k], times[k + 1])

        ls = self.scale * ms**2

        gn = self.rng.normal(size=n)
        ys = gn**2

        xs = ms + ms**2 * ys / 2 / ls - ms / 2 / ys * np.sqrt(4 * ms * ls * ys + ms**2 * ys**2)

        zs = self.rng.uniform(size=n)

        ig = np.cumsum(np.where(zs <= ms / (ms + xs), xs, ms**2 / xs))
        if times[0] == 0:
            ig = np.insert(ig, 0, [0])

        return ig

    @override
    def sample_at(
        self,
        times: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        return self._sample_inverse_gaussian_process_at(times)
