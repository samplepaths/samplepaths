"""Base class of the diffusions driven by a fractional Brownian motion."""

import numpy as np
import numpy.typing as npt
from numpy.random import Generator

from .._compat import override
from ..noise._driver import follow
from ..noise.fractional_gaussian_noise import FractionalGaussianNoise
from ..utils.validation import check_positive_integer
from .euler_maruyama_process import EulerMaruyamaProcess


class FractionalEulerProcess(EulerMaruyamaProcess):
    r"""Base class of the diffusions driven by a fractional Brownian motion.

    A process :math:`X_t` which satisfies the stochastic differential equation

    .. math::

        dX_t = a(X_t) dt + b(X_t) dB^H_t

    with fractional Brownian motion :math:`B^H_t` of Hurst parameter
    :math:`H`. With :math:`H > 1/2` the increments of the motion are
    positively correlated and the process has a long memory; with
    :math:`H < 1/2` they are negatively correlated and the process is rough.
    With :math:`H = 1/2` the motion is a Wiener process.

    Realizations are generated using the Euler method with exact increments
    of the fractional Brownian motion, and every value is confined to the
    domain of the process. The method is known to converge for
    :math:`H > 1/2`.

    :param hurst: the Hurst parameter :math:`H` in :math:`(0,1)`
    :param t: the right hand endpoint of the time interval :math:`[0,t]`
        for the process
    :param rng: a custom random number generator
    """

    def __init__(
        self,
        *,
        hurst: float = 0.7,
        t: float = 1.0,
        rng: Generator | None = None,
    ) -> None:
        super().__init__(t=t, rng=rng)

        self.__noise = FractionalGaussianNoise(hurst=hurst, t=self.t, rng=self.rng)

    @property
    def hurst(self) -> float:
        """Hurst parameter."""
        return self.__noise.hurst

    @hurst.setter
    def hurst(self, value: float) -> None:
        self.__noise.hurst = value

    def _driver(self) -> FractionalGaussianNoise:
        """Return the fractional Gaussian noise which drives the process."""
        self.__noise = follow(self.__noise, self)
        return self.__noise

    @override
    def _sample_gaussian_noise(self, n: int) -> npt.NDArray[np.float64]:
        check_positive_integer(n=n, name="Number of increments")
        self.set_times(n)

        return self._driver().sample(n)

    @override
    def _sample_gaussian_noise_paths(self, n: int, paths: int) -> npt.NDArray[np.float64]:
        check_positive_integer(n=n, name="Number of increments")
        check_positive_integer(n=paths, name="Number of paths")
        self.set_times(n)

        return self._driver()._sample_paths(n, paths)
