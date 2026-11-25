"""Fractional Gaussian noise process."""

from functools import lru_cache
from typing import Literal

import numpy as np
import numpy.typing as npt
from numpy.random import Generator

from .._compat import override
from ..base import BaseTimeProcess
from ..utils.validation import check_positive_integer


def _fgn_auto_covariance(
    hurst: float,
    n: int,
) -> npt.NDArray[np.float64]:
    """Autocovariance function for fGn."""
    ns_2h = np.arange(n + 1) ** (2 * hurst)
    return np.insert(
        (ns_2h[:-2] - 2 * ns_2h[1:-1] + ns_2h[2:]) / 2,
        0,
        1,
    )


def _fgn_dh_sqrt_eigen_vals(
    hurst: float,
    n: int,
) -> npt.NDArray[np.float64]:
    """Square-roots of normalized circulant matrix eigenvalues for fGn."""
    return np.fft.irfft(_fgn_auto_covariance(hurst, n))[:n] ** (1 / 2)


#: Largest number of increments for which the coefficients of Hosking's method
#: are kept between realizations. They take 4 n squared bytes: 17 MB at this
#: limit.
_HOSKING_CACHE_MAX_INCREMENTS = 2048


class FractionalGaussianNoise(BaseTimeProcess):
    """Fractional Gaussian noise process.

    .. image:: /_static/fractional_gaussian_noise.png
        :scale: 50%

    Generate sequences of fractional Gaussian noise.

    Hosking's method:

    * Hosking, Jonathan RM. "Modeling persistence in hydrological time series
      using fractional differencing." Water resources research 20, no. 12
      (1984): 1898-1908.

    Davies Harte method:

    * Davies, Robert B., and D. S. Harte. "Tests for Hurst effect." Biometrika
      74, no. 1 (1987): 95-101.

    :param hurst: The Hurst parameter value in :math:`(0,1)`.
    :param t: the right hand endpoint of the time interval :math:`[0,t]`
        for the process
    :param rng: a custom random number generator
    """

    _parallel_paths = True

    def __init__(
        self,
        *,
        hurst: float = 0.5,
        t: float = 1.0,
        rng: Generator | None = None,
    ):
        super().__init__(t=t, rng=rng)

        self.hurst = hurst

        self.__hosking_key: tuple[float, int] | None = None
        self.__hosking_coefficients: tuple[
            list[npt.NDArray[np.float64]], npt.NDArray[np.float64]
        ] = ([], np.zeros(0))
        self.__auto_covariance = lru_cache(1)(_fgn_auto_covariance)
        self.__dh_sqrt_eigen_vals = lru_cache(1)(_fgn_dh_sqrt_eigen_vals)

    def __str__(self) -> str:
        return f"Fractional Gaussian noise with Hurst {self.hurst} on [0, {self.t}]."

    def __repr__(self) -> str:
        return f"FractionalGaussianNoise(hurst={self.hurst}, t={self.t})"

    @property
    def hurst(self) -> float:
        """Hurst parameter."""
        return self.__hurst

    @hurst.setter
    def hurst(self, value: float) -> None:
        if not isinstance(value, (float, np.floating)):
            raise TypeError("Hurst value must be a number on interval (0,1).")

        if value <= 0 or value >= 1:
            raise ValueError("Hurst value must be in interval (0,1).")

        self.__hurst = value

    def _davies_harte(self, n: int) -> npt.NDArray[np.float64]:
        """Generate a fractional Gaussian noise using davies-harte method.

        Uses Davies and Harte method (exact method) from:
        Davies, Robert B., and D. S. Harte. "Tests for Hurst effect."
        Biometrika 74, no. 1 (1987): 95-101.
        """
        check_positive_integer(n=n, name="Number of increments")
        self.set_times(n)

        # For scaling to interval [0, T]
        increment = self.t / n
        scale = increment**self.hurst

        # If H = 0.5 then just generate a standard Brownian motion, otherwise
        # proceed with the Davies Harte method
        if self.hurst == 0.5:
            return self.rng.normal(scale=scale, size=n)

        # Generate some more fGns to use power-of-two FFTs for speed.
        m = 2 ** (n - 2).bit_length() + 1
        sqrt_eigen_vals = self.__dh_sqrt_eigen_vals(self.hurst, m)

        # irfft results will be normalized by (2(m-1))**(3/2) but we only
        # want to normalize by 2(m-1)**(1/2).
        scale *= 2 ** (1 / 2) * (m - 1)

        w = self.rng.normal(scale=scale, size=2 * m).view(complex)
        w[0] = w[0].real * 2 ** (1 / 2)
        w[-1] = w[-1].real * 2 ** (1 / 2)

        # Resulting z is fft of sequence w.
        return np.fft.irfft(sqrt_eigen_vals * w)[:n]

    def _hosking(self, n: int) -> npt.NDArray[np.float64]:
        """Generate fractional Gaussian noise using Hosking's method.

        Method of generation is Hosking's method (exact method) from his paper:
        Hosking, J. R. (1984). Modeling persistence in hydrological time series
        using fractional differencing. Water resources research, 20(12),
        1898-1908.

        Hosking's method generates a fractional Gaussian noise (fGn)
        realization. The cumulative sum of this realization gives a fBm.
        """
        check_positive_integer(n=n, name="Number of increments")
        self.set_times(n)

        # For scaling to interval [0, T]
        increment = self.t / n
        scale = increment**self.hurst

        gn = self.rng.normal(0.0, 1.0, n)

        # If H = 0.5 then just generate a standard Brownian motion, otherwise
        # proceed with Hosking's method
        if self.hurst == 0.5:
            fgn = gn
        else:
            coefficients, deviations = self._hosking_coefficients(n)

            # Generates fgn realization with n increments of size 1. The
            # first increment comes from the stationary distribution, and
            # each of the others is predicted from the previous ones,
            # fgn[i - 1], ..., fgn[0], plus an innovation.
            fgn = np.zeros(n)
            fgn[0] = gn[0]
            for i in range(1, n):
                fgn[i] = coefficients[i] @ fgn[i - 1 :: -1][:i] + deviations[i] * gn[i]

        # Scale to interval [0, T]
        fgn *= scale

        return fgn

    def _hosking_coefficients(
        self,
        n: int,
    ) -> tuple[list[npt.NDArray[np.float64]], npt.NDArray[np.float64]]:
        """Run the Durbin-Levinson recursion of Hosking's method.

        Entry i of the first result holds the coefficients which predict
        increment i from the i previous ones, and entry i of the second the
        standard deviation of its innovation. They depend on the Hurst
        parameter and on the number of increments only, so they are kept for
        the next realization unless they would take too much memory.
        """
        key = (self.hurst, n)
        if self.__hosking_key == key:
            return self.__hosking_coefficients

        phi = np.zeros(n)
        psi = np.zeros(n)
        cov = self.__auto_covariance(self.hurst, n)

        coefficients = [np.zeros(0)]
        deviations = np.ones(n)
        v = 1
        for i in range(1, n):
            psi[: i - 1] = phi[: i - 1]
            # cov[i - 1], ..., cov[1] and psi[i - 2], ..., psi[0]
            reflection = (cov[i] - psi[: i - 1] @ cov[i - 1 : 0 : -1]) / v
            phi[: i - 1] = psi[: i - 1] - reflection * psi[i - 2 :: -1][: i - 1]
            phi[i - 1] = reflection
            v *= 1 - reflection * reflection
            coefficients.append(phi[:i].copy())
            deviations[i] = np.sqrt(v)

        result = (coefficients, deviations)
        if n <= _HOSKING_CACHE_MAX_INCREMENTS:
            self.__hosking_key = key
            self.__hosking_coefficients = result

        return result

    def _sample_fractional_gaussian_noise(
        self,
        n: int,
        algorithm: Literal["davies-harte", "hosking"],
    ) -> npt.NDArray[np.float64]:
        """Generate a realization of fractional Gaussian noise."""
        if algorithm == "davies-harte":
            return self._davies_harte(n)

        if algorithm == "hosking":
            return self._hosking(n)

        raise ValueError("Algorithm must be 'davies-harte' or 'hosking'.")

    def sample_davies_harte(self, n: int) -> npt.NDArray[np.float64]:
        """Generate a realization using the Davies-Harte method.

        :param n: the number of increments to generate
        :returns: the realization
        """
        return self._sample_fractional_gaussian_noise(n, "davies-harte")

    def sample_hosking(self, n: int) -> npt.NDArray[np.float64]:
        """Generate a realization using Hosking's method.

        :param n: the number of increments to generate
        :returns: the realization
        """
        return self._sample_fractional_gaussian_noise(n, "hosking")

    @override
    def sample(
        self,
        n: int,
    ) -> npt.NDArray[np.float64]:
        return self.sample_davies_harte(n)

    def _davies_harte_paths(self, n: int, paths: int) -> npt.NDArray[np.float64]:
        """Generate several realizations with the Davies-Harte method."""
        check_positive_integer(n=n, name="Number of increments")
        check_positive_integer(n=paths, name="Number of paths")
        self.set_times(n)

        # For scaling to interval [0, T]
        increment = self.t / n
        scale = increment**self.hurst

        if self.hurst == 0.5:
            return self.rng.normal(scale=scale, size=(paths, n))

        m = 2 ** (n - 2).bit_length() + 1
        sqrt_eigen_vals = self.__dh_sqrt_eigen_vals(self.hurst, m)
        scale *= 2 ** (1 / 2) * (m - 1)

        w = self.rng.normal(scale=scale, size=(paths, 2 * m)).view(complex)
        w[:, 0] = w[:, 0].real * 2 ** (1 / 2)
        w[:, -1] = w[:, -1].real * 2 ** (1 / 2)

        realizations: npt.NDArray[np.float64] = np.fft.irfft(sqrt_eigen_vals * w, axis=1)[:, :n]
        return realizations

    @override
    def _sample_paths(self, n: int, paths: int) -> npt.NDArray[np.float64]:
        return self._davies_harte_paths(n, paths)
