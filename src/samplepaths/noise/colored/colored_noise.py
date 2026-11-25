"""Colored noise processes."""

import numpy as np
import numpy.typing as npt
from numpy.random import Generator

from ..._compat import override
from ...base import BaseTimeProcess
from ...utils.validation import check_numeric, check_positive_integer


class ColoredNoise(BaseTimeProcess):
    r"""Colored noise processes.

    .. image:: /_static/colored_noise.png
        :scale: 50%

    Also referred to as power law noise, colored noise refers to noise
    processes with power law spectral density. That is, their spectral density
    per unit bandwidth is proportional to :math:`(1/f)^\beta`, where
    :math:`f` is frequency with exponent :math:`\beta`.

    Uses the algorithm from:

    * Timmer, J., and M. Koenig. "On generating power law noise."
      Astronomy and Astrophysics 300 (1995): 707.

    A realization is scaled so that its variance is close to :math:`t/n`, the
    variance of Gaussian noise with the same number of increments :math:`n`
    on the same interval, whatever the exponent. Its mean is zero.

    :param beta: the power law exponent for the spectral density, with 0
        being white noise, 1 being pink noise, 2 being red noise (Brownian
        noise), -1 being blue noise, -2 being violet noise. Default is 0
        (white noise).
    :param t: the right hand endpoint of the time interval :math:`[0,t]`
        for the process
    :param rng: a custom random number generator
    """

    def __init__(
        self,
        *,
        beta: float = 0,
        t: float = 1.0,
        rng: Generator | None = None,
    ) -> None:
        super().__init__(t=t, rng=rng)

        self.beta = beta

        self.__key: tuple[int, float, float] | None = None
        self.__half: int | None = None
        self.__frequencies: npt.NDArray[np.float64] | None = None
        self.__scale: npt.NDArray[np.float64] | None = None

    def __str__(self) -> str:
        return f"Colored noise generator with exponent {self.beta} on interval [0, {self.t}]"

    def __repr__(self) -> str:
        return f"ColoredNoise(beta={self.beta}, t={self.t})"

    @property
    def beta(self) -> float:
        """Power law exponent."""
        return self.__beta

    @beta.setter
    def beta(self, value: float) -> None:
        check_numeric(value=value, name="beta")

        self.__beta = value

    def _update_spectrum(self, n: int) -> None:
        """Compute the frequencies and their scale for n increments.

        They depend on n, beta and t only, so they are reused by consecutive
        realizations.
        """
        if n < 2:
            raise ValueError(f"Number of increments must be at least 2; got {n}.")

        key = (n, self.beta, self.t)
        if self.__key != key:
            self.__key = key

            frequencies = np.asarray(np.fft.fftfreq(n, self.t), dtype=np.float64)
            self.__half = (n + 1) // 2
            self.__frequencies = frequencies
            self.__scale = np.sqrt(0.5 * (1 / frequencies[1 : self.__half]) ** self.beta)

    def _sample_colored_noise(self, n: int) -> npt.NDArray[np.float64]:
        """Generate colored noise increments at specified times from zero."""
        check_positive_integer(n=n, name="Number of increments")
        self.set_times(n)
        self._update_spectrum(n)

        assert self.__frequencies is not None
        assert self.__half is not None

        gn_real = self.rng.normal(size=self.__half - 1)
        gn_imag = self.rng.normal(size=self.__half - 1)
        fft = self.__scale * (gn_real + 1j * gn_imag)

        f: npt.NDArray[np.float64]
        if n % 2 == 0:
            f = np.concatenate(
                (
                    [0],
                    fft,
                    [
                        np.sqrt(0.5 * (1 / -self.__frequencies[self.__half]) ** self.beta)
                        * self.rng.normal()
                    ],
                    np.conj(fft)[::-1],
                )
            )
        else:
            f = np.concatenate(([0], fft, np.conj(fft)[::-1]))

        realization: npt.NDArray[np.float64] = np.sqrt(self.t) * np.fft.ifft(f).real / np.std(f)
        return realization

    @override
    def sample(self, n: int) -> npt.NDArray[np.float64]:
        return self._sample_colored_noise(n)

    @override
    def sample_paths(self, n: int, paths: int) -> npt.NDArray[np.float64]:
        check_positive_integer(n=n, name="Number of increments")
        check_positive_integer(n=paths, name="Number of paths")
        self.set_times(n)
        self._update_spectrum(n)

        assert self.__frequencies is not None
        assert self.__half is not None

        gn_real = self.rng.normal(size=(paths, self.__half - 1))
        gn_imag = self.rng.normal(size=(paths, self.__half - 1))
        fft = self.__scale * (gn_real + 1j * gn_imag)

        zeros = np.zeros((paths, 1))
        mirrored = np.conj(fft)[:, ::-1]
        if n % 2 == 0:
            middle = np.sqrt(
                0.5 * (1 / -self.__frequencies[self.__half]) ** self.beta
            ) * self.rng.normal(size=(paths, 1))
            f = np.concatenate((zeros, fft, middle, mirrored), axis=1)
        else:
            f = np.concatenate((zeros, fft, mirrored), axis=1)

        realizations: npt.NDArray[np.float64] = (
            np.sqrt(self.t) * np.fft.ifft(f, axis=1).real / np.std(f, axis=1, keepdims=True)
        )
        return realizations
