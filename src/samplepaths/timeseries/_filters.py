"""Helper functions for the linear time series models."""

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
from scipy.signal import lfilter


def lag_polynomial(
    coefficients: Sequence[float], *, sign: float, step: int = 1
) -> npt.NDArray[np.float64]:
    """Return the polynomial ``1 + sign * (c1 B^step + c2 B^(2 step) + ...)``.

    :param coefficients: the coefficients of the lags
    :param sign: the sign of the coefficients in the polynomial: -1 for an
        autoregressive polynomial, +1 for a moving average polynomial
    :param step: the lag between two coefficients, more than one for a
        seasonal polynomial
    :returns: the coefficients of the polynomial, by increasing power of the
        lag operator ``B``
    """
    polynomial = np.zeros(len(coefficients) * step + 1)
    polynomial[0] = 1.0
    polynomial[step::step] = sign * np.asarray(coefficients, dtype=np.float64)
    return polynomial


def differencing_polynomial(order: int, step: int = 1) -> npt.NDArray[np.float64]:
    """Return the polynomial ``(1 - B^step)^order``.

    :param order: the number of times the series is differenced
    :param step: the lag of the differences, more than one for seasonal
        differences
    :returns: the coefficients of the polynomial, by increasing power of the
        lag operator ``B``
    """
    polynomial = np.ones(1)
    for _ in range(order):
        polynomial = np.convolve(polynomial, lag_polynomial([1.0], sign=-1.0, step=step))
    return polynomial


def filter_noise(
    noise: npt.NDArray[np.float64],
    autoregressive: npt.NDArray[np.float64],
    moving_average: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Turn white noise into a series with given lag polynomials.

    Solves ``autoregressive(B) X = moving_average(B) noise`` along the last
    axis, with a series and a noise equal to zero before the first value.

    :param noise: the white noise, one series per row or a single series
    :param autoregressive: the autoregressive polynomial
    :param moving_average: the moving average polynomial
    :returns: the series
    """
    series: npt.NDArray[np.float64] = np.asarray(
        lfilter(moving_average, autoregressive, noise, axis=-1),
        dtype=np.float64,
    )
    return series


def apply_moving_average(
    noise: npt.NDArray[np.float64],
    moving_average: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Apply a moving average polynomial to white noise along the first axis.

    Uses whole-array operations only, which release the interpreter lock, so
    that several threads can each work on their own noise.

    :param noise: the white noise, one series per column, equal to zero
        before the first row
    :param moving_average: the moving average polynomial, starting with 1
    :returns: the filtered noise, which is the given array when the
        polynomial is 1
    """
    lags = np.flatnonzero(moving_average[1:]) + 1
    lags = lags[lags < len(noise)]
    if len(lags) == 0:
        return noise

    series = noise.copy()
    for lag in lags.tolist():
        series[lag:] += moving_average[lag] * noise[:-lag]
    return series


def apply_autoregression(
    series: npt.NDArray[np.float64],
    autoregressive: npt.NDArray[np.float64],
) -> None:
    """Solve ``autoregressive(B) X = series`` in place, one step at a time.

    Every step updates all series together, so the cost of the Python loop
    is shared by them. Worth it for many series and few coefficients.

    :param series: the right-hand side, one series per column and contiguous
        by rows, replaced by the solution
    :param autoregressive: the autoregressive polynomial, starting with 1
    """
    lags = (np.flatnonzero(autoregressive[1:]) + 1).tolist()
    terms = [(lag, -float(autoregressive[lag])) for lag in lags]
    if not terms:
        return

    scaled = np.empty(series.shape[1:])
    multiply = np.multiply
    for step in range(1, len(series)):
        row = series[step]
        for lag, coefficient in terms:
            if lag > step:
                break
            multiply(series[step - lag], coefficient, out=scaled)
            row += scaled
