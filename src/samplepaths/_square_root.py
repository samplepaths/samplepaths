"""Exact sampling of a square-root (Cox-Ingersoll-Ross) process."""

import math

import numpy as np
import numpy.typing as npt
from numpy.random import Generator


def square_root_paths(
    rng: Generator,
    n: int,
    paths: int,
    *,
    speed: float,
    mean: float,
    vol: float,
    initial: float,
    t: float,
    vectorize_from: int,
) -> npt.NDArray[np.float64]:
    r"""Generate realizations of a square-root process from its exact law.

    The process satisfies
    :math:`dX_t = \theta (\mu - X_t) dt + \sigma \sqrt{X_t} dW_t`. Over a
    time step its value is a multiple of a noncentral chi-square variate
    whose noncentrality depends on the previous value, so the realizations
    are exact for any number of steps and never negative.

    :param rng: the generator to draw from
    :param n: the number of increments
    :param paths: the number of realizations
    :param speed: the speed of reversion :math:`\theta`, positive
    :param mean: the mean :math:`\mu`, nonnegative
    :param vol: the volatility coefficient :math:`\sigma`, nonnegative
    :param initial: the value at time zero, nonnegative
    :param t: the right hand endpoint of the time interval
    :param vectorize_from: the number of realizations from which all of them
        are drawn together at each step, instead of one after the other
    :returns: an array of shape ``(n + 1, paths)``, stored step by step
    """
    decay = math.exp(-speed * t / n)
    values = np.empty((n + 1, paths))
    values[0] = initial

    if vol == 0:
        # Without noise the process follows its mean.
        values[1:] = (mean + (initial - mean) * decay ** np.arange(1, n + 1))[:, np.newaxis]
        return values

    scale = vol**2 * (1.0 - decay) / (4.0 * speed)
    degrees = 4.0 * speed * mean / vol**2
    factor = decay / scale

    if degrees == 0:
        # With a mean of zero the chi-square distribution has no degrees of
        # freedom: it is a chi-square distribution with twice a Poisson
        # number of them, and zero when that number is zero.
        for step in range(n):
            counts = rng.poisson(values[step] * factor / 2.0)
            values[step + 1] = 2.0 * scale * rng.standard_gamma(counts)
        return values

    draw = rng.noncentral_chisquare
    if paths < vectorize_from:
        # Few realizations: numbers are faster than short arrays.
        for column in range(paths):
            value = initial
            column_values = []
            for _ in range(n):
                value = scale * draw(degrees, value * factor)
                column_values.append(value)
            values[1:, column] = column_values
        return values

    for step in range(n):
        values[step + 1] = scale * draw(degrees, values[step] * factor)
    return values
