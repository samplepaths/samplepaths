"""Exact sampling of a square-root (Cox-Ingersoll-Ross) process."""

import math
from collections.abc import Iterable
from itertools import repeat

import numpy as np
import numpy.typing as npt
from numpy.random import Generator


def _advance(
    rng: Generator,
    values: npt.NDArray[np.float64],
    decays: list[float] | float,
    *,
    speed: float,
    mean: float,
    vol: float,
    vectorize_from: int,
) -> None:
    """Fill realizations of a square-root process with a positive volatility.

    :param values: an array of shape ``(steps + 1, paths)`` whose first row
        holds the initial values; the other rows are filled
    :param decays: the exponential of minus the speed times the length of a
        step: a list with one value per step, or a number for equal steps
    """
    n, paths = len(values) - 1, values.shape[1]
    degrees = 4.0 * speed * mean / vol**2

    def constants(decay: float) -> tuple[float, float]:
        scale = vol**2 * (1.0 - decay) / (4.0 * speed)
        return scale, decay / scale

    # For equal steps the two constants are computed once, and the loop over
    # one realization uses them without looking them up at every step.
    equal_steps = isinstance(decays, float)
    pairs: Iterable[tuple[float, float]]
    if isinstance(decays, float):
        scale, factor = constants(decays)
        pairs = repeat((scale, factor), n)
    else:
        pairs = [constants(decay) for decay in decays]

    if degrees == 0:
        # With a mean of zero the chi-square distribution has no degrees of
        # freedom: it is a chi-square distribution with twice a Poisson
        # number of them, and zero when that number is zero.
        if paths == 1:
            # One realization: numbers are faster than arrays of one value.
            value = float(values[0, 0])
            column_values = []
            for scale, factor in pairs:
                value = 2.0 * scale * rng.standard_gamma(rng.poisson(value * factor / 2.0))
                column_values.append(value)
            values[1:, 0] = column_values
            return

        for step, (scale, factor) in enumerate(pairs):
            counts = rng.poisson(values[step] * factor / 2.0)
            values[step + 1] = 2.0 * scale * rng.standard_gamma(counts)
        return

    draw = rng.noncentral_chisquare
    if paths < vectorize_from:
        # Few realizations: numbers are faster than short arrays.
        for column in range(paths):
            value = float(values[0, column])
            column_values = []
            if equal_steps:
                for _ in range(n):
                    value = scale * draw(degrees, value * factor)
                    column_values.append(value)
            else:
                for scale, factor in pairs:
                    value = scale * draw(degrees, value * factor)
                    column_values.append(value)
            values[1:, column] = column_values
        return

    for step, (scale, factor) in enumerate(pairs):
        values[step + 1] = scale * draw(degrees, values[step] * factor)


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

    _advance(rng, values, decay, speed=speed, mean=mean, vol=vol, vectorize_from=vectorize_from)
    return values


def square_root_values(
    rng: Generator,
    steps: npt.NDArray[np.float64],
    paths: int,
    *,
    speed: float,
    mean: float,
    vol: float,
    initial: float,
    vectorize_from: int,
) -> npt.NDArray[np.float64]:
    """Generate realizations of a square-root process over given time steps.

    The same as :py:func:`square_root_paths`, for time steps which need not
    be equal.

    :param rng: the generator to draw from
    :param steps: the lengths of the time steps
    :param paths: the number of realizations
    :param speed: the speed of reversion, positive
    :param mean: the mean, nonnegative
    :param vol: the volatility coefficient, nonnegative
    :param initial: the value at time zero, nonnegative
    :param vectorize_from: the number of realizations from which all of them
        are drawn together at each step, instead of one after the other
    :returns: an array of shape ``(len(steps) + 1, paths)``, stored step by
        step
    """
    decays = [math.exp(-speed * step) for step in steps.tolist()]
    values = np.empty((len(decays) + 1, paths))
    values[0] = initial

    if vol == 0:
        # Without noise the process follows its mean.
        values[1:] = (mean + (initial - mean) * np.cumprod(decays))[:, np.newaxis]
        return values

    _advance(rng, values, decays, speed=speed, mean=mean, vol=vol, vectorize_from=vectorize_from)
    return values
