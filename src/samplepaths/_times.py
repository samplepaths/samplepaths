"""Helper for sampling at times chosen by the caller."""

import numpy as np
import numpy.typing as npt

from .utils.validation import check_increments


def steps_from_zero(
    times: npt.ArrayLike,
) -> tuple[npt.NDArray[np.float64], bool]:
    """Return the time steps which lead from time zero through given times.

    A process starts at time zero. When the first of the times is not zero,
    a first step leads from zero to it, and the process has no value to
    return for time zero.

    :param times: nonnegative, strictly increasing time values
    :returns: the time steps, and whether the first time is zero
    :raises ValueError: if the times are not a nonempty vector of
        nonnegative, strictly increasing values
    """
    values = np.asarray(times, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("Times must be a nonempty one-dimensional vector.")

    starts_at_zero = bool(values[0] == 0)
    if not starts_at_zero:
        values = np.concatenate(([0.0], values))
    return check_increments(times=values), starts_at_zero
