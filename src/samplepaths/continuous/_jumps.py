"""Helper functions for the processes with jumps."""

from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt
from numpy.random import Generator

#: A function which draws jump sizes: it takes a generator and the number of
#: jumps and returns that many sizes.
JumpSizes = Callable[[Generator, int], npt.NDArray[np.float64]]


def standard_normal_jumps(rng: Generator, size: int) -> npt.NDArray[np.float64]:
    """Draw jump sizes from the standard normal distribution.

    :param rng: the generator to draw from
    :param size: the number of jumps
    :returns: the jump sizes
    """
    return rng.standard_normal(size)


def compound_increments(
    rng: Generator,
    expected: float | npt.NDArray[np.float64],
    shape: tuple[int, ...],
    jumps: JumpSizes,
) -> npt.NDArray[np.float64]:
    """Draw the increments of a compound Poisson process.

    Every increment is the sum of a Poisson number of jumps.

    :param rng: the generator to draw from
    :param expected: the expected number of jumps of each increment, a number
        or an array which broadcasts to ``shape``
    :param shape: the shape of the result
    :param jumps: draws the jump sizes
    :returns: the increments
    :raises ValueError: if ``jumps`` does not return one size per jump
    """
    counts = rng.poisson(expected, size=shape)
    total = int(counts.sum())
    if total == 0:
        return np.zeros(shape)

    sizes: npt.NDArray[Any] = np.asarray(jumps(rng, total), dtype=np.float64)
    if sizes.shape != (total,):
        raise ValueError(
            f"The jump size function must return {total} values for {total} jumps; "
            f"got an array of shape {sizes.shape}."
        )

    # Add up the sizes of the jumps which fall in the same increment.
    cells = np.repeat(np.arange(counts.size), counts.ravel())
    increments: npt.NDArray[np.float64] = np.asarray(
        np.bincount(cells, weights=sizes, minlength=counts.size),
        dtype=np.float64,
    ).reshape(shape)
    return increments
