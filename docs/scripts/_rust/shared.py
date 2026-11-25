"""What the comparisons with stochastic-rs share."""

from collections.abc import Callable

import numpy as np
import numpy.typing as npt

Array = npt.NDArray[np.float64]

#: Adds a comparison: the name of the process, what is compared, the values
#: of stochastic-rs and the values of samplepaths.
Add = Callable[[str, str, Array, Array], None]

#: Number of increments of the realizations on [0, 1].
STEPS = 64

#: stochastic-rs counts the points of a realization, not its increments.
POINTS = STEPS + 1

#: Number of values of the time series. stochastic-rs starts a series from
#: zeros and samplepaths from the long-run distribution, so the series are
#: long enough for the start to be forgotten.
SERIES = 300

#: What most comparisons look at.
FINAL = "final value"


def final_values(process: object, count: int) -> Array:
    """Return the final values of realizations of a stochastic-rs process."""
    return np.asarray(process.sample_par(count))[:, -1]  # type: ignore[attr-defined]


def final_components(process: object, count: int) -> tuple[Array, Array]:
    """Return the final values of both components of a stochastic-rs process."""
    first, second = process.sample_par(count)  # type: ignore[attr-defined]
    return np.asarray(first)[:, -1], np.asarray(second)[:, -1]


def looped(sample: Callable[[], float], count: int) -> Array:
    """Collect a summary of realizations generated one by one."""
    return np.array([sample() for _ in range(count)])
