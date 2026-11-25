"""Solution of a linear recurrence with array operations."""

import math
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.signal import lfilter

#: Largest logarithm of a power of the decay within a block of steps.
_LOG_POWER_RANGE = 60.0


def solve_linear_recurrence(
    inputs: npt.NDArray[np.float64],
    decay: float,
    initial: float | npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Solve ``X[k + 1] = decay X[k] + inputs[k]`` along the last axis.

    The Euler method applied to a linear equation gives such a recurrence.
    Its solution is a weighted cumulative sum, which is evaluated with array
    operations: they release the interpreter lock, so several threads can
    each solve their own.

    :param inputs: the inputs, with the steps along the last axis
    :param decay: the factor applied to the previous value
    :param initial: the value before the first step, a number or an array
        which broadcasts to the other axes of ``inputs``
    :returns: an array with one more value than ``inputs`` along the last
        axis, which starts with the initial value
    """
    n = inputs.shape[-1]
    solution = np.empty((*inputs.shape[:-1], n + 1))
    solution[..., 0] = initial

    if decay <= 0:
        # A time step this large makes the Euler method unstable. A compiled
        # filter evaluates the recurrence as it is written.
        state: npt.NDArray[Any] = decay * solution[..., :1]
        solution[..., 1:], _ = lfilter([1.0], [1.0, -decay], inputs, axis=-1, zi=state)
        return solution

    # The powers of the decay are kept within the range of floats by solving
    # the recurrence over blocks of steps.
    block = n if decay == 1 else max(1, int(_LOG_POWER_RANGE / abs(math.log(decay))))
    for start in range(0, n, block):
        stop = min(n, start + block)
        powers = decay ** np.arange(1, stop - start + 1)
        previous = solution[..., start : start + 1]
        solution[..., start + 1 : stop + 1] = powers * (
            previous + np.cumsum(inputs[..., start:stop] / powers, axis=-1)
        )
    return solution
