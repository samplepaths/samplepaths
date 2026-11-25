"""Helper functions."""

from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt


def generate_times(
    *,
    end: float,
    n: int,
) -> npt.NDArray[np.float64]:
    """Generate equally spaced time values on [0, end].

    :param end: the right hand endpoint of the time interval
    :param n: the number of increments
    :returns: ``n + 1`` time values from zero to ``end``
    """
    return np.linspace(0, end, n + 1)


def single_arg_constant_function(
    *,
    value: float | int,
) -> Callable[[Any], float | int]:
    """Create a single argument function which returns a constant value.

    :param value: the value to return
    :returns: a function of one argument returning ``value``
    """
    return lambda x: value


def ensure_single_arg_constant_function(
    *,
    value: Callable[[Any], float | int] | float | int,
) -> Callable[[Any], float | int]:
    """Convert a value to a constant function unless it is callable.

    :param value: a number or a single argument callable
    :returns: ``value`` if it is callable, otherwise a function of one
        argument returning ``value``
    """
    if not callable(value):
        return single_arg_constant_function(value=value)
    return value
