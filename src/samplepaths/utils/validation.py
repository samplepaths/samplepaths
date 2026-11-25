"""Validation of parameters and arguments."""

from collections.abc import Callable, Sequence
from inspect import signature
from numbers import Integral, Real
from typing import Any

import numpy as np
import numpy.typing as npt


def is_integer(value: object) -> bool:
    """Check whether a value is an integer.

    Python integers and NumPy integer scalars are integers.

    :param value: the value to check
    :returns: ``True`` if ``value`` is an integer
    """
    return isinstance(value, Integral)


def is_number(value: object) -> bool:
    """Check whether a value is a real number.

    Python integers and floats and NumPy integer and floating point scalars
    are real numbers.

    :param value: the value to check
    :returns: ``True`` if ``value`` is a real number
    """
    return isinstance(value, Real)


def check_positive_integer(
    *,
    n: int,
    name: str = "",
) -> None:
    """Ensure that a number is a positive integer.

    :param n: the number to check
    :param name: the name of the number, used in the error message
    :raises TypeError: if ``n`` is not an integer
    :raises ValueError: if ``n`` is not positive
    """
    if not is_integer(n):
        raise TypeError(f"{name} must be an integer; got {type(n).__name__}.")
    if n <= 0:
        raise ValueError(f"{name} must be positive; got {n}.")


def check_nonnegative_integer(
    *,
    n: int,
    name: str = "",
) -> None:
    """Ensure that a number is a nonnegative integer.

    :param n: the number to check
    :param name: the name of the number, used in the error message
    :raises TypeError: if ``n`` is not an integer
    :raises ValueError: if ``n`` is negative
    """
    if not is_integer(n):
        raise TypeError(f"{name} must be an integer; got {type(n).__name__}.")
    if n < 0:
        raise ValueError(f"{name} must be nonnegative; got {n}.")


def check_numeric(
    *,
    value: float | int,
    name: str = "",
) -> None:
    """Ensure that a value is a number.

    :param value: the value to check
    :param name: the name of the value, used in the error message
    :raises TypeError: if ``value`` is not a number
    """
    if not is_number(value):
        raise TypeError(f"{name} value must be a number; got {type(value).__name__}.")


def check_positive_number(
    *,
    value: float,
    name: str = "",
) -> None:
    """Ensure that a value is a positive number.

    :param value: the value to check
    :param name: the name of the value, used in the error message
    :raises TypeError: if ``value`` is not a number
    :raises ValueError: if ``value`` is not positive
    """
    check_numeric(value=value, name=name)
    if value <= 0:
        raise ValueError(f"{name} value must be positive; got {value}.")


def check_nonnegative_number(
    *,
    value: float,
    name: str = "",
) -> None:
    """Ensure that a value is a nonnegative number.

    :param value: the value to check
    :param name: the name of the value, used in the error message
    :raises TypeError: if ``value`` is not a number
    :raises ValueError: if ``value`` is negative
    """
    check_numeric(value=value, name=name)
    if value < 0:
        raise ValueError(f"{name} value must be nonnegative; got {value}.")


def check_increments(
    *,
    times: npt.NDArray[np.float64] | list[float],
) -> npt.NDArray[np.float64]:
    """Ensure that time values are nonnegative and strictly increasing.

    :param times: the time values to check
    :returns: the increments between consecutive time values
    :raises ValueError: if a time value is negative or the time values are
        not strictly increasing
    """
    values = np.asarray(times, dtype=np.float64)
    increments = np.diff(values)

    if np.any(values < 0):
        raise ValueError("Times must be nonnegative.")

    if np.any(increments <= 0):
        raise ValueError("Times must be strictly increasing.")

    return increments


def times_to_increments(
    *,
    times: npt.NDArray[np.float64] | list[float],
) -> npt.NDArray[np.float64]:
    """Convert time values to the increments between them.

    :param times: nonnegative, strictly increasing time values
    :returns: the increments between consecutive time values
    :raises ValueError: if a time value is negative or the time values are
        not strictly increasing
    """
    return check_increments(times=times)


def check_numeric_or_single_arg_callable(
    *,
    value: Callable[[Any], float | int] | float | int,
    name: str = "",
) -> None:
    """Ensure that a value is a number or a single argument callable.

    :param value: the value to check
    :param name: the name of the value, used in the error message
    :raises TypeError: if ``value`` is neither a number nor a callable
    :raises ValueError: if ``value`` is a callable which does not take
        exactly one argument
    """
    if callable(value) and len(signature(value).parameters) != 1:
        raise ValueError(
            f"{name} callable must have a single argument; got {len(signature(value).parameters)}."
        )
    if not is_number(value) and not callable(value):
        raise TypeError(
            f"{name} must be numeric or a single argument callable; got {type(value).__name__}.",
        )


def check_coefficients(
    *,
    value: object,
    name: str = "",
) -> tuple[float, ...]:
    """Ensure that a value is a number or a sequence of numbers.

    :param value: the value to check
    :param name: the name of the value, used in the error message
    :returns: the coefficients as a tuple of floats; a single number gives a
        tuple of one coefficient
    :raises TypeError: if ``value`` is not a number or a sequence of numbers
    """
    if is_number(value):
        return (float(value),)  # type: ignore[arg-type]

    if isinstance(value, str) or not isinstance(value, Sequence | np.ndarray):
        raise TypeError(
            f"{name} must be a number or a sequence of numbers; got {type(value).__name__}."
        )

    coefficients = tuple(value)
    for coefficient in coefficients:
        if not is_number(coefficient):
            raise TypeError(f"{name} must hold numbers; got {type(coefficient).__name__}.")

    return tuple(float(coefficient) for coefficient in coefficients)
