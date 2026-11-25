"""Default random number generator of the package."""

import numpy as np

from .utils.validation import is_integer

_default_rng = np.random.default_rng()

#: The default random number generator
generator = _default_rng


def use_generator(
    *,
    rng: np.random.Generator | None = None,
) -> None:
    """Set the default random number generator.

    Processes created without an ``rng`` use the default generator.

    :param rng: the generator to use as the default. If not provided, the
        original default generator of the package is restored.
    :raises TypeError: if ``rng`` is not a :py:class:`numpy.random.Generator`
    """
    global generator

    if rng is not None and not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be of type np.random.Generator")

    generator = rng or _default_rng


def seed(
    *,
    value: int,
) -> None:
    """Seed the default random number generator.

    A new default generator is created using
    ``numpy.random.default_rng(value)``.

    :param value: the seed
    :raises TypeError: if ``value`` is not an integer
    """
    global generator

    if not is_integer(value):
        raise TypeError("Seed value must be an integer.")

    generator = np.random.default_rng(seed=value)
