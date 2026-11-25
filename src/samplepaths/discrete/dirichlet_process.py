"""Dirichlet process."""

from collections.abc import Callable
from typing import override

import numpy as np
import numpy.typing as npt
from numpy.random import Generator

from ..base import BaseSequenceProcess
from ..utils.validation import check_positive_number


class DirichletProcess(BaseSequenceProcess):
    r"""Dirichlet process.

    .. image:: /_static/dirichlet_process.png
        :scale: 50%

    A Dirichlet process is a stochastic process in which the resulting samples
    can be interpreted as discrete probability distributions.

    For each step :math:`k \geq 1`, draw from the base distribution with
    probability

    .. math::
        \frac{\alpha}{\alpha + k - 1}

    Otherwise draw randomly from the previous steps.

    :param base: a zero argument callable used as the base
        distribution sampler. The default base distribution is Uniform(0, 1).
    :param alpha: a non-negative value used to determine probability of
        drawing a new value from the base distribution
    :param rng: a custom random number generator
    """

    def __init__(
        self,
        base: Callable[[], float] | None = None,
        alpha: float = 1,
        rng: Generator | None = None,
    ) -> None:
        super().__init__(rng=rng)

        if base is None:
            base = self.rng.uniform

        self.base = base
        self.alpha = alpha

    def __str__(self) -> str:
        return f"Dirichlet process with alpha={self.alpha} and base distribution {self.base}"

    def __repr__(self) -> str:
        return f"DirichletProcess(alpha={self.alpha}, base={self.base})"

    @property
    def base(self) -> Callable[[], float]:
        """The base distribution callable for sampling new step values."""
        return self.__base

    @base.setter
    def base(self, value: Callable[[], float]) -> None:
        if not callable(value):
            raise TypeError("base must be callable")

        self.__base = value

    @property
    def alpha(self) -> float:
        """Parameter for determining the probability of sampling new values."""
        return self.__alpha

    @alpha.setter
    def alpha(self, value: float) -> None:
        check_positive_number(value=value, name="alpha")

        self.__alpha = value

    def _sample(self, n: int) -> npt.NDArray[np.float64]:
        """Generate a realization of the Dirichlet process.

        :param n: the number of steps of the Dirichlet process to generate.
        """
        alpha = self.alpha
        base = self.base
        uniform = self.rng.uniform
        integers = self.rng.integers

        sequence: list[float] = []
        for k in range(n):
            if uniform() < alpha / (alpha + k):
                sequence.append(base())
            else:
                # Repeat one of the previous values, chosen uniformly.
                sequence.append(sequence[integers(0, k)])
        return np.array(sequence, dtype=np.float64)

    @override
    def sample(self, n: int) -> npt.NDArray[np.float64]:
        return self._sample(n)
