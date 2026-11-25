"""Brownian motion."""

from numpy.random import Generator

from ..utils.validation import check_numeric, check_positive_number
from .brownian_base import BrownianBase


class BrownianMotion(BrownianBase):
    """Brownian motion.

    .. image:: /_static/brownian_motion.png
        :scale: 50%

    A standard Brownian motion (discretely sampled) has independent and
    identically distributed Gaussian increments with variance equal to
    increment length. Non-standard Brownian motion includes a linear drift
    parameter and scale factor.

    :param drift: rate of change of the expected value
    :param scale: scale factor of the Gaussian process
    :param t: the right hand endpoint of the time interval :math:`[0,t]`
        for the process
    :param rng: a custom random number generator
    """

    def __init__(
        self,
        *,
        drift: float = 0.0,
        scale: float = 1.0,
        t: float = 1.0,
        rng: Generator | None = None,
    ) -> None:
        super().__init__(t=t, rng=rng)

        self.drift = drift
        self.scale = scale

    def __str__(self) -> str:
        if self.drift == 0 and self.scale == 1:
            return f"Standard Brownian motion on interval [0, {self.t}]."

        return (
            f"Brownian motion with drift {self.drift} and "
            f"scale {self.scale} on interval [0, {self.t}]."
        )

    def __repr__(self) -> str:
        return f"BrownianMotion(drift={self.drift}, scale={self.scale}, t={self.t})"

    @property
    def drift(self) -> float:
        """Drift parameter."""
        return self._drift

    @drift.setter
    def drift(self, value: float) -> None:
        check_numeric(value=value, name="Drift")

        self._drift = value

    @property
    def scale(self) -> float:
        """Scale parameter."""
        return self._scale

    @scale.setter
    def scale(self, value: float) -> None:
        check_positive_number(value=value, name="Scale")

        self._scale = value
