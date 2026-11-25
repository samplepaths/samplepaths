"""Wiener process, or standard Brownian motion."""

from numpy.random import Generator

from .brownian_base import BrownianBase


class WienerProcess(BrownianBase):
    """Wiener process, or standard Brownian motion.

    .. image:: /_static/wiener_process.png
        :scale: 50%

    :param t: the right hand endpoint of the time interval :math:`[0,t]`
        for the process
    :param rng: a custom random number generator
    """

    def __init__(
        self,
        *,
        t: float = 1.0,
        rng: Generator | None = None,
    ) -> None:
        super().__init__(t=t, rng=rng)

    def __str__(self) -> str:
        return f"Wiener process on [0, {self.t}]"

    def __repr__(self) -> str:
        return f"WienerProcess(t={self.t})"
