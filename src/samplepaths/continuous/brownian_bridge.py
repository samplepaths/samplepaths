"""Brownian bridge."""

import numpy as np
import numpy.typing as npt
from numpy.random import Generator

from .._compat import override
from ..utils.validation import check_numeric
from .brownian_base import BrownianBase


class BrownianBridge(BrownianBase):
    """Brownian bridge.

    .. image:: /_static/brownian_bridge.png
        :scale: 50%

    A Brownian bridge is a Brownian motion with a conditional value on the
    right endpoint of the process.

    :param b: the right endpoint value of the Brownian bridge at time t
    :param t: the right hand endpoint of the time interval :math:`[0,t]`
        for the process
    :param rng: a custom random number generator
    """

    def __init__(
        self,
        *,
        b: float = 0.0,
        t: float = 1.0,
        rng: Generator | None = None,
    ) -> None:
        super().__init__(t=t, rng=rng)

        self.b = b

    def __str__(self) -> str:
        return f"Brownian bridge from 0 to {self.b} on [0, {self.t}]"

    def __repr__(self) -> str:
        return f"BrownianBridge(b={self.b}, t={self.t})"

    @property
    def b(self) -> float:
        """Right endpoint value."""
        return self._b

    @b.setter
    def b(self, value: float) -> None:
        check_numeric(value=value, name="Right endpoint value")

        self._b = value

    def _sample_brownian_bridge(self, n: int) -> npt.NDArray[np.float64]:
        """Generate a realization of a Brownian bridge."""
        self.set_times(n)
        bm = self._sample_brownian_motion(n)

        ret = bm + self.times * (self.b - bm[-1]) / self.t

        assert isinstance(ret, np.ndarray)

        return ret

    def _sample_brownian_bridge_at(
        self,
        times: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Generate a realization of a Brownian bridge at times."""
        bm = self._sample_brownian_motion_at(times)
        ret = bm + np.array(times) * (self.b - bm[-1]) / times[-1]

        assert isinstance(ret, np.ndarray)

        return ret

    @override
    def sample(self, n: int) -> npt.NDArray[np.float64]:
        return self._sample_brownian_bridge(n)

    @override
    def sample_at(
        self,
        times: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        return self._sample_brownian_bridge_at(times)

    def _sample_brownian_bridge_paths(self, n: int, paths: int) -> npt.NDArray[np.float64]:
        """Generate several realizations of a Brownian bridge, one per row."""
        bm = self._sample_brownian_motion_paths(n, paths)

        bridges: npt.NDArray[np.float64] = bm + self.times * (self.b - bm[:, -1:]) / self.t
        return bridges

    @override
    def sample_paths(self, n: int, paths: int) -> npt.NDArray[np.float64]:
        return self._sample_brownian_bridge_paths(n, paths)
