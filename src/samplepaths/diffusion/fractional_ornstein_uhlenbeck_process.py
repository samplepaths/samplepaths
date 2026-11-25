"""Fractional Ornstein-Uhlenbeck process."""

from numpy.random import Generator

from .fractional_vasicek_process import FractionalVasicekProcess


class FractionalOrnsteinUhlenbeckProcess(FractionalVasicekProcess):
    r"""Fractional Ornstein-Uhlenbeck process.

    .. image:: /_static/fractional_ornstein_uhlenbeck_process.png
        :scale: 50%

    A process :math:`X_t` which reverts to zero, driven by a fractional
    Brownian motion :math:`B^H_t`:

    .. math::

        dX_t = -\theta X_t dt + \sigma dB^H_t

    It is a :py:class:`~samplepaths.diffusion.FractionalVasicekProcess` with
    a mean of zero. Realizations are generated using the Euler method with
    exact increments of the fractional Brownian motion. :py:meth:`sample`
    starts from 1.

    :param hurst: the Hurst parameter :math:`H` in :math:`(0,1)`
    :param speed: the speed of reversion, or :math:`\theta` above
    :param vol: volatility coefficient of the process, or :math:`\sigma`
        above
    :param t: the right hand endpoint of the time interval :math:`[0,t]`
        for the process
    :param rng: a custom random number generator
    """

    def __init__(
        self,
        *,
        hurst: float = 0.7,
        speed: float = 1.0,
        vol: float = 1.0,
        t: float = 1.0,
        rng: Generator | None = None,
    ) -> None:
        super().__init__(hurst=hurst, speed=speed, mean=0.0, vol=vol, t=t, rng=rng)

    def __str__(self) -> str:
        return (
            f"Fractional Ornstein-Uhlenbeck process with Hurst {self.hurst}, "
            f"speed={self.speed}, vol={self.vol} on [0, {self.t}]"
        )

    def __repr__(self) -> str:
        return (
            f"FractionalOrnsteinUhlenbeckProcess(hurst={self.hurst}, speed={self.speed}, "
            f"vol={self.vol}, t={self.t})"
        )
