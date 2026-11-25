"""Brownian (red) noise."""

from .red_noise import RedNoise


class BrownianNoise(RedNoise):
    r"""Brownian (red) noise.

    .. image:: /_static/red_noise.png
        :scale: 50%

    Colored noise, or power law noise with spectral density exponent
    :math:`\beta = 2`.

    :param t: the right hand endpoint of the time interval :math:`[0,t]`
        for the process
    :param rng: a custom random number generator
    """
