Noise Processes
===============

The :py:mod:`samplepaths.noise` module provides noise processes: sequences
of increments instead of cumulative paths. A realization of ``n`` increments
on the interval ``[0, t]`` holds ``n`` values.

Overview
--------

.. list-table::
   :header-rows: 1
   :widths: 40 30 30

   * - Process
     - Parameters
     - Spectral density
   * - :py:class:`~samplepaths.noise.GaussianNoise`
     - ``t``
     - flat
   * - :py:class:`~samplepaths.noise.FractionalGaussianNoise`
     - ``hurst``, ``t``
     - depends on ``hurst``
   * - :py:class:`~samplepaths.noise.ColoredNoise`
     - ``beta``, ``t``
     - :math:`1 / f^\beta`
   * - :py:class:`~samplepaths.noise.VioletNoise`
     - ``t``
     - :math:`f^2`
   * - :py:class:`~samplepaths.noise.BlueNoise`
     - ``t``
     - :math:`f`
   * - :py:class:`~samplepaths.noise.WhiteNoise`
     - ``t``
     - flat
   * - :py:class:`~samplepaths.noise.PinkNoise`
     - ``t``
     - :math:`1 / f`
   * - :py:class:`~samplepaths.noise.RedNoise` and
       :py:class:`~samplepaths.noise.BrownianNoise`
     - ``t``
     - :math:`1 / f^2`

Gaussian increments
-------------------

.. image:: /_static/gaussian_noise.png
    :scale: 50%

These noises are the increments of their continuous-time counterparts: the
cumulative sum of a :py:class:`~samplepaths.noise.GaussianNoise` is a Wiener
process, and the cumulative sum of a
:py:class:`~samplepaths.noise.FractionalGaussianNoise` is a fractional
Brownian motion.

Each increment of a Gaussian noise is normally distributed with variance
equal to the time step ``t / n``.

.. code-block:: python

    import numpy as np
    from samplepaths import GaussianNoise

    gn = GaussianNoise(t=1)
    increments = gn.sample(1000)

    wiener = np.concatenate(([0], np.cumsum(increments)))

    # increments between times of your choice
    increments = gn.sample_at([0, 0.5, 1, 3])

The increments of a fractional Gaussian noise are correlated, positively for
``hurst > 0.5`` and negatively for ``hurst < 0.5``. Two exact algorithms are
available:

.. code-block:: python

    from samplepaths import FractionalGaussianNoise

    fgn = FractionalGaussianNoise(hurst=0.2, t=1)

    increments = fgn.sample(1024)  # Davies-Harte method
    increments = fgn.sample_hosking(256)  # Hosking's method

The Davies-Harte method uses fast Fourier transforms and is the faster of
the two. Hosking's method takes a number of operations proportional to the
square of ``n``.

Colored noise
-------------

.. image:: /_static/colored_noise.png
    :scale: 50%

A colored noise is a signal whose power spectral density follows a power
law, :math:`S(f) \propto 1 / f^\beta`. The exponent ``beta`` determines the
"color": a positive ``beta`` puts more power in the low frequencies, which
gives slowly varying signals, and a negative ``beta`` puts more power in the
high frequencies.

.. image:: /_static/noise_spectra.png
    :alt: Power spectral densities of the five named colors of noise
    :width: 100%

:py:class:`~samplepaths.noise.ColoredNoise` takes any ``beta``. The named
colors are the same process with a fixed exponent.

.. code-block:: python

    from samplepaths import ColoredNoise
    from samplepaths import PinkNoise

    pink = PinkNoise(t=1)
    signal = pink.sample(4096)

    # the same color, and one between pink and red
    signal = ColoredNoise(beta=1, t=1).sample(4096)
    signal = ColoredNoise(beta=1.5, t=1).sample(4096)

Realizations are generated in the frequency domain by the method of Timmer
and Koenig. A realization has a zero mean and is scaled so that its variance
is close to ``t / n``, the variance of Gaussian noise with the same number of
increments on the same interval, whatever the exponent. It needs at least
two increments.
