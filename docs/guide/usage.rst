Sampling
========

Every process in ``samplepaths`` is a class. You create an instance with the
parameters of the process and call its sampling methods to generate
realizations as ``numpy`` arrays.

This page covers one realization at a time. :doc:`paths` covers many
realizations in one call.

Creating a process
------------------

Processes can be imported from the top-level package or from the subpackage
of their family:

.. code-block:: python

    from samplepaths import BernoulliProcess
    from samplepaths.discrete import BernoulliProcess

The parameters are validated at instantiation and available as attributes
afterwards. Assigning an attribute validates the new value as well.

.. code-block:: python

    from samplepaths import BernoulliProcess

    bp = BernoulliProcess(p=0.6)
    print(bp.p)
    # 0.6

    bp.p = 0.25  # validated, raises ValueError for a value outside [0, 1]

The continuous-time, diffusion and noise processes take keyword arguments
only:

.. code-block:: python

    from samplepaths import BrownianMotion

    bm = BrownianMotion(drift=1, scale=2, t=10)

The ``sample()`` method
-----------------------

Every process has a ``sample(n)`` method which generates a realization of
``n`` steps.

.. code-block:: python

    from samplepaths import BernoulliProcess

    bp = BernoulliProcess(p=0.6)
    s = bp.sample(16)

.. image:: /_static/sampling_grid.png
    :alt: A realization of 8 increments holds 9 values on the time interval
    :width: 100%

What ``n`` counts depends on the family of the process:

.. list-table::
   :header-rows: 1
   :widths: 25 45 30

   * - Family
     - Meaning of ``n``
     - Length of the realization
   * - Continuous-time and diffusion
     - number of equally spaced increments on ``[0, t]``
     - ``n + 1``, starting at time zero
   * - Noise
     - number of increments on ``[0, t]``
     - ``n``
   * - Discrete-time
     - number of steps
     - ``n`` (``n + 1`` for a random walk, which starts at zero)
   * - Poisson processes
     - number of arrivals
     - ``n + 1`` arrival times, starting at zero

The time interval
-----------------

The continuous-time, diffusion and noise processes are defined on a time
interval ``[0, t]``. The parameter ``t`` is given at instantiation and
defaults to 1.

After a call to ``sample(n)``, the attribute ``times`` holds the ``n + 1``
equally spaced time values of the realization and ``n`` holds the number of
increments. This is particularly useful for plotting.

.. code-block:: python

    import matplotlib.pyplot as plt
    from samplepaths import FractionalBrownianMotion

    fbm = FractionalBrownianMotion(hurst=0.7, t=5)
    s = fbm.sample(512)

    print(fbm.n, fbm.times[0], fbm.times[-1])
    # 512 0.0 5.0

    plt.plot(fbm.times, s)
    plt.show()

Reading ``times`` before the first call to ``sample()`` raises a
``RuntimeError``.

The ``sample_at()`` method
--------------------------

Most continuous-time processes can also generate a realization at times of
your choice, which do not need to be equally spaced. ``sample_at()`` takes
a sequence of increasing time values and ignores the parameter ``t``.

.. code-block:: python

    from samplepaths import BrownianMotion

    bm = BrownianMotion(drift=1, scale=1)
    times = [0, 3, 10, 11, 11.2, 20]
    s = bm.sample_at(times)

The following processes cannot be sampled at specified times: the diffusion
processes, the fractional and multifractional Brownian motions, and the
fractional Gaussian and colored noises. Their ``sample_at()`` raises
``NotImplementedError``.

The ``sample_paths()`` method
-----------------------------

Every process has a ``sample_paths(n, paths)`` method which generates
``paths`` independent realizations of ``n`` steps and returns them as the rows
of a two-dimensional array. It is the method to use for Monte Carlo
simulation. See :doc:`paths`.

.. code-block:: python

    from samplepaths import BrownianMotion

    bm = BrownianMotion(drift=1, scale=1)
    realizations = bm.sample_paths(64, paths=1000)  # shape (1000, 65)

Sampling with additional parameters
-----------------------------------

``sample(n)`` and ``sample_paths(n, paths)`` have the same signature on every
process. Processes whose realizations depend on an additional value have
dedicated methods:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Method
     - Available on
   * - ``sample_with_initial(n, initial)``
     - :py:class:`~samplepaths.continuous.GeometricBrownianMotion`,
       :py:class:`~samplepaths.continuous.FractionalGeometricBrownianMotion`
       and all diffusion processes
   * - ``sample_at_with_initial(times, initial)``
     - :py:class:`~samplepaths.continuous.GeometricBrownianMotion`
   * - ``sample_paths_with_initial(n, paths, initial)``
     - :py:class:`~samplepaths.continuous.GeometricBrownianMotion`,
       :py:class:`~samplepaths.continuous.FractionalGeometricBrownianMotion`
       and all diffusion processes
   * - ``sample_with_endpoint(n, b)`` and
       ``sample_at_with_endpoint(times, b)``
     - :py:class:`~samplepaths.continuous.BrownianMeander`
   * - ``sample_with_length(length)``
     - :py:class:`~samplepaths.continuous.PoissonProcess` and
       :py:class:`~samplepaths.continuous.MixedPoissonProcess`
   * - ``sample_with_start(n, start)``
     - :py:class:`~samplepaths.discrete.MoranProcess`
   * - ``sample_increments(n)``
     - :py:class:`~samplepaths.discrete.RandomWalk`
   * - ``sample_partition(n)``
     - :py:class:`~samplepaths.discrete.ChineseRestaurantProcess`

.. code-block:: python

    from samplepaths import GeometricBrownianMotion
    from samplepaths import PoissonProcess

    gbm = GeometricBrownianMotion(drift=0.1, volatility=0.2, t=1)
    s = gbm.sample_with_initial(32, initial=100.0)

    pp = PoissonProcess(rate=2.0)
    arrivals = pp.sample_with_length(10.0)

Choosing an algorithm
---------------------

The fractional Gaussian noise and the fractional Brownian motion can be
generated by two exact algorithms. ``sample()`` uses the Davies-Harte
method, which is the faster one. Each algorithm also has its own method.

.. code-block:: python

    from samplepaths import FractionalGaussianNoise

    fgn = FractionalGaussianNoise(hurst=0.6, t=1)
    s = fgn.sample(32)  # Davies-Harte method (default)
    s = fgn.sample_davies_harte(32)
    s = fgn.sample_hosking(32)

Writing generic code
--------------------

All processes derive from :py:class:`~samplepaths.base.BaseProcess`, which
defines ``sample(n)`` and ``sample_paths(n, paths)``. Processes on a time
interval derive from
:py:class:`~samplepaths.base.BaseTimeProcess`, which adds ``t``, ``times``
and ``sample_at()``. Use these classes to annotate code which works with any
process.

.. code-block:: python

    import numpy as np
    from samplepaths import BrownianMotion
    from samplepaths import OrnsteinUhlenbeckProcess
    from samplepaths.base import BaseTimeProcess


    def envelope(process: BaseTimeProcess, n: int, paths: int) -> tuple[np.ndarray, np.ndarray]:
        """Estimate the 5% and 95% quantiles of a process over time."""
        samples = process.sample_paths(n, paths)
        return np.quantile(samples, 0.05, axis=0), np.quantile(samples, 0.95, axis=0)


    low, high = envelope(BrownianMotion(), n=64, paths=500)
    low, high = envelope(OrnsteinUhlenbeckProcess(speed=2), n=64, paths=500)

Performance
-----------

Sampling is fast, and making it faster does not change seeded results.
:doc:`performance` gives measured times for every process and explains how
to get the most out of the library.
