Continuous-time Processes
=========================

The :py:mod:`samplepaths.continuous` module provides discretely sampled
continuous-time processes. All of them are created with keyword arguments,
and all except the Poisson processes are defined on a time interval
``[0, t]`` with ``t`` defaulting to 1.

Overview
--------

.. list-table::
   :header-rows: 1
   :widths: 34 46 20

   * - Process
     - Parameters
     - ``sample_at()``
   * - :py:class:`~samplepaths.continuous.WienerProcess`
     - ``t``
     - yes
   * - :py:class:`~samplepaths.continuous.BrownianMotion`
     - ``drift``, ``scale``, ``t``
     - yes
   * - :py:class:`~samplepaths.continuous.GeometricBrownianMotion`
     - ``drift``, ``volatility``, ``t``
     - yes
   * - :py:class:`~samplepaths.continuous.BrownianBridge`
     - ``b``, ``t``
     - yes
   * - :py:class:`~samplepaths.continuous.BrownianExcursion`
     - ``t``
     - yes
   * - :py:class:`~samplepaths.continuous.BrownianMeander`
     - ``t``
     - yes
   * - :py:class:`~samplepaths.continuous.BesselProcess`
     - ``dim``, ``t``
     - yes
   * - :py:class:`~samplepaths.continuous.SquaredBesselProcess`
     - ``dim``, ``t``
     - yes
   * - :py:class:`~samplepaths.continuous.FractionalBrownianMotion`
     - ``hurst``, ``t``
     - no
   * - :py:class:`~samplepaths.continuous.MultifractionalBrownianMotion`
     - ``hurst`` (a function of time), ``t``
     - no
   * - :py:class:`~samplepaths.continuous.CauchyProcess`
     - ``t``
     - yes
   * - :py:class:`~samplepaths.continuous.GammaProcess`
     - ``mean`` and ``variance``, or ``rate`` and ``scale``, ``t``
     - yes
   * - :py:class:`~samplepaths.continuous.VarianceGammaProcess`
     - ``drift``, ``variance``, ``scale``, ``t``
     - yes
   * - :py:class:`~samplepaths.continuous.InverseGaussianProcess`
     - ``mean`` (a function of time), ``scale``, ``t``
     - yes
   * - :py:class:`~samplepaths.continuous.PoissonProcess`
     - ``rate``
     - no
   * - :py:class:`~samplepaths.continuous.MixedPoissonProcess`
     - ``rate_func``, ``rate_args``, ``rate_kwargs``
     - no

Brownian motion and its relatives
---------------------------------

.. image:: /_static/brownian_motion.png
    :scale: 50%

A :py:class:`~samplepaths.continuous.WienerProcess` is a standard Brownian
motion. :py:class:`~samplepaths.continuous.BrownianMotion` adds a ``drift``
and a ``scale``, so that its value at time :math:`s` is normally distributed
with mean ``drift * s`` and variance ``scale**2 * s``.

.. code-block:: python

    from samplepaths import BrownianMotion

    bm = BrownianMotion(drift=0.5, scale=2, t=10)
    path = bm.sample(1000)

    # at times of your choice
    values = bm.sample_at([0, 1, 2.5, 10])

:py:class:`~samplepaths.continuous.GeometricBrownianMotion` is the
exponential of a Brownian motion and stays positive, which makes it a common
model for prices. Use ``sample_with_initial()`` to start from a value other
than 1.

.. code-block:: python

    from samplepaths import GeometricBrownianMotion

    gbm = GeometricBrownianMotion(drift=0.05, volatility=0.2, t=1)
    prices = gbm.sample_with_initial(252, initial=100.0)

Conditioned Brownian paths
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. image:: /_static/brownian_bridge.png
    :scale: 50%

Three processes are Brownian motions conditioned on their path:

* :py:class:`~samplepaths.continuous.BrownianBridge` ends at the value ``b``
  at time ``t``
* :py:class:`~samplepaths.continuous.BrownianExcursion` stays nonnegative
  and returns to zero at time ``t``
* :py:class:`~samplepaths.continuous.BrownianMeander` stays nonnegative; its
  endpoint is random, or fixed with ``sample_with_endpoint()``

.. code-block:: python

    from samplepaths import BrownianBridge
    from samplepaths import BrownianMeander

    bridge = BrownianBridge(b=2, t=4)
    path = bridge.sample(256)  # path[0] == 0 and path[-1] == 2

    meander = BrownianMeander(t=1)
    path = meander.sample_with_endpoint(256, b=1.5)

Bessel processes
~~~~~~~~~~~~~~~~

A :py:class:`~samplepaths.continuous.BesselProcess` is the Euclidean norm of
``dim`` independent Brownian motions, and a
:py:class:`~samplepaths.continuous.SquaredBesselProcess` is its square.

.. code-block:: python

    from samplepaths import BesselProcess

    bessel = BesselProcess(dim=3, t=1)
    path = bessel.sample(512)

Fractional Brownian motion
--------------------------

.. image:: /_static/fractional_brownian_motion.png
    :scale: 50%

A :py:class:`~samplepaths.continuous.FractionalBrownianMotion` has
correlated increments, controlled by the Hurst parameter ``hurst`` in
``(0, 1)``:

* ``hurst > 0.5``: positively correlated increments, smoother paths
* ``hurst = 0.5``: a standard Brownian motion
* ``hurst < 0.5``: negatively correlated increments, rougher paths

.. image:: /_static/hurst.png
    :alt: Fractional Brownian motion for three values of the Hurst parameter
    :width: 100%

.. code-block:: python

    from samplepaths import FractionalBrownianMotion

    fbm = FractionalBrownianMotion(hurst=0.7, t=1)
    path = fbm.sample(1024)  # Davies-Harte method
    path = fbm.sample_hosking(256)  # Hosking's method

In a :py:class:`~samplepaths.continuous.MultifractionalBrownianMotion` the
Hurst parameter is a function of time with values in ``(0, 1)``, so the
roughness of the path changes over time. It is generated with the
Riemann-Liouville method, which takes a number of operations proportional to
the square of ``n``.

.. code-block:: python

    from samplepaths import MultifractionalBrownianMotion

    mbm = MultifractionalBrownianMotion(hurst=lambda t: 0.3 + 0.4 * t, t=1)
    path = mbm.sample(256)

Jump processes
--------------

.. image:: /_static/gamma_process.png
    :scale: 50%

These processes move by jumps instead of continuous paths.

A :py:class:`~samplepaths.continuous.GammaProcess` is nondecreasing with
gamma distributed increments. Specify it either by the ``mean`` and
``variance`` of its increase per unit time, or by the ``rate`` and ``scale``
of its jumps. The other pair is derived.

.. code-block:: python

    from samplepaths import GammaProcess

    gp = GammaProcess(mean=1, variance=2)
    print(gp.rate, gp.scale)
    # 0.5 0.5

    path = gp.sample(512)

A :py:class:`~samplepaths.continuous.VarianceGammaProcess` is a Brownian
motion with drift evaluated at a random time given by a gamma process, and a
:py:class:`~samplepaths.continuous.CauchyProcess` is a Brownian motion
evaluated at a random time given by a Lévy process. An
:py:class:`~samplepaths.continuous.InverseGaussianProcess` is nondecreasing
with inverse Gaussian increments; its ``mean`` is an increasing function of
time.

.. code-block:: python

    from samplepaths import InverseGaussianProcess
    from samplepaths import VarianceGammaProcess

    vg = VarianceGammaProcess(drift=0, variance=1, scale=1, t=1)
    path = vg.sample(512)

    ig = InverseGaussianProcess(mean=lambda t: 2 * t, scale=1, t=1)
    path = ig.sample(512)

Poisson processes
-----------------

.. image:: /_static/poisson_process.png
    :scale: 50%

The Poisson processes are different from the other continuous-time
processes: a realization is the sequence of arrival times, not the value of
the process on a time grid. They have no parameter ``t``.

``sample(n)`` generates the times of ``n`` arrivals, preceded by zero.
``sample_with_length(length)`` generates the arrivals over a length of time
instead; its last value is the first arrival after ``length``.

.. code-block:: python

    from samplepaths import PoissonProcess

    pp = PoissonProcess(rate=2.0)

    times = pp.sample(100)  # 101 values, starting at zero
    times = pp.sample_with_length(10.0)

In a :py:class:`~samplepaths.continuous.MixedPoissonProcess` the rate is
random. A new rate is drawn from ``rate_func`` for each realization, and is
available as the ``rate`` attribute afterwards.

.. code-block:: python

    import numpy as np
    from samplepaths import MixedPoissonProcess

    rng = np.random.default_rng(3)
    mpp = MixedPoissonProcess(rate_func=rng.uniform, rate_args=(1, 5), rng=rng)

    times = mpp.sample(100)
    print(mpp.rate)  # the rate used for this realization
