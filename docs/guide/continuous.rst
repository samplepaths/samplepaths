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
   * - :py:class:`~samplepaths.continuous.NormalInverseGaussianProcess`
     - ``alpha``, ``beta``, ``delta``, ``mu``, ``t``
     - yes
   * - :py:class:`~samplepaths.continuous.CompoundPoissonProcess`
     - ``rate``, ``jumps``, ``t``
     - yes
   * - :py:class:`~samplepaths.continuous.MertonJumpDiffusion`
     - ``drift``, ``volatility``, ``rate``, ``jump_mean``,
       ``jump_volatility``, ``t``
     - yes
   * - :py:class:`~samplepaths.continuous.KouJumpDiffusion`
     - ``drift``, ``volatility``, ``rate``, ``up_probability``, ``up_rate``,
       ``down_rate``, ``t``
     - yes
   * - :py:class:`~samplepaths.continuous.PoissonProcess`
     - ``rate``
     - no
   * - :py:class:`~samplepaths.continuous.HawkesProcess`
     - ``baseline``, ``excitation``, ``decay``
     - no
   * - :py:class:`~samplepaths.continuous.MixedPoissonProcess`
     - ``rate_func``, ``rate_args``, ``rate_kwargs``
     - no

Brownian motion and its relatives
---------------------------------

.. image:: /_static/brownian_motion.png
    :scale: 50%

A :py:class:`~samplepaths.continuous.WienerProcess` is a standard Brownian
motion, without parameters other than ``t``.
:py:class:`~samplepaths.continuous.BrownianMotion` adds a ``drift`` and a
``scale``, so that its value at time :math:`s` is normally distributed
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

All of its sampling methods return the motion, which starts at zero. The
increments alone are a
:py:class:`~samplepaths.noise.FractionalGaussianNoise`.

A :py:class:`~samplepaths.continuous.FractionalGeometricBrownianMotion` is
the exponential of a fractional Brownian motion with a trend, the fractional
counterpart of the geometric Brownian motion. Its mean grows as
:math:`e^{\mu t}` whatever the Hurst parameter, and its realizations are
exact. The :doc:`diffusion processes <diffusion>` driven by a fractional
Brownian motion are described on their own page.

.. image:: /_static/fractional_geometric_brownian_motion.png
    :scale: 50%

.. code-block:: python

    from samplepaths import FractionalGeometricBrownianMotion

    fgbm = FractionalGeometricBrownianMotion(hurst=0.7, drift=0.05, volatility=0.2)
    prices = fgbm.sample_paths_with_initial(252, paths=1000, initial=100.0)

In a :py:class:`~samplepaths.continuous.MultifractionalBrownianMotion` the
Hurst parameter is a function of time with values in ``(0, 1)``, so the
roughness of the path changes over time. It is generated with the
Riemann-Liouville method, which takes a number of operations proportional to
the square of ``n``.

.. code-block:: python

    from samplepaths import MultifractionalBrownianMotion

    mbm = MultifractionalBrownianMotion(hurst=lambda t: 0.3 + 0.4 * t, t=1)
    path = mbm.sample(256)

Jump and Poisson processes
--------------------------

The processes which move by jumps, the jump diffusions and the Poisson
processes are described in :doc:`/guide/continuous/jumps`.

.. toctree::
   :hidden:

   continuous/jumps
