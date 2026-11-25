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

A :py:class:`~samplepaths.continuous.NormalInverseGaussianProcess` is, like
the variance gamma process, a Brownian motion with drift on a random clock,
here an inverse Gaussian process. ``alpha`` sets how heavy its tails are and
``beta`` its asymmetry.

.. code-block:: python

    from samplepaths import NormalInverseGaussianProcess

    nig = NormalInverseGaussianProcess(alpha=2.0, beta=-0.5, delta=1.0, t=1)
    path = nig.sample(512)

Compound Poisson process
~~~~~~~~~~~~~~~~~~~~~~~~

A :py:class:`~samplepaths.continuous.CompoundPoissonProcess` is constant
between the jumps of a Poisson process with the given ``rate`` and moves by a
random amount at each jump. The amounts are drawn by a function which takes a
generator and the number of jumps; the default draws from the standard normal
distribution.

.. code-block:: python

    from samplepaths import CompoundPoissonProcess

    # claims of an insurer: 5 per year on average, with exponential sizes
    claims = CompoundPoissonProcess(
        rate=5.0,
        jumps=lambda rng, size: rng.exponential(1000.0, size),
        t=10,
    )
    total = claims.sample(1000)

Draw from the generator the function is given, not from another one. This
keeps the realizations reproducible, also on several threads.

Jump diffusions
~~~~~~~~~~~~~~~

A jump diffusion is a geometric Brownian motion whose value is multiplied by
a random factor at the jumps of a Poisson process. It models a price which
moves continuously most of the time and occasionally gaps.

* :py:class:`~samplepaths.continuous.MertonJumpDiffusion` has jumps with a
  normal distribution in the logarithm, with mean ``jump_mean`` and standard
  deviation ``jump_volatility``.
* :py:class:`~samplepaths.continuous.KouJumpDiffusion` has upward jumps, with
  probability ``up_probability``, and downward jumps, with exponential sizes
  of rates ``up_rate`` and ``down_rate``. Different rates give the asymmetry
  seen in returns.

In both, ``drift`` is the expected rate of return including the jumps: the
expected value at time ``t`` is ``initial * exp(drift * t)``.

.. code-block:: python

    from samplepaths import KouJumpDiffusion
    from samplepaths import MertonJumpDiffusion

    merton = MertonJumpDiffusion(
        drift=0.05, volatility=0.2, rate=1.0, jump_mean=-0.1, jump_volatility=0.15
    )
    prices = merton.sample_with_initial(252, initial=100.0)

    kou = KouJumpDiffusion(
        drift=0.05, volatility=0.2, rate=1.0, up_probability=0.3, up_rate=20.0, down_rate=10.0
    )
    prices = kou.sample_paths_with_initial(252, paths=10_000, initial=100.0)

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

A :py:class:`~samplepaths.continuous.HawkesProcess` is self-exciting: every
arrival raises the rate of the following arrivals by ``excitation``, and the
effect fades at the rate ``decay``. Its arrivals come in clusters, as
earthquakes and their aftershocks or trades in a market do. Without
excitation it is a Poisson process with the rate ``baseline``.

.. code-block:: python

    from samplepaths import HawkesProcess

    hawkes = HawkesProcess(baseline=1.0, excitation=1.5, decay=2.0)

    times = hawkes.sample(100)  # 101 values, starting at zero
    times = hawkes.sample_with_length(10.0)

``excitation`` must be smaller than ``decay``, so that an arrival triggers
less than one further arrival on average and the process settles to the rate
``baseline / (1 - excitation / decay)``.

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
