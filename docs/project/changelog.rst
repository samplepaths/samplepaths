Changelog
=========

All notable changes to ``samplepaths``, most recent first. Times were
measured as described in :doc:`/guide/performance`.

Unreleased
----------

Added
~~~~~

* A fifth family, the :doc:`time series models </guide/timeseries>`, in the
  new subpackage ``samplepaths.timeseries``, with nine processes:

  * :py:class:`~samplepaths.timeseries.AutoregressiveProcess`,
    :py:class:`~samplepaths.timeseries.MovingAverageProcess`,
    :py:class:`~samplepaths.timeseries.ARIMAProcess` and
    :py:class:`~samplepaths.timeseries.SARIMAProcess`, the linear models
  * :py:class:`~samplepaths.timeseries.ARCHProcess` and
    :py:class:`~samplepaths.timeseries.GARCHProcess`, whose variance depends
    on the past
  * :py:class:`~samplepaths.timeseries.TGARCHProcess`,
    :py:class:`~samplepaths.timeseries.AGARCHProcess` and
    :py:class:`~samplepaths.timeseries.EGARCHProcess`, in which falls raise
    the variance more than rises

  The models with a changing variance keep the standard deviation of every
  value in the attribute ``volatility``. Each model is tested against its
  recurrence and its long-run variance, and all agree in distribution with
  ``stochastic-rs``. With 14 threads, 100,000 realizations of 64 values take
  8 milliseconds for an AR(2) process and 16 for a GARCH(1, 1) process,
  against 11 and 11 for ``stochastic-rs``; see :doc:`comparison`.
* A sixth family, the :doc:`processes with several components
  </guide/multivariate>`, in the new subpackage ``samplepaths.multivariate``.
  A realization is a two-dimensional array with one component per row, and
  ``sample_paths()`` returns an array of shape
  ``(paths, components, n + 1)``.

  * :py:class:`~samplepaths.multivariate.CorrelatedBrownianMotions`, for any
    number of motions and any correlation matrix
  * :py:class:`~samplepaths.multivariate.HestonModel`, with the variance
    drawn from its exact distribution
  * :py:class:`~samplepaths.multivariate.BatesModel`, a Heston model with
    jumps
  * :py:class:`~samplepaths.multivariate.SABRModel`

  All agree in distribution with ``stochastic-rs``. With 14 threads, 100,000
  realizations of 64 increments take 10 milliseconds for correlated Brownian
  motions and 61 for a Heston model, against 16 and 26 for ``stochastic-rs``.
* Five fractional processes, driven by a fractional Brownian motion with a
  Hurst parameter of your choice:

  * :py:class:`~samplepaths.diffusion.FractionalOrnsteinUhlenbeckProcess`,
    :py:class:`~samplepaths.diffusion.FractionalVasicekProcess`,
    :py:class:`~samplepaths.diffusion.FractionalCoxIngersollRossProcess` and
    :py:class:`~samplepaths.diffusion.FractionalJacobiProcess`, sampled with
    the Euler method on exact increments of the motion
  * :py:class:`~samplepaths.continuous.FractionalGeometricBrownianMotion`,
    which is exact

  All agree in distribution with ``stochastic-rs``. With 14 threads, 100,000
  realizations of 64 increments of a fractional Vasicek process take 16
  milliseconds, against 15 for ``stochastic-rs``.
* Ten further processes:

  * :py:class:`~samplepaths.continuous.CompoundPoissonProcess`, with jump
    sizes drawn by a function of your choice
  * :py:class:`~samplepaths.continuous.HawkesProcess`, a self-exciting point
    process
  * :py:class:`~samplepaths.continuous.NormalInverseGaussianProcess`
  * :py:class:`~samplepaths.continuous.MertonJumpDiffusion` and
    :py:class:`~samplepaths.continuous.KouJumpDiffusion`
  * :py:class:`~samplepaths.diffusion.JacobiProcess` and
    :py:class:`~samplepaths.diffusion.KimuraProcess`, on the unit interval
  * :py:class:`~samplepaths.diffusion.VerhulstProcess`,
    :py:class:`~samplepaths.diffusion.GompertzProcess` and
    :py:class:`~samplepaths.diffusion.FellerLogisticProcess`, for the size of
    a population

  Each is tested against its known moments or behavior, and all agree in
  distribution with ``stochastic-rs``; see :doc:`comparison`.
* ``docs/scripts/compare_rust_outputs.py`` compares the distributions of the
  results of ``samplepaths`` and ``stochastic-rs`` for the processes they
  share, in 39 comparisons.

* ``sample_paths(n, paths, workers=...)`` and ``sample_paths_with_initial()``
  generate the realizations on several threads. With 14 threads, 100,000
  realizations of 64 increments take 5.7 instead of 33.8 milliseconds for a
  Brownian motion and 11.3 instead of 81.2 for a fractional Brownian motion.
  The result is reproducible for a given seed and number of workers. See
  :doc:`/guide/paths`.
* A :doc:`comparison with other packages <comparison>`, with times measured
  against ``stochastic-rs``, and ``docs/scripts/compare_rust.py`` to
  reproduce them.

Faster
~~~~~~

* ``sample_paths()`` of the diffusion processes with a constant speed and no
  volatility exponent, such as the Vasicek and the Ornstein-Uhlenbeck
  processes, computes all steps with array operations instead of a loop over
  the steps. With threads, 20,000 realizations of 1024 increments take 26
  instead of 197 milliseconds. For 32 realizations or more, the values agree
  with those of version 0.4.0 to a relative difference of 1e-9 instead of
  being identical.

Fixed
~~~~~

Comparing the results with those of ``stochastic-rs`` showed two processes to
be wrong. Both defects were inherited from the original package, and both
change results.

* :py:class:`~samplepaths.continuous.VarianceGammaProcess` scaled its normal
  variates by the square root of the time step, on top of the gamma clock
  which already carries the time. The spread of a realization was too small
  and depended on the number of increments: with a unit ``scale`` and
  ``variance`` the standard deviation at time 1 was 0.27 for 16 increments
  and 0.10 for 1024, instead of 1.0. ``sample()``, ``sample_at()`` and
  ``sample_paths()`` are corrected.
* :py:class:`~samplepaths.diffusion.ConstantElasticityVarianceProcess` had
  the drift :math:`\mu (X_t - 1)` instead of the documented
  :math:`\mu X_t`. Started from 1, its mean stayed near 1 instead of growing
  as :math:`e^{\mu t}`.

Apart from these two processes, realizations of ``sample()`` and
``sample_at()`` for a given seed are identical bit for bit to those of
version 0.4.0: 879 of the 906 compared cases, the other 27 being the two
corrected processes.

0.4.0 (2026-11-25)
------------------

This version corrects behavior inherited from the original package. Some
results and parts of the interface change; each change is listed below.
Realizations for a given seed are otherwise identical bit for bit to those
of version 0.3.0.

Changed
~~~~~~~

* **Fractional Brownian motion.** ``sample_davies_harte()`` and
  ``sample_hosking()`` of
  :py:class:`~samplepaths.continuous.FractionalBrownianMotion` return the
  motion, ``n + 1`` values starting at zero, like ``sample()``. They used to
  return the ``n`` increments. Use
  :py:class:`~samplepaths.noise.FractionalGaussianNoise` for the increments.
* **Colored noise.** A realization of
  :py:class:`~samplepaths.noise.ColoredNoise` and of the named colors is
  scaled by the square root of ``t``, so that its variance is close to
  ``t / n``, the variance of Gaussian noise on the same interval. The
  parameter ``t`` used to have no effect. Results for the default ``t = 1``
  do not change.
* **Colored noise needs two increments.** ``sample(1)`` raises ``ValueError``
  instead of returning ``nan``.
* **Volatility exponent of the diffusion processes.** A ``volexp`` given as a
  function is called with the time, as documented and like the other
  parameters. It used to be called with the value of the process.
* **Processes built on Brownian motion.**
  :py:class:`~samplepaths.continuous.WienerProcess`,
  :py:class:`~samplepaths.continuous.BrownianBridge`,
  :py:class:`~samplepaths.continuous.BrownianExcursion`,
  :py:class:`~samplepaths.continuous.BrownianMeander`,
  :py:class:`~samplepaths.continuous.BesselProcess`,
  :py:class:`~samplepaths.continuous.SquaredBesselProcess` and
  :py:class:`~samplepaths.continuous.CauchyProcess` no longer have the
  attributes ``drift`` and ``scale`` and are no longer instances of
  :py:class:`~samplepaths.continuous.BrownianMotion`. The attributes were
  not parameters of these processes, but setting them changed the
  realizations. :py:class:`~samplepaths.continuous.GeometricBrownianMotion`
  keeps its own ``drift`` and loses ``scale``.

Fixed
~~~~~

* :py:class:`~samplepaths.continuous.InverseGaussianProcess` computes the
  increments of its mean again when ``mean`` is changed on an instance. It
  used to keep those of the previous function.

Added
~~~~~

* Support for Python 3.11. The package used to require Python 3.12.
* A :doc:`gallery of examples </gallery/index>`, one for every process and
  three on Monte Carlo simulation. The examples are run when the
  documentation is built.

Documentation and project
~~~~~~~~~~~~~~~~~~~~~~~~~

* The API reference writes array types as ``NDArray[float64]``.
* Documentation built from a commit which is not a release is titled
  "development version" instead of a version number derived from the commit.
* The notebooks which compared the package with ``stochastic`` during the
  port were removed. Seeded tests cover the comparison between versions, and
  the gallery replaces the notebooks as examples.
* Every pull request compares the sampling times with those of the base
  branch and fails when a method becomes more than two times slower.
* A contributing guide, issue templates and automated dependency updates
  were added.

0.3.0 (2026-11-25)
------------------

Added
~~~~~

* ``sample_paths(n, paths)`` on every process generates many independent
  realizations in one call, as a two-dimensional array. For 1000 realizations
  of 256 increments it is 2 to 13 times faster than a loop over ``sample()``
  for most processes: 1.8 instead of 14.6 milliseconds for a Vasicek process,
  1.4 instead of 5.2 for a Brownian motion. See :doc:`/guide/paths`.
* ``sample_paths_with_initial(n, paths, initial)`` on
  :py:class:`~samplepaths.continuous.GeometricBrownianMotion` and on the
  diffusion processes.
* ``docs/scripts/benchmark.py --paths`` measures many realizations.

Faster
~~~~~~

* Diffusion processes: 0.21 instead of 0.98 milliseconds for 4096 increments
  with constant parameters (4.7 times faster), and 0.44 instead of 1.00 for
  the Cox-Ingersoll-Ross process.
* :py:class:`~samplepaths.continuous.MultifractionalBrownianMotion` and
  Hosking's method keep the part of the computation which does not depend on
  the random numbers. From the second realization of an instance on, 512
  increments of the former take 0.34 instead of 1.6 milliseconds and 1024
  increments of the latter 0.98 instead of 3.4.
* :py:class:`~samplepaths.discrete.ChineseRestaurantProcess` with many
  tables: 12 instead of 33 milliseconds for 4096 customers with
  ``discount=0.5`` and ``strength=2``, and 25 instead of 281 with
  ``discount=0.9`` and ``strength=5``.

Realizations of ``sample()`` and ``sample_at()`` for a given seed are
identical bit for bit to those of version 0.2.0 in all 906 compared cases,
which cover every process and sampling method.

Changed
~~~~~~~

* In the diffusion processes, an initial value outside the domain of the
  volatility term behaves like any later value. A negative initial value with
  a fractional exponent gives ``nan``, where the first step used to produce a
  complex number, and an initial value of zero with a negative exponent gives
  ``inf`` instead of raising ``ZeroDivisionError``. Both emit a
  ``RuntimeWarning``.

0.2.0 (2026-11-25)
------------------

Faster
~~~~~~

Loops over the increments in Python were replaced with ``numpy`` operations.
Times in milliseconds, for 4096 values unless noted:

.. list-table::
   :header-rows: 1
   :widths: 52 16 16 16

   * - Method
     - 0.1.1
     - 0.2.0
     - Gain
   * - ``CauchyProcess.sample_at()``
     - 38.6
     - 0.10
     - 370x
   * - ``BesselProcess.sample()``
     - 2.39
     - 0.03
     - 70x
   * - ``MarkovChain.sample()``
     - 15.1
     - 0.21
     - 73x
   * - ``FractionalGaussianNoise.sample_hosking()`` (1024)
     - 237
     - 3.3
     - 72x
   * - ``GaussianNoise.sample_at()``
     - 1.68
     - 0.03
     - 49x
   * - ``BrownianMotion.sample_at()``
     - 1.77
     - 0.05
     - 36x
   * - ``BrownianMeander.sample_at()``
     - 5.31
     - 0.17
     - 32x
   * - ``MultifractionalBrownianMotion.sample()`` (512)
     - 45.5
     - 1.6
     - 29x
   * - ``MoranProcess.sample()``
     - 8.70
     - 0.39
     - 23x
   * - ``InverseGaussianProcess.sample()``
     - 0.95
     - 0.07
     - 15x
   * - ``ChineseRestaurantProcess.sample()``
     - 17.3
     - 3.4
     - 5x
   * - ``DirichletProcess.sample()``
     - 12.3
     - 4.9
     - 2.5x

Of 831 seeded realizations compared with version 0.1.1, 781 are identical
bit for bit, 48 agree to a relative difference of 1e-9, where ``numpy`` now
adds the terms of a sum in a different order, and 2, both from
``CauchyProcess.sample_at()``, agree to 2.5e-8 because of the fix below.

Fixed
~~~~~

* ``CauchyProcess.sample_at()`` no longer fails with "Times must be strictly
  increasing" when its random time change makes a very large jump. The path
  is built from the increments of the time change directly.
* :py:class:`~samplepaths.noise.ColoredNoise` computes its frequencies again
  when ``beta`` or ``t`` is changed on an instance.
* :py:class:`~samplepaths.discrete.MoranProcess` computes its transition
  probabilities again when ``maximum`` is changed on an instance.

Added
~~~~~

* ``docs/scripts/benchmark.py`` and ``make benchmark``.

0.1.1 (2026-11-25)
------------------

* The package version is read from the git tag of the release.
* The README was rewritten in Markdown. The library code is unchanged from
  0.1.0.

0.1.0 (2026-11-25)
------------------

First release, a fork of ``stochastic`` 0.7.0 under a new name. The changes
from the original package are described in :doc:`migration`. In short:

* runs on Python 3.12 or newer with ``numpy`` 2 and ``scipy`` 1.15
* type hints throughout
* one ``sample(n)`` signature on every process, with dedicated methods for
  the variants
* keyword-only arguments on the continuous-time, diffusion and noise
  processes
* random number generation with :py:class:`numpy.random.Generator` only
* the ``times`` attribute replaces the ``times(n)`` method
* corrected drift of :py:class:`~samplepaths.continuous.BrownianMotion` and
  :py:class:`~samplepaths.continuous.GeometricBrownianMotion` for an end time
  other than 1
