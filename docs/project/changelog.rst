Changelog
=========

All notable changes to ``samplepaths``, most recent first. Times were
measured as described in :doc:`/guide/performance`.

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
