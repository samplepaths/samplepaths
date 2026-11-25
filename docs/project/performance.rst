Performance
===========

``samplepaths`` generates realizations with ``numpy`` operations wherever the
algorithm allows it. This page reports measured sampling times and how they
compare with version 0.1.1, in which several methods still looped over the
increments in Python.

How the times were measured
---------------------------

The times below are the mean time of one call, measured with
``docs/scripts/benchmark.py`` on one machine:

* Apple M4 Pro, macOS 27.0
* Python 3.13.2, ``numpy`` 2.4.1, ``scipy`` 1.17.0

Each realization has 4096 increments, except for the two algorithms whose cost
grows with the square of the number of increments: Hosking's method (1024)
and the multifractional Brownian motion (512). Absolute times depend on the
machine; the ratios are what carries over.

Run the benchmark on your own machine with:

.. code-block:: bash

   make benchmark

Methods that became faster
--------------------------

The times are in milliseconds per call. "Values" is the number of increments
passed to ``sample()``, or the number of time values passed to
``sample_at()``.

.. list-table::
   :header-rows: 1
   :widths: 46 12 14 14 14

   * - Method
     - Values
     - 0.1.1
     - Now
     - Speedup
   * - ``CauchyProcess.sample_at()``
     - 4097
     - 38.6
     - 0.10
     - 373x
   * - ``BesselProcess.sample_at()``
     - 4097
     - 4.10
     - 0.05
     - 76x
   * - ``FractionalGaussianNoise.sample_hosking()``
     - 1024
     - 237
     - 3.31
     - 72x
   * - ``MarkovChain.sample()``
     - 4096
     - 15.1
     - 0.21
     - 71x
   * - ``BesselProcess.sample()``
     - 4096
     - 2.39
     - 0.03
     - 71x
   * - ``GaussianNoise.sample_at()``
     - 4097
     - 1.68
     - 0.03
     - 50x
   * - ``SquaredBesselProcess.sample_at()``
     - 4097
     - 2.35
     - 0.05
     - 45x
   * - ``BrownianMotion.sample_at()``
     - 4097
     - 1.76
     - 0.05
     - 38x
   * - ``BrownianExcursion.sample_at()``
     - 4097
     - 2.45
     - 0.07
     - 37x
   * - ``WienerProcess.sample_at()``
     - 4097
     - 1.69
     - 0.05
     - 36x
   * - ``BrownianMeander.sample_at()``
     - 4097
     - 5.31
     - 0.16
     - 33x
   * - ``GeometricBrownianMotion.sample_at()``
     - 4097
     - 1.77
     - 0.06
     - 31x
   * - ``BrownianBridge.sample_at()``
     - 4097
     - 1.70
     - 0.06
     - 31x
   * - ``MultifractionalBrownianMotion.sample()``
     - 512
     - 45.5
     - 1.59
     - 29x
   * - ``VarianceGammaProcess.sample_at()``
     - 4097
     - 3.34
     - 0.12
     - 28x
   * - ``GammaProcess.sample_at()``
     - 4097
     - 1.72
     - 0.08
     - 22x
   * - ``MoranProcess.sample()``
     - 4096
     - 8.70
     - 0.40
     - 22x
   * - ``SquaredBesselProcess.sample()``
     - 4096
     - 0.67
     - 0.03
     - 21x
   * - ``CauchyProcess.sample()``
     - 4096
     - 1.76
     - 0.09
     - 19x
   * - ``InverseGaussianProcess.sample()``
     - 4096
     - 0.95
     - 0.06
     - 15x
   * - ``BernoulliProcess.sample()``
     - 4096
     - 0.22
     - 0.02
     - 14x
   * - ``BrownianExcursion.sample()``
     - 4096
     - 0.60
     - 0.04
     - 14x
   * - ``PoissonProcess.sample()``
     - 4096
     - 0.13
     - 0.02
     - 5.4x
   * - ``MixedPoissonProcess.sample()``
     - 4096
     - 0.14
     - 0.03
     - 5.2x
   * - ``ChineseRestaurantProcess.sample_partition()``
     - 4096
     - 17.9
     - 3.48
     - 5.1x
   * - ``ChineseRestaurantProcess.sample()``
     - 4096
     - 17.3
     - 3.52
     - 4.9x
   * - ``RandomWalk.sample()``
     - 4096
     - 0.16
     - 0.06
     - 2.8x
   * - ``DirichletProcess.sample()``
     - 4096
     - 12.3
     - 5.01
     - 2.5x
   * - ``InverseGaussianProcess.sample_at()``
     - 4097
     - 1.64
     - 0.76
     - 2.2x

The largest gains come from three kinds of change:

* **Sampling at specified times.** ``sample_at()`` drew one normal variate
  per increment in a Python loop. It now draws all of them in one call, which
  makes it about 30 to 75 times faster for the Brownian motion and the
  processes built on it.
* **Discrete processes.** The Markov chain, the Moran process and the Chinese
  restaurant process called ``Generator.choice`` at every step. They now look
  up the next state in a precomputed cumulative distribution, using the same
  uniform variates.
* **Quadratic algorithms.** Hosking's method and the multifractional Brownian
  motion ran two nested Python loops. The inner loop is now a vector
  operation, which makes them about 70 and 30 times faster.

Methods that were already fast
------------------------------

The remaining methods did not change, because they were already vectorized:

* ``sample()`` of the Gaussian, fractional Gaussian and colored noises, and of
  the Brownian motion, Wiener process, Brownian bridge, Brownian meander,
  fractional Brownian motion, geometric Brownian motion, gamma process and
  variance gamma process takes between 0.02 and 0.11 milliseconds for 4096
  increments.
* ``sample()`` of the diffusion processes takes about 1 millisecond for 4096
  increments. The Euler-Maruyama method computes each value from the previous
  one and calls the parameter functions at every step, so it stays a Python
  loop.

Results do not change
---------------------

A faster method generates the same realization as before for a given seed.
This was checked by comparing 831 seeded realizations, covering every process
and sampling method, between version 0.1.1 and the current code:

* 781 are identical bit for bit.
* 48 agree to a relative difference of 1e-9. In these, a sum which used to be
  accumulated in a Python loop is now computed by ``numpy``, which adds the
  terms in a different order.
* 2, both from ``CauchyProcess.sample_at()``, agree to 2.5e-8. The method now
  builds the path from the increments of its random time change directly,
  which is more accurate than accumulating them and taking differences.

Getting the most out of it
--------------------------

* **Reuse instances.** Some processes cache values which depend only on the
  parameters and on ``n``, such as the frequencies of a colored noise or the
  mean increments of an inverse Gaussian process. Create the process once and
  call ``sample(n)`` repeatedly.
* **Prefer** ``sample()`` **to** ``sample_at()`` **for equally spaced times.**
  ``sample()`` skips the validation of the time values.
* **Use the default algorithm for fractional noise.** The Davies-Harte method
  used by ``sample()`` takes 0.06 milliseconds for 4096 increments. Hosking's
  method takes 3.3 milliseconds for 1024 increments and four times as long
  each time the number of increments doubles.
* **Keep** ``n`` **moderate for the multifractional Brownian motion.** Its
  cost also grows with the square of ``n``: 1.6 milliseconds for 512
  increments.
