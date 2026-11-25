Performance
===========

``samplepaths`` generates realizations with ``numpy`` operations wherever the
algorithm allows it. This page gives measured sampling times and explains
how to get the most out of the library. For how the times changed between
versions, see the :doc:`/project/changelog`; for how they compare with a
package written in Rust, see :doc:`/project/comparison`.

How the times were measured
---------------------------

All times on this page are the mean time of one call, measured with
``docs/scripts/benchmark.py`` on one machine:

* Apple M4 Pro, macOS 27.0
* Python 3.13.2, ``numpy`` 2.4.1, ``scipy`` 1.17.0

Absolute times depend on the machine. Run the benchmark on your own with:

.. code-block:: bash

   make benchmark                                        # one realization
   uv run python docs/scripts/benchmark.py --paths       # many realizations

Times per process
-----------------

The tables give, in milliseconds:

* ``sample()``: one realization of 4096 increments
* ``sample_at()``: one realization at 4097 time values, where supported
* ``sample_paths()``: 1000 realizations of 256 increments in one call
* Gain: how much faster that call is than 1000 calls to ``sample()``

Continuous-time
~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 40 15 15 15 15

   * - Process
     - ``sample()``
     - ``sample_at()``
     - ``sample_paths()``
     - Gain
   * - ``WienerProcess``
     - 0.028
     - 0.048
     - 1.43
     - 3.7x
   * - ``BrownianMotion``
     - 0.027
     - 0.048
     - 1.43
     - 3.7x
   * - ``GeometricBrownianMotion``
     - 0.040
     - 0.060
     - 2.02
     - 3.8x
   * - ``FractionalGeometricBrownianMotion``
     - 0.109
     - —
     - 3.93
     - 4.0x
   * - ``BrownianBridge``
     - 0.032
     - 0.060
     - 1.58
     - 4.4x
   * - ``BrownianExcursion``
     - 0.045
     - 0.064
     - 2.73
     - 3.4x
   * - ``BrownianMeander``
     - 0.11
     - 0.17
     - 5.12
     - 4.6x
   * - ``BesselProcess``
     - 0.034
     - 0.054
     - 1.65
     - 4.5x
   * - ``SquaredBesselProcess``
     - 0.033
     - 0.056
     - 1.52
     - 4.8x
   * - ``FractionalBrownianMotion``
     - 0.028
     - —
     - 1.42
     - 3.2x
   * - ``MultifractionalBrownianMotion``
     - 0.34 [#q]_
     - —
     - 18.9
     - 7.1x
   * - ``CauchyProcess``
     - 0.10
     - 0.11
     - 4.72
     - 4.9x
   * - ``GammaProcess``
     - 0.059
     - 0.082
     - 3.18
     - 1.9x
   * - ``VarianceGammaProcess``
     - 0.079
     - 0.12
     - 4.25
     - 2.0x
   * - ``InverseGaussianProcess``
     - 0.063
     - 0.77
     - 3.20
     - 3.7x
   * - ``NormalInverseGaussianProcess``
     - 0.058
     - 0.082
     - 3.50
     - 2.1x
   * - ``CompoundPoissonProcess``
     - 0.032
     - 0.051
     - 1.71
     - 3.2x
   * - ``MertonJumpDiffusion``
     - 0.077
     - 0.12
     - 4.27
     - 2.5x
   * - ``KouJumpDiffusion``
     - 0.060
     - 0.10
     - 3.28
     - 3.2x
   * - ``PoissonProcess``
     - 0.025
     - —
     - 1.41
     - 2.2x
   * - ``MixedPoissonProcess``
     - 0.026
     - —
     - 4.32
     - 1.0x
   * - ``HawkesProcess``
     - 1.90
     - —
     - 149
     - 1.0x

Diffusion
~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 40 15 15 15 15

   * - Process
     - ``sample()``
     - ``sample_at()``
     - ``sample_paths()``
     - Gain
   * - ``DiffusionProcess``
     - 0.21
     - —
     - 1.75
     - 8.4x
   * - ``OrnsteinUhlenbeckProcess``
     - 0.21
     - —
     - 1.77
     - 8.3x
   * - ``VasicekProcess``
     - 0.21
     - —
     - 1.77
     - 8.3x
   * - ``ExtendedVasicekProcess``
     - 0.21
     - —
     - 1.76
     - 8.2x
   * - ``CoxIngersollRossProcess``
     - 0.45
     - —
     - 2.21
     - 13.6x
   * - ``ConstantElasticityVarianceProcess``
     - 0.21
     - —
     - 1.83
     - 8.3x
   * - ``JacobiProcess``
     - 0.76
     - —
     - 2.68
     - 18.4x
   * - ``KimuraProcess``
     - 0.72
     - —
     - 2.54
     - 18.7x
   * - ``VerhulstProcess``
     - 0.80
     - —
     - 2.58
     - 20.3x
   * - ``GompertzProcess``
     - 0.86
     - —
     - 3.02
     - 17.9x
   * - ``FellerLogisticProcess``
     - 0.81
     - —
     - 2.55
     - 20.6x
   * - ``FractionalOrnsteinUhlenbeckProcess``
     - 0.102
     - —
     - 3.64
     - 4.3x
   * - ``FractionalVasicekProcess``
     - 0.105
     - —
     - 3.62
     - 4.3x
   * - ``FractionalCoxIngersollRossProcess``
     - 0.73
     - —
     - 4.18
     - 12.2x
   * - ``FractionalJacobiProcess``
     - 0.85
     - —
     - 4.58
     - 12.8x

Discrete-time
~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 40 15 15 15 15

   * - Process
     - ``sample()``
     - ``sample_at()``
     - ``sample_paths()``
     - Gain
   * - ``BernoulliProcess``
     - 0.015
     - —
     - 0.84
     - 2.7x
   * - ``RandomWalk``
     - 0.058
     - —
     - 2.92
     - 3.2x
   * - ``MarkovChain``
     - 0.22
     - —
     - 4.46
     - 3.8x
   * - ``MoranProcess``
     - 0.42
     - —
     - 48.9
     - 1.0x
   * - ``ChineseRestaurantProcess``
     - 3.66
     - —
     - 196
     - 1.0x
   * - ``DirichletProcess``
     - 5.16
     - —
     - 304
     - 1.0x

Time series
~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 40 15 15 15 15

   * - Process
     - ``sample()``
     - ``sample_at()``
     - ``sample_paths()``
     - Gain
   * - ``MovingAverageProcess``
     - 0.026
     - —
     - 1.04
     - 8.8x
   * - ``ARIMAProcess``
     - 0.028
     - —
     - 0.94
     - 11.5x
   * - ``AutoregressiveProcess``
     - 0.029
     - —
     - 1.14
     - 4.7x
   * - ``SARIMAProcess``
     - 0.031
     - —
     - 0.95
     - 15.0x
   * - ``ARCHProcess``
     - 1.50
     - —
     - 1.89
     - 52x
   * - ``GARCHProcess``
     - 2.23
     - —
     - 2.11
     - 68x
   * - ``AGARCHProcess``
     - 2.37
     - —
     - 2.42
     - 63x
   * - ``TGARCHProcess``
     - 2.53
     - —
     - 2.57
     - 62x
   * - ``EGARCHProcess``
     - 2.80
     - —
     - 3.39
     - 51x

The linear models filter white noise in compiled code and cost little more
than the noise itself. The models with a changing variance advance one value
at a time in Python, at about half a microsecond per value, so one
realization is slow and ``sample_paths()``, which advances all realizations
together, gains the most of all processes.

Several components
~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 40 15 15 15 15

   * - Process
     - ``sample()``
     - ``sample_at()``
     - ``sample_paths()``
     - Gain
   * - ``CorrelatedBrownianMotions``
     - 0.058
     - —
     - 3.11
     - 2.1x
   * - ``SABRModel``
     - 0.40
     - —
     - 4.49
     - 7.2x
   * - ``HestonModel``
     - 1.20
     - —
     - 8.16
     - 9.9x
   * - ``BatesModel``
     - 1.24
     - —
     - 10.7
     - 8.1x

A realization of these processes holds two components, so twice the values
of the other tables. The Heston and the Bates models draw the variance from
a noncentral chi-square distribution at every step, which costs more than a
normal variate and buys a variance which is exact for any number of steps.


Noise
~~~~~

.. list-table::
   :header-rows: 1
   :widths: 40 15 15 15 15

   * - Process
     - ``sample()``
     - ``sample_at()``
     - ``sample_paths()``
     - Gain
   * - ``GaussianNoise``
     - 0.016
     - 0.034
     - 0.91
     - 2.1x
   * - ``FractionalGaussianNoise``
     - 0.016
     - —
     - 0.91
     - 2.1x
   * - ``ColoredNoise``
     - 0.058
     - —
     - 2.22
     - 6.8x
   * - ``VioletNoise``
     - 0.057
     - —
     - 2.24
     - 7.0x
   * - ``BlueNoise``
     - 0.059
     - —
     - 2.22
     - 6.8x
   * - ``WhiteNoise``
     - 0.057
     - —
     - 2.25
     - 6.8x
   * - ``PinkNoise``
     - 0.058
     - —
     - 2.23
     - 6.7x
   * - ``RedNoise``
     - 0.057
     - —
     - 2.25
     - 6.9x
   * - ``BrownianNoise``
     - 0.058
     - —
     - 2.22
     - 6.8x


.. [#q] For 512 increments. The cost of this algorithm grows with the square
   of the number of increments.

Three things stand out:

* **One realization is cheap.** For most processes, 4096 increments take well
  under a tenth of a millisecond.
* **Many realizations are cheaper together.** ``sample_paths()`` is 2 to 20
  times faster than a loop over ``sample()`` for the processes which can
  generate all realizations in one pass. See :doc:`paths`.
* **Some processes are sequential by nature.** The Chinese restaurant, the
  Dirichlet, the Moran and the Hawkes processes compute every step from the
  previous ones, with a random decision at each. They run in a Python loop, and
  ``sample_paths()`` gives no gain for them.

Algorithms with a quadratic cost
--------------------------------

Two algorithms take a time which grows with the square of the number of
increments ``n``. Both keep the part which does not depend on the random
numbers, so the first realization from an instance is slower than the next
ones:

.. list-table::
   :header-rows: 1
   :widths: 46 18 18 18

   * - Method
     - Increments
     - First call
     - Next calls
   * - ``MultifractionalBrownianMotion.sample()``
     - 512
     - 2.6 ms
     - 0.34 ms
   * - ``FractionalGaussianNoise.sample_hosking()``
     - 1024
     - 3.7 ms
     - 0.98 ms

The values which are kept take ``4 n**2`` bytes, 17 MB for 2048 increments.
They are not kept above 2048 increments, where every call takes the time of
a first call. They are computed again when a parameter of the process or
``n`` changes.

For fractional Gaussian noise and fractional Brownian motion, prefer the
default Davies-Harte method: ``sample()`` takes 0.06 milliseconds for 4096
increments.

Getting the most out of it
--------------------------

* **Use** ``sample_paths()`` **for many realizations.** It avoids the cost of
  a Python call per realization, which dominates for short paths.
* **Use** ``workers`` **for large batches.** With 14 threads, 100,000
  realizations of 64 increments of a Brownian motion take 5.7 instead of 33.8
  milliseconds. See :doc:`paths`.
* **Simulate the models with a changing variance in batches.** One
  realization of 4096 values of a GARCH process takes 2.2 milliseconds, and
  1000 realizations of 256 values take 2.1.
* **Reuse instances.** Several processes keep values which depend only on
  the parameters and on ``n``: the two above, the frequencies of a colored
  noise and the mean increments of an inverse Gaussian process. Create the
  process once and sample from it repeatedly.
* **Prefer** ``sample()`` **to** ``sample_at()`` **for equally spaced times.**
  ``sample()`` skips the validation of the time values.
* **Pass numbers for constant diffusion parameters.** A parameter given as a
  function is called at every step. For 4096 increments, a diffusion process
  takes 0.21 milliseconds with constant parameters, 0.51 milliseconds with
  two parameters given as functions and 0.79 milliseconds with all four.
* **Mind the number of tables of a Chinese restaurant process.** Its cost
  grows with the number of occupied tables. 4096 customers take 3.6
  milliseconds with the default parameters (about 9 tables), 12 milliseconds
  with ``discount=0.5`` and ``strength=2`` (about 200 tables) and 25
  milliseconds with ``discount=0.9`` and ``strength=5`` (about 2500 tables).

Speed and reproducibility
-------------------------

Making the library faster never changes the realizations of ``sample()`` and
``sample_at()`` for a given seed beyond floating point rounding. Each release
is checked against the previous one on seeded realizations of every process
and sampling method; the :doc:`/project/changelog` gives the result for each
version.

``sample_paths()`` draws its random numbers in a different order, so its
realizations are not those of repeated calls to ``sample()`` with the same
seed. They have the same distribution, and a given seed always gives the
same result.
