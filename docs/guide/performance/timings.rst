Sampling Times
==============

The time of the sampling methods of every process, measured as described in
:doc:`/guide/performance`.

The tables give, in milliseconds:

* ``sample()``: one realization of 4096 increments
* ``sample_at()``: one realization at 4097 time values, where supported
* ``sample_paths()``: 1000 realizations of 256 increments in one call
* Gain: how much faster that call is than 1000 calls to ``sample()``

Continuous-time
---------------

.. csv-table::
   :header: Process, ``sample()``, ``sample_at()``, ``sample_paths()``, Gain
   :widths: 40 15 15 15 15

   ``WienerProcess``, 0.028, 0.048, 1.43, 3.7x
   ``BrownianMotion``, 0.027, 0.048, 1.43, 3.7x
   ``GeometricBrownianMotion``, 0.040, 0.060, 2.02, 3.8x
   ``FractionalGeometricBrownianMotion``, 0.109, —, 3.93, 4.0x
   ``BrownianBridge``, 0.032, 0.060, 1.58, 4.4x
   ``BrownianExcursion``, 0.045, 0.064, 2.73, 3.4x
   ``BrownianMeander``, 0.11, 0.17, 5.12, 4.6x
   ``BesselProcess``, 0.034, 0.054, 1.65, 4.5x
   ``SquaredBesselProcess``, 0.033, 0.056, 1.52, 4.8x
   ``FractionalBrownianMotion``, 0.028, —, 1.42, 3.2x
   ``MultifractionalBrownianMotion``, 0.34 [#q]_, —, 18.9, 7.1x
   ``CauchyProcess``, 0.10, 0.11, 4.72, 4.9x
   ``GammaProcess``, 0.059, 0.082, 3.18, 1.9x
   ``VarianceGammaProcess``, 0.079, 0.12, 4.25, 2.0x
   ``InverseGaussianProcess``, 0.063, 0.77, 3.20, 3.7x
   ``NormalInverseGaussianProcess``, 0.058, 0.082, 3.50, 2.1x
   ``CompoundPoissonProcess``, 0.032, 0.051, 1.71, 3.2x
   ``MertonJumpDiffusion``, 0.077, 0.12, 4.27, 2.5x
   ``KouJumpDiffusion``, 0.060, 0.10, 3.28, 3.2x
   ``PoissonProcess``, 0.025, —, 1.41, 2.2x
   ``MixedPoissonProcess``, 0.026, —, 4.32, 1.0x
   ``HawkesProcess``, 1.90, —, 149, 1.0x

Diffusion
---------

.. csv-table::
   :header: Process, ``sample()``, ``sample_at()``, ``sample_paths()``, Gain
   :widths: 40 15 15 15 15

   ``DiffusionProcess``, 0.21, —, 1.75, 8.4x
   ``OrnsteinUhlenbeckProcess``, 0.21, —, 1.77, 8.3x
   ``VasicekProcess``, 0.21, —, 1.77, 8.3x
   ``ExtendedVasicekProcess``, 0.21, —, 1.76, 8.2x
   ``CoxIngersollRossProcess``, 0.45, —, 2.21, 13.6x
   ``ConstantElasticityVarianceProcess``, 0.21, —, 1.83, 8.3x
   ``JacobiProcess``, 0.37, —, 2.75, 9.3x
   ``KimuraProcess``, 0.37, —, 2.68, 9.4x
   ``VerhulstProcess``, 0.35, —, 2.62, 9.0x
   ``GompertzProcess``, 0.37, —, 3.17, 8.2x
   ``FellerLogisticProcess``, 0.36, —, 2.64, 9.4x
   ``HoLeeProcess``, 0.029, —, 1.54, 3.5x
   ``FractionalOrnsteinUhlenbeckProcess``, 0.102, —, 3.64, 4.3x
   ``FractionalVasicekProcess``, 0.105, —, 3.62, 4.3x
   ``FractionalCoxIngersollRossProcess``, 0.35, —, 4.17, 7.0x
   ``FractionalJacobiProcess``, 0.41, —, 4.66, 7.1x

Discrete-time
-------------

.. csv-table::
   :header: Process, ``sample()``, ``sample_at()``, ``sample_paths()``, Gain
   :widths: 40 15 15 15 15

   ``BernoulliProcess``, 0.015, —, 0.84, 2.7x
   ``RandomWalk``, 0.058, —, 2.92, 3.2x
   ``MarkovChain``, 0.22, —, 4.46, 3.8x
   ``MoranProcess``, 0.42, —, 48.9, 1.0x
   ``ChineseRestaurantProcess``, 3.66, —, 196, 1.0x
   ``DirichletProcess``, 5.16, —, 304, 1.0x

Time series
-----------

.. csv-table::
   :header: Process, ``sample()``, ``sample_at()``, ``sample_paths()``, Gain
   :widths: 40 15 15 15 15

   ``MovingAverageProcess``, 0.026, —, 1.04, 8.8x
   ``ARIMAProcess``, 0.028, —, 0.94, 11.5x
   ``AutoregressiveProcess``, 0.029, —, 1.14, 4.7x
   ``SARIMAProcess``, 0.031, —, 0.95, 15.0x
   ``ARCHProcess``, 0.40, —, 1.90, 13.9x
   ``GARCHProcess``, 0.42, —, 2.12, 13.4x
   ``AGARCHProcess``, 0.45, —, 2.41, 12.3x
   ``TGARCHProcess``, 0.49, —, 2.69, 12.2x
   ``EGARCHProcess``, 0.63, —, 3.53, 11.8x

The linear models filter white noise in compiled code and cost little more
than the noise itself. The models with a changing variance advance one value
at a time in Python, at about a tenth of a microsecond per value for a model
with one previous value and half a microsecond for higher orders.
``sample_paths()`` advances all realizations together.

Several components
------------------

.. csv-table::
   :header: Process, ``sample()``, ``sample_at()``, ``sample_paths()``, Gain
   :widths: 40 15 15 15 15

   ``CorrelatedBrownianMotions``, 0.058, —, 3.11, 2.1x
   ``HullWhiteTwoFactorModel``, 0.110, —, 4.04, 4.7x
   ``SABRModel``, 0.40, —, 4.49, 7.2x
   ``HestonModel``, 1.20, —, 8.16, 9.9x
   ``BatesModel``, 1.24, —, 10.7, 8.1x
   ``TwoFactorCoxIngersollRossModel``, 2.23, —, 10.2, 14.0x

A realization of these processes holds two or three components, so more
values than in the other tables. The Heston and the Bates models draw the variance from
a noncentral chi-square distribution at every step, which costs more than a
normal variate and buys a variance which is exact for any number of steps.


Noise
-----

.. csv-table::
   :header: Process, ``sample()``, ``sample_at()``, ``sample_paths()``, Gain
   :widths: 40 15 15 15 15

   ``GaussianNoise``, 0.016, 0.034, 0.91, 2.1x
   ``FractionalGaussianNoise``, 0.016, —, 0.91, 2.1x
   ``ColoredNoise``, 0.058, —, 2.22, 6.8x
   ``VioletNoise``, 0.057, —, 2.24, 7.0x
   ``BlueNoise``, 0.059, —, 2.22, 6.8x
   ``WhiteNoise``, 0.057, —, 2.25, 6.8x
   ``PinkNoise``, 0.058, —, 2.23, 6.7x
   ``RedNoise``, 0.057, —, 2.25, 6.9x
   ``BrownianNoise``, 0.058, —, 2.22, 6.8x


.. [#q] For 512 increments. The cost of this algorithm grows with the square
   of the number of increments.

Three things stand out:

* **One realization is cheap.** For most processes, 4096 increments take well
  under a tenth of a millisecond.
* **Many realizations are cheaper together.** ``sample_paths()`` is 2 to 20
  times faster than a loop over ``sample()`` for the processes which can
  generate all realizations in one pass. See :doc:`/guide/paths`.
* **Some processes are sequential by nature.** The Chinese restaurant, the
  Dirichlet, the Moran and the Hawkes processes compute every step from the
  previous ones, with a random decision at each. They run in a Python loop, and
  ``sample_paths()`` gives no gain for them.
