Comparison with Other Packages
==============================

``samplepaths`` is written in Python and depends on ``numpy`` and ``scipy``
only. This page says where that puts it next to the other packages which
simulate stochastic processes, with measured times where a comparison is
possible.

Where it fits
-------------

.. list-table::
   :header-rows: 1
   :widths: 22 39 39

   * - Package
     - What it is
     - Choose it when
   * - ``samplepaths``
     - 56 named processes in Python on ``numpy``; typed; reproducible
       results checked between versions
     - you want readable, dependency-light code, exact reproducibility, or a
       maintained replacement for ``stochastic``
   * - `stochastic <https://github.com/crflynn/stochastic>`_
     - the package ``samplepaths`` was forked from; last released in 2022 and
       limited to Python 3.10 and ``numpy`` 1
     - you cannot change an existing environment; otherwise see
       :doc:`migration`
   * - `stochastic-rs <https://pypi.org/project/stochastic-rs/>`_
     - about 85 processes with a Rust core, plus option pricing, calibration
       and statistics for quantitative finance
     - you need its additional models or the fastest single realization
   * - `aleatory <https://pypi.org/project/aleatory/>`_
     - processes with built-in plots and analytical quantities
     - you want charts and expectations out of the box
   * - `diffrax <https://pypi.org/project/diffrax/>`_,
       `torchsde <https://pypi.org/project/torchsde/>`_
     - general solvers for differential equations on GPUs, with gradients
     - your model is your own equation, or you need gradients

``samplepaths`` simulates named processes. It is not a solver for arbitrary
stochastic differential equations.

Speed against a Rust core
-------------------------

``stochastic-rs`` 2.6.2 was measured against the development version of
``samplepaths`` on the machine described in :doc:`/guide/performance`, which
has 14 cores, for the processes which both packages provide with the same
parameters. The times are in milliseconds, and the last column is the time
of ``samplepaths`` divided by the time of ``stochastic-rs``: a value below 1
means ``samplepaths`` is faster. Reproduce them with:

.. code-block:: bash

   uv run --with stochastic-rs python docs/scripts/compare_rust.py

One realization
~~~~~~~~~~~~~~~

One realization of 4096 increments.

.. list-table::
   :header-rows: 1
   :widths: 40 20 20 20

   * - Process
     - ``stochastic-rs``
     - ``samplepaths``
     - Ratio
   * - Fractional Gaussian noise
     - 0.061
     - 0.060
     - 1.0
   * - Fractional Brownian motion
     - 0.064
     - 0.071
     - 1.1
   * - Variance gamma process
     - 0.065
     - 0.081
     - 1.3
   * - AR(2) process
     - 0.024
     - 0.035
     - 1.4
   * - ARIMA(1, 1, 1) process
     - 0.024
     - 0.038
     - 1.6
   * - Gaussian noise
     - 0.007
     - 0.015
     - 2.1
   * - Geometric Brownian motion
     - 0.015
     - 0.040
     - 2.6
   * - Brownian motion
     - 0.010
     - 0.027
     - 2.9
   * - Poisson process
     - 0.008
     - 0.026
     - 3.0
   * - CEV process
     - 0.092
     - 0.446
     - 4.8
   * - Vasicek process
     - 0.017
     - 0.212
     - 12.8
   * - Cox-Ingersoll-Ross process
     - 0.033
     - 0.453
     - 13.7
   * - EGARCH(1, 1) process
     - 0.093
     - 2.699
     - 29.0
   * - GARCH(1, 1) process
     - 0.041
     - 2.157
     - 53.2

For one realization, compiled code wins: by a factor of 1 to 3 where
``samplepaths`` calls ``numpy`` a few times, by up to 14 for the diffusion
processes, and by 29 to 53 for the models with a changing variance. These
two groups advance one value at a time in Python. A realization of 4096
values takes less than half a millisecond, and 2 to 3 milliseconds for the
models with a changing variance.

Many realizations
~~~~~~~~~~~~~~~~~

Both packages can use all cores: ``stochastic-rs`` with ``sample_par()``,
``samplepaths`` with ``sample_paths(..., workers=14)``.

100,000 realizations of 64 increments:

.. list-table::
   :header-rows: 1
   :widths: 40 20 20 20

   * - Process
     - ``stochastic-rs``
     - ``samplepaths``
     - Ratio
   * - Gaussian noise
     - 7.1
     - 4.0
     - 0.6
   * - Fractional Brownian motion
     - 17.0
     - 11.3
     - 0.7
   * - AR(2) process
     - 10.9
     - 8.3
     - 0.8
   * - Fractional Gaussian noise
     - 13.8
     - 10.8
     - 0.8
   * - Brownian motion
     - 6.5
     - 5.7
     - 0.9
   * - Poisson process
     - 6.8
     - 5.9
     - 0.9
   * - Variance gamma process
     - 14.5
     - 14.9
     - 1.0
   * - ARIMA(1, 1, 1) process
     - 8.2
     - 9.1
     - 1.1
   * - CEV process
     - 18.5
     - 20.3
     - 1.1
   * - Vasicek process
     - 7.8
     - 8.9
     - 1.1
   * - Geometric Brownian motion
     - 7.6
     - 9.0
     - 1.2
   * - GARCH(1, 1) process
     - 11.5
     - 16.0
     - 1.4
   * - EGARCH(1, 1) process
     - 19.3
     - 32.9
     - 1.7
   * - Cox-Ingersoll-Ross process
     - 9.9
     - 20.3
     - 2.0

20,000 realizations of 1024 increments:

.. list-table::
   :header-rows: 1
   :widths: 40 20 20 20

   * - Process
     - ``stochastic-rs``
     - ``samplepaths``
     - Ratio
   * - Fractional Brownian motion
     - 45.5
     - 37.6
     - 0.8
   * - Gaussian noise
     - 15.1
     - 12.4
     - 0.8
   * - Fractional Gaussian noise
     - 39.7
     - 34.9
     - 0.9
   * - Poisson process
     - 15.5
     - 19.0
     - 1.2
   * - Variance gamma process
     - 50.9
     - 63.7
     - 1.3
   * - Vasicek process
     - 19.3
     - 26.0
     - 1.3
   * - Brownian motion
     - 15.8
     - 21.5
     - 1.4
   * - Geometric Brownian motion
     - 18.7
     - 26.6
     - 1.4
   * - AR(2) process
     - 20.2
     - 32.8
     - 1.6
   * - ARIMA(1, 1, 1) process
     - 22.3
     - 34.7
     - 1.6
   * - GARCH(1, 1) process
     - 31.3
     - 62.0
     - 2.0
   * - EGARCH(1, 1) process
     - 54.5
     - 121.2
     - 2.2
   * - CEV process
     - 55.3
     - 212.8
     - 3.9
   * - Cox-Ingersoll-Ross process
     - 27.2
     - 213.9
     - 7.9

For large batches of short paths the two are close: ``samplepaths`` is
between 0.6 and 1.4 times the time of ``stochastic-rs`` for 12 of the 14
processes, and faster for six of them.

Long paths separate the processes by how they are computed. Those written as
array operations over time stay between 0.8 and 1.4. The time series models
generate their random numbers on all threads but advance on one, and take
1.6 to 2.2 times as long. The CEV and the Cox-Ingersoll-Ross processes,
whose volatility depends on the value of the process, advance step by step
in Python and are 4 to 8 times slower.

For small batches the fixed cost of a call dominates. For 1000 realizations
of 256 increments ``samplepaths`` takes 0.5 to 3.0 milliseconds and is 1.4 to
4.9 times slower than ``stochastic-rs`` in parallel.

Do the two packages agree?
--------------------------

The two packages use different random number generators, so a seed does not
give them the same realization. What can be compared is the distribution of
their results. For each of the processes which both provide, 40,000
realizations were generated with each package (4000 for the three processes
generated one realization at a time), and one number per realization was
compared with a two-sample Kolmogorov-Smirnov test. The processes on a time
interval have 64 increments; the time series have 300 values, enough to
forget their start, which the two packages choose differently.
Reproduce it with:

.. code-block:: bash

   uv run --with stochastic-rs python docs/scripts/compare_rust_outputs.py

.. list-table::
   :header-rows: 1
   :widths: 30 22 12 12 12 12

   * - Process
     - Compared
     - Mean, ``stochastic-rs``
     - Mean, ``samplepaths``
     - Standard deviation, both
     - p-value
   * - Brownian motion
     - final value
     - 0.000
     - -0.001
     - 1.00, 1.00
     - 0.76
   * - Geometric Brownian motion
     - final value
     - 1.052
     - 1.052
     - 0.21, 0.21
     - 0.70
   * - Fractional Brownian motion
     - final value
     - -0.008
     - 0.001
     - 1.00, 1.00
     - 0.60
   * - Vasicek process
     - final value
     - 0.610
     - 0.609
     - 0.17, 0.17
     - 0.22
   * - Cox-Ingersoll-Ross process
     - final value
     - 1.000
     - 1.002
     - 0.20, 0.20
     - 0.31
   * - CEV process
     - final value
     - 1.104
     - 1.107
     - 0.21, 0.22
     - 0.010
   * - Variance gamma process
     - final value
     - 0.094
     - 0.105
     - 1.01, 1.01
     - 0.38
   * - Poisson process
     - time of arrival 64
     - 31.98
     - 32.04
     - 4.03, 4.01
     - 0.05
   * - Jacobi process
     - final value
     - 0.475
     - 0.473
     - 0.12, 0.12
     - 0.06
   * - Verhulst process
     - final value
     - 2.305
     - 2.302
     - 0.42, 0.42
     - 0.49
   * - Gompertz process
     - final value
     - 2.187
     - 2.191
     - 0.35, 0.35
     - 0.65
   * - Kimura process
     - final value
     - 0.510
     - 0.509
     - 0.24, 0.24
     - 0.96
   * - Feller logistic process
     - final value
     - 1.740
     - 1.741
     - 0.23, 0.23
     - 0.57
   * - Normal inverse Gaussian process
     - final value
     - 0.102
     - 0.102
     - 0.50, 0.51
     - 0.11
   * - Merton jump diffusion
     - final logarithm
     - -0.095
     - -0.081
     - 0.58, 0.59
     - 0.26
   * - Compound Poisson process
     - value at time 2
     - -0.809
     - -0.806
     - 0.88, 0.90
     - 0.42
   * - Hawkes process
     - arrivals in [0, 20]
     - 79.0
     - 78.5
     - 17.7, 17.4
     - 0.29
   * - AR(2) process
     - value 300
     - 0.011
     - -0.013
     - 2.28, 2.28
     - 0.06
   * - MA(2) process
     - value 300
     - -0.013
     - 0.002
     - 1.78, 1.76
     - 0.17
   * - ARIMA(1, 1, 1) process
     - value 300
     - 0.0
     - 0.2
     - 44.5, 44.9
     - 0.41
   * - SARIMA(1, 0, 1)(1, 1, 1) process, period 4
     - value 300
     - -0.1
     - -0.0
     - 22.2, 22.2
     - 0.74
   * - ARCH(1) process
     - value 300
     - 0.006
     - -0.000
     - 0.99, 1.00
     - 0.43
   * - GARCH(1, 1) process
     - value 300
     - -0.001
     - -0.001
     - 0.99, 1.01
     - 0.39
   * - GARCH(2, 2) process
     - value 300
     - -0.002
     - -0.002
     - 1.00, 1.00
     - 0.99
   * - EGARCH(1, 1) process
     - value 300
     - 0.003
     - -0.004
     - 0.63, 0.63
     - 0.33
   * - Threshold GARCH(1, 1) process
     - value 300
     - 0.002
     - 0.006
     - 0.99, 1.00
     - 0.43
   * - AGARCH of ``stochastic-rs``, threshold GARCH here
     - value 300
     - -0.007
     - 0.001
     - 0.99, 1.00
     - 0.17

The test finds no difference in 26 of the 27 comparisons. The CEV process
has a small p-value in this run; with 27 tests, one that small is not
unusual. Repeated three times with 400,000 realizations, the same comparison
gave p-values of 0.92, 0.72 and 0.19, with means of 1.1052 to 1.1056 from
both packages against an exact mean of 1.1052. So the two packages agree for
every process they share.

The comparison was also useful in the other direction. Its first run showed
two processes of ``samplepaths`` to be wrong, both inherited from the package
it was forked from: the variance gamma process had a spread which depended on
the number of increments, and the CEV process had the drift
:math:`\mu (X_t - 1)` instead of :math:`\mu X_t`. Both are fixed; see the
:doc:`changelog`.

The parameters are not always named or defined alike. The script documents
the mapping for each process: ``stochastic-rs`` counts the points of a
realization where ``samplepaths`` counts its increments, simulates the
logarithm of a jump diffusion where ``samplepaths`` returns the price,
writes the normal inverse Gaussian process with a variance of its clock where
``samplepaths`` uses the usual :math:`\alpha`, :math:`\beta` and
:math:`\delta`, and calls AGARCH the model which ``samplepaths`` calls
threshold GARCH.

What the numbers do not show
----------------------------

* **Reproducibility.** A seed gives ``samplepaths`` the same realizations on
  every run, on one thread or several. ``stochastic-rs`` documents that the
  rows of its parallel sampling are not reproducible across calls under a
  seed.
* **Installation.** ``samplepaths`` is a pure Python wheel which runs
  wherever ``numpy`` and ``scipy`` do. A package with a compiled core needs a
  wheel for the platform and the Python version, or a Rust toolchain.
* **Reading the algorithm.** Every process is one Python module.
* **Coverage.** ``stochastic-rs`` has about 50 models which ``samplepaths``
  does not have yet: further jump and Lévy processes, stochastic volatility
  models, interest rate models, fractional diffusions and correlated
  processes. ``samplepaths`` has the Chinese restaurant, the Dirichlet and
  the Moran processes, the Markov chain, the Bessel processes, the Brownian
  bridge, excursion and meander, the colored noises, the multifractional
  Brownian motion and the asymmetric GARCH model with a shift, which
  ``stochastic-rs`` does not list.

Planned
-------

Two batches of missing models are in: the jump processes and the diffusions
on an interval or for a population, then the time series models. The others
are being added in this order: fractional models, then stochastic volatility,
interest rate and correlated models, which have several components. Exact
sampling for the Cox-Ingersoll-Ross process, which would remove the
step-by-step loop, is planned with them.
