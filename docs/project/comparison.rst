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
     - 61 named processes in Python on ``numpy``; typed; reproducible
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
   * - Fractional Vasicek process
     - 0.074
     - 0.098
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
   * - Fractional Cox-Ingersoll-Ross process
     - 0.087
     - 0.702
     - 8.1
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
   * - Fractional Vasicek process
     - 15.0
     - 15.6
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
   * - Fractional Cox-Ingersoll-Ross process
     - 17.0
     - 33.0
     - 1.9
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
   * - Fractional Vasicek process
     - 46.2
     - 59.6
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
   * - Fractional Cox-Ingersoll-Ross process
     - 52.8
     - 404.4
     - 7.7
   * - Cox-Ingersoll-Ross process
     - 27.2
     - 213.9
     - 7.9

For large batches of short paths the two are close: ``samplepaths`` is
between 0.6 and 1.4 times the time of ``stochastic-rs`` for 13 of the 16
processes, and faster for six of them.

Long paths separate the processes by how they are computed. Those written as
array operations over time, the fractional Vasicek process among them, stay
between 0.8 and 1.4. The time series models generate their random numbers on
all threads but advance on one, and take 1.6 to 2.2 times as long. The CEV,
the Cox-Ingersoll-Ross and the fractional Cox-Ingersoll-Ross processes,
whose volatility depends on the value of the process, advance step by step
in Python and are 4 to 8 times slower.

For small batches the fixed cost of a call dominates. For 1000 realizations
of 256 increments ``samplepaths`` takes 0.5 to 4.2 milliseconds and is 1.4 to
5.4 times slower than ``stochastic-rs`` in parallel.

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
     - 0.006
     - -0.001
     - 1.00, 1.00
     - 0.29
   * - Geometric Brownian motion
     - final value
     - 1.050
     - 1.052
     - 0.21, 0.21
     - 0.51
   * - Fractional Brownian motion
     - final value
     - -0.001
     - 0.001
     - 1.00, 1.00
     - 0.88
   * - Vasicek process
     - final value
     - 0.610
     - 0.609
     - 0.17, 0.17
     - 0.86
   * - Cox-Ingersoll-Ross process
     - final value
     - 1.000
     - 1.002
     - 0.20, 0.20
     - 0.19
   * - CEV process
     - final value
     - 1.104
     - 1.107
     - 0.22, 0.22
     - 0.10
   * - Variance gamma process
     - final value
     - 0.093
     - 0.105
     - 1.01, 1.01
     - 0.34
   * - Poisson process
     - time of arrival 64
     - 32.04
     - 32.04
     - 4.01, 4.01
     - 0.95
   * - Jacobi process
     - final value
     - 0.474
     - 0.473
     - 0.12, 0.12
     - 0.30
   * - Verhulst process
     - final value
     - 2.302
     - 2.302
     - 0.42, 0.42
     - 0.63
   * - Gompertz process
     - final value
     - 2.188
     - 2.191
     - 0.35, 0.35
     - 0.25
   * - Kimura process
     - final value
     - 0.509
     - 0.509
     - 0.24, 0.24
     - 0.75
   * - Feller logistic process
     - final value
     - 1.742
     - 1.741
     - 0.23, 0.23
     - 0.30
   * - Normal inverse Gaussian process
     - final value
     - 0.100
     - 0.102
     - 0.50, 0.51
     - 0.45
   * - Merton jump diffusion
     - final logarithm
     - -0.098
     - -0.081
     - 0.58, 0.59
     - 0.11
   * - Compound Poisson process
     - value at time 2
     - -0.808
     - -0.806
     - 0.88, 0.90
     - 0.89
   * - Hawkes process
     - arrivals in [0, 20]
     - 79.0
     - 78.5
     - 17.5, 17.4
     - 0.14
   * - Fractional Vasicek process
     - final value
     - 0.609
     - 0.610
     - 0.16, 0.16
     - 0.14
   * - Fractional Ornstein-Uhlenbeck process
     - final value
     - 0.219
     - 0.219
     - 0.16, 0.16
     - 0.42
   * - Fractional Ornstein-Uhlenbeck process, H = 0.3
     - final value
     - 0.219
     - 0.220
     - 0.18, 0.18
     - 0.78
   * - Fractional Cox-Ingersoll-Ross process
     - final value
     - 1.010
     - 1.010
     - 0.19, 0.20
     - 0.98
   * - Fractional Jacobi process
     - final value
     - 0.476
     - 0.475
     - 0.11, 0.11
     - 0.28
   * - Fractional geometric Brownian motion
     - final logarithm
     - 0.049
     - 0.048
     - 0.20, 0.20
     - 0.93
   * - AR(2) process
     - value 300
     - -0.012
     - 0.005
     - 2.26, 2.28
     - 0.29
   * - MA(2) process
     - value 300
     - -0.005
     - 0.009
     - 1.78, 1.77
     - 0.49
   * - ARIMA(1, 1, 1) process
     - value 300
     - -0.0
     - -0.3
     - 45.1, 45.2
     - 0.29
   * - SARIMA(1, 0, 1)(1, 1, 1) process, period 4
     - value 300
     - 0.1
     - 0.0
     - 22.3, 22.3
     - 0.44
   * - ARCH(1) process
     - value 300
     - 0.001
     - 0.002
     - 1.00, 1.01
     - 0.51
   * - GARCH(1, 1) process
     - value 300
     - 0.003
     - 0.008
     - 0.99, 1.00
     - 0.42
   * - GARCH(2, 2) process
     - value 300
     - 0.001
     - -0.001
     - 1.00, 1.00
     - 0.80
   * - EGARCH(1, 1) process
     - value 300
     - -0.002
     - 0.000
     - 0.63, 0.62
     - 0.24
   * - Threshold GARCH(1, 1) process
     - value 300
     - -0.005
     - 0.001
     - 1.00, 1.01
     - 0.54
   * - AGARCH of ``stochastic-rs``, threshold GARCH here
     - value 300
     - -0.001
     - -0.005
     - 1.00, 1.00
     - 0.77

The test finds no difference in any of the 33 comparisons: the smallest
p-value of this run is 0.10. A run now and then gives one small p-value, as
33 tests should: an earlier one gave 0.008 for the CEV process, which
repeated three times with 400,000 realizations gave 0.92, 0.72 and 0.19,
with means of 1.1052 to 1.1056 from both packages against an exact mean of
1.1052. So the two packages agree for every process they share.

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
threshold GARCH. Its fractional geometric Brownian motion applies the Euler
method to the equation, where ``samplepaths`` returns the exact solution
with the Wick product; the logarithms of the two differ by a known term.

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
* **Coverage.** ``stochastic-rs`` has about 45 models which ``samplepaths``
  does not have yet: further jump and Lévy processes, stochastic volatility
  models, interest rate models and correlated processes. ``samplepaths`` has
  the Chinese restaurant, the Dirichlet and the Moran processes, the Markov
  chain, the Bessel processes, the Brownian bridge, excursion and meander,
  the colored noises, the multifractional Brownian motion and the asymmetric
  GARCH model with a shift, which ``stochastic-rs`` does not list.

Planned
-------

Three batches of missing models are in: the jump processes and the
diffusions on an interval or for a population, the time series models, and
the fractional models. Next are the stochastic volatility, interest rate and
correlated models, which have several components. Exact sampling for the
Cox-Ingersoll-Ross process, which would remove the step-by-step loop, is
planned with them.
