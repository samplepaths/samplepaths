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
     - 68 named processes in Python on ``numpy``; typed; reproducible
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

``stochastic-rs`` 2.6.2 was measured against version 0.5.0 of
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
   * - Correlated Brownian motions
     - 0.022
     - 0.063
     - 2.9
   * - Poisson process
     - 0.008
     - 0.026
     - 3.0
   * - SABR model
     - 0.109
     - 0.396
     - 3.6
   * - CEV process
     - 0.092
     - 0.446
     - 4.8
   * - Fractional Cox-Ingersoll-Ross process
     - 0.087
     - 0.702
     - 8.1
   * - Heston model
     - 0.111
     - 1.188
     - 10.7
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

For one realization, compiled code wins: by a factor of 1 to 4 where
``samplepaths`` calls ``numpy`` a few times, by 5 to 14 for the diffusion
processes and the Heston model, and by 29 to 53 for the models with a
changing variance. These two groups advance one value at a time in Python. A
realization of 4096 values takes about a millisecond or less, and 2 to 3
milliseconds for the models with a changing variance.

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
   * - Correlated Brownian motions
     - 18.3
     - 10.0
     - 0.5
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
   * - SABR model
     - 26.0
     - 30.4
     - 1.2
   * - Geometric Brownian motion
     - 7.6
     - 9.0
     - 1.2
   * - GARCH(1, 1) process
     - 11.5
     - 16.0
     - 1.4
   * - Heston model
     - 28.3
     - 39.3
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
   * - Correlated Brownian motions
     - 36.4
     - 37.4
     - 1.0
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
   * - SABR model
     - 76.1
     - 237.5
     - 3.1
   * - Heston model
     - 81.5
     - 255.7
     - 3.1
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
between 0.5 and 1.4 times the time of ``stochastic-rs`` for 16 of the 19
processes, and faster for seven of them. The other three take 1.7 to 2.0
times as long.

Long paths separate the processes by how they are computed. Those written as
array operations over time, the fractional Vasicek process and the correlated
Brownian motions among them, stay between 0.8 and 1.4. The time series
models, the Heston and the SABR models advance step by step with all
realizations together, and take 1.6 to 3.1 times as long. The CEV, the
Cox-Ingersoll-Ross and the fractional Cox-Ingersoll-Ross processes, whose
volatility depends on the value of the process, are 4 to 8 times slower with
the Euler-Maruyama method. For the Cox-Ingersoll-Ross process the exact
method of ``samplepaths`` is the better choice: it needs no fine grid, and
one step gives 100,000 exact final values in 1.6 milliseconds.

For small batches the fixed cost of a call dominates. For 1000 realizations
of 256 increments ``samplepaths`` takes 0.5 to 8.0 milliseconds and is 1.4 to
6.8 times slower than ``stochastic-rs`` in parallel.

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
     - 0.008
     - -0.001
     - 0.99, 1.00
     - 0.23
   * - Geometric Brownian motion
     - final value
     - 1.054
     - 1.052
     - 0.21, 0.21
     - 0.32
   * - Fractional Brownian motion
     - final value
     - -0.000
     - 0.001
     - 0.99, 1.00
     - 0.59
   * - Vasicek process
     - final value
     - 0.611
     - 0.609
     - 0.17, 0.17
     - 0.09
   * - Cox-Ingersoll-Ross process
     - final value
     - 1.000
     - 1.002
     - 0.20, 0.20
     - 0.18
   * - CEV process
     - final value
     - 1.106
     - 1.107
     - 0.22, 0.22
     - 0.40
   * - Variance gamma process
     - final value
     - 0.104
     - 0.105
     - 1.01, 1.01
     - 0.81
   * - Poisson process
     - time of arrival 64
     - 32.03
     - 32.04
     - 4.00, 4.01
     - 0.87
   * - Jacobi process
     - final value
     - 0.474
     - 0.473
     - 0.12, 0.12
     - 0.52
   * - Verhulst process
     - final value
     - 2.306
     - 2.302
     - 0.42, 0.42
     - 0.25
   * - Gompertz process
     - final value
     - 2.187
     - 2.191
     - 0.35, 0.35
     - 0.47
   * - Kimura process
     - final value
     - 0.509
     - 0.509
     - 0.24, 0.24
     - 0.23
   * - Feller logistic process
     - final value
     - 1.742
     - 1.741
     - 0.23, 0.23
     - 0.32
   * - Normal inverse Gaussian process
     - final value
     - 0.101
     - 0.102
     - 0.50, 0.51
     - 0.74
   * - Merton jump diffusion
     - final logarithm
     - -0.093
     - -0.081
     - 0.58, 0.59
     - 0.15
   * - Compound Poisson process
     - value at time 2
     - -0.813
     - -0.806
     - 0.89, 0.90
     - 0.33
   * - Hawkes process
     - arrivals in [0, 20]
     - 79.1
     - 78.5
     - 17.6, 17.4
     - 0.04
   * - Fractional Vasicek process
     - final value
     - 0.609
     - 0.610
     - 0.16, 0.16
     - 0.57
   * - Fractional Ornstein-Uhlenbeck process
     - final value
     - 0.220
     - 0.219
     - 0.16, 0.16
     - 0.56
   * - Fractional Ornstein-Uhlenbeck process, H = 0.3
     - final value
     - 0.218
     - 0.220
     - 0.18, 0.18
     - 0.03
   * - Fractional Cox-Ingersoll-Ross process
     - final value
     - 1.011
     - 1.010
     - 0.20, 0.20
     - 0.77
   * - Fractional Jacobi process
     - final value
     - 0.476
     - 0.475
     - 0.11, 0.11
     - 0.18
   * - Fractional geometric Brownian motion
     - final logarithm
     - 0.048
     - 0.048
     - 0.20, 0.20
     - 0.89
   * - Heston, price
     - final logarithm
     - 0.017
     - 0.017
     - 0.26, 0.26
     - 0.84
   * - Heston, variance
     - final value
     - 0.047
     - 0.047
     - 0.03, 0.03
     - 0.09
   * - Bates, price
     - final logarithm
     - -0.011
     - -0.008
     - 0.36, 0.37
     - 0.45
   * - SABR, forward
     - final value
     - 0.998
     - 0.999
     - 0.20, 0.20
     - 0.92
   * - SABR, volatility
     - final value
     - 0.200
     - 0.200
     - 0.08, 0.08
     - 0.52
   * - Correlated Brownian motions  sum of
     - final values
     - 0.009
     - 0.006
     - 1.79, 1.79
     - 0.95
   * - Cox-Ingersoll-Ross, exact
     - final value
     - 1.000
     - 1.000
     - 0.20, 0.20
     - 0.28
   * - Ho-Lee
     - final value
     - 0.506
     - 0.499
     - 0.31, 0.31
     - 0.87
   * - Hull-White two-factor, rate
     - final value
     - 0.279
     - 0.284
     - 0.16, 0.16
     - 0.19
   * - Hull-White two-factor, factor
     - final value
     - -0.001
     - -0.002
     - 0.14, 0.14
     - 0.24
   * - AR(2) process
     - value 300
     - 0.010
     - -0.007
     - 2.25, 2.29
     - 0.05
   * - MA(2) process
     - value 300
     - 0.015
     - -0.000
     - 1.77, 1.77
     - 0.17
   * - ARIMA(1, 1, 1) process
     - value 300
     - 0.3
     - -0.0
     - 44.8, 44.9
     - 0.44
   * - SARIMA(1, 0, 1)(1, 1, 1) process, period 4
     - value 300
     - 0.1
     - -0.0
     - 22.3, 22.2
     - 0.42
   * - ARCH(1) process
     - value 300
     - 0.007
     - 0.006
     - 1.01, 1.00
     - 0.81
   * - GARCH(1, 1) process
     - value 300
     - 0.002
     - 0.001
     - 1.01, 1.00
     - 0.96
   * - GARCH(2, 2) process
     - value 300
     - -0.005
     - 0.004
     - 1.00, 1.00
     - 0.28
   * - EGARCH(1, 1) process
     - value 300
     - 0.001
     - -0.001
     - 0.63, 0.63
     - 0.64
   * - Threshold GARCH(1, 1) process
     - value 300
     - 0.002
     - -0.002
     - 1.00, 1.00
     - 0.54
   * - AGARCH of ``stochastic-rs``, threshold GARCH here
     - value 300
     - 0.002
     - -0.005
     - 0.99, 1.00
     - 0.42

The test finds no difference in 40 of the 43 comparisons, and three have a
p-value between 0.03 and 0.05; among 43 tests, two below 0.05 are expected
by chance. Which comparisons come out low changes from run to run. The
lowest seen in any run was 0.0003 for the forward of the SABR model; with
two million realizations from each package the same comparison gives 0.71,
and twelve repetitions with new random numbers gave p-values from 0.05 to
0.94, the same spread as ``stochastic-rs`` compared with itself. The counts
of the Hawkes process, low in two runs, have means of 78.68 and 78.65 over
60,000 realizations against an exact mean of 78.67. So the two packages
agree for every process they share.

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
For the Heston and the Bates models ``stochastic-rs`` applies the Euler
method, so it is given 1024 steps, where ``samplepaths`` draws the variance
from its exact distribution and is given 64. The exact method of the
Cox-Ingersoll-Ross process, with 8 steps, is compared in the same way.

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
* **Coverage.** ``stochastic-rs`` has about 35 models which ``samplepaths``
  does not have: further jump and Lévy processes, rough and multi-factor
  stochastic volatility models, term-structure models of interest rates and
  correlated fractional processes. ``samplepaths`` has the Chinese
  restaurant, the Dirichlet and the Moran processes, the Markov chain, the
  Bessel processes, the Brownian bridge, excursion and meander, the colored
  noises, the multifractional Brownian motion, the asymmetric GARCH model
  with a shift, correlated Brownian motions in any number and exact sampling
  of the square-root processes, which ``stochastic-rs`` does not list.

What was added, and what is left
--------------------------------

The models which ``stochastic-rs`` had and ``samplepaths`` lacked were added
in five batches, each compared with ``stochastic-rs`` as above:

1. jump processes, and diffusions on an interval or for a population
2. time series models
3. fractional models
4. correlated Brownian motions and the Heston, Bates and SABR models
5. the Ho-Lee process, two-factor interest rate models and exact sampling
   of the Cox-Ingersoll-Ross process

Not planned for now are the models listed under coverage above, and what
``stochastic-rs`` offers beyond simulation: option pricing, calibration and
statistical tests.
