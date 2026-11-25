Speed Against a Rust Core
=========================

``stochastic-rs`` 2.6.2 was measured against version 0.6.0 of
``samplepaths`` on the machine described in :doc:`/guide/performance`, which
has 14 cores, for the processes which both packages provide with the same
parameters. The times are in milliseconds, and the last column is the time
of ``samplepaths`` divided by the time of ``stochastic-rs``: a value below 1
means ``samplepaths`` is faster. Reproduce them with:

.. code-block:: bash

   uv run --with stochastic-rs python docs/scripts/compare_rust.py

One realization
---------------

One realization of 4096 increments.

.. csv-table::
   :header: Process, ``stochastic-rs``, ``samplepaths``, Ratio
   :widths: 40 20 20 20

   Fractional Gaussian noise, 0.061, 0.060, 1.0
   Fractional Brownian motion, 0.064, 0.071, 1.1
   Variance gamma process, 0.065, 0.081, 1.3
   Fractional Vasicek process, 0.074, 0.098, 1.3
   AR(2) process, 0.024, 0.035, 1.4
   "ARIMA(1, 1, 1) process", 0.024, 0.038, 1.6
   Gaussian noise, 0.007, 0.015, 2.1
   Geometric Brownian motion, 0.015, 0.040, 2.6
   Brownian motion, 0.010, 0.027, 2.9
   Correlated Brownian motions, 0.022, 0.063, 2.9
   Poisson process, 0.008, 0.026, 3.0
   SABR model, 0.109, 0.396, 3.6
   Fractional Cox-Ingersoll-Ross process, 0.089, 0.344, 3.9
   CEV process, 0.092, 0.446, 4.8
   "EGARCH(1, 1) process", 0.102, 0.627, 6.1
   "GARCH(1, 1) process", 0.043, 0.417, 9.7
   Heston model, 0.111, 1.188, 10.7
   Vasicek process, 0.017, 0.212, 12.8
   Cox-Ingersoll-Ross process, 0.033, 0.453, 13.7

For one realization, compiled code wins: by a factor of 1 to 4 where
``samplepaths`` calls ``numpy`` a few times, and by 4 to 14 for the processes
which advance one value at a time in Python: the diffusion processes, the
Heston model and the models with a changing variance. A realization of 4096
values takes about a millisecond or less. The Vasicek and the
Cox-Ingersoll-Ross processes also have exact methods; that of the Vasicek
process takes 0.06 milliseconds, 3 times the time of ``stochastic-rs``.

Many realizations
-----------------

Both packages can use all cores: ``stochastic-rs`` with ``sample_par()``,
``samplepaths`` with ``sample_paths(..., workers=14)``.

100,000 realizations of 64 increments:

.. csv-table::
   :header: Process, ``stochastic-rs``, ``samplepaths``, Ratio
   :widths: 40 20 20 20

   Correlated Brownian motions, 18.3, 10.0, 0.5
   Gaussian noise, 7.1, 4.0, 0.6
   Fractional Brownian motion, 17.0, 11.3, 0.7
   AR(2) process, 10.9, 8.3, 0.8
   Fractional Gaussian noise, 13.8, 10.8, 0.8
   Brownian motion, 6.5, 5.7, 0.9
   Poisson process, 6.8, 5.9, 0.9
   Variance gamma process, 14.5, 14.9, 1.0
   Fractional Vasicek process, 15.0, 15.6, 1.0
   "ARIMA(1, 1, 1) process", 8.2, 9.1, 1.1
   CEV process, 18.5, 20.3, 1.1
   Vasicek process, 7.8, 8.9, 1.1
   SABR model, 26.0, 30.4, 1.2
   Geometric Brownian motion, 7.6, 9.0, 1.2
   "GARCH(1, 1) process", 11.5, 16.0, 1.4
   Heston model, 28.3, 39.3, 1.4
   "EGARCH(1, 1) process", 19.3, 32.9, 1.7
   Fractional Cox-Ingersoll-Ross process, 17.0, 33.0, 1.9
   Cox-Ingersoll-Ross process, 9.9, 20.3, 2.0

20,000 realizations of 1024 increments:

.. csv-table::
   :header: Process, ``stochastic-rs``, ``samplepaths``, Ratio
   :widths: 40 20 20 20

   Fractional Brownian motion, 45.5, 37.6, 0.8
   Gaussian noise, 15.1, 12.4, 0.8
   Fractional Gaussian noise, 39.7, 34.9, 0.9
   Correlated Brownian motions, 36.4, 37.4, 1.0
   Poisson process, 15.5, 19.0, 1.2
   Variance gamma process, 50.9, 63.7, 1.3
   Vasicek process, 19.3, 26.0, 1.3
   Fractional Vasicek process, 46.2, 59.6, 1.3
   Brownian motion, 15.8, 21.5, 1.4
   Geometric Brownian motion, 18.7, 26.6, 1.4
   AR(2) process, 20.2, 32.8, 1.6
   "ARIMA(1, 1, 1) process", 22.3, 34.7, 1.6
   "GARCH(1, 1) process", 31.3, 62.0, 2.0
   "EGARCH(1, 1) process", 54.5, 121.2, 2.2
   SABR model, 76.1, 237.5, 3.1
   Heston model, 81.5, 255.7, 3.1
   CEV process, 55.3, 212.8, 3.9
   Fractional Cox-Ingersoll-Ross process, 52.8, 404.4, 7.7
   Cox-Ingersoll-Ross process, 27.2, 213.9, 7.9

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
