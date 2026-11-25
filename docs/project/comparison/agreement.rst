Agreement with stochastic-rs
============================

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

.. csv-table::
   :header: Process, Compared, "Mean, ``stochastic-rs``", "Mean, ``samplepaths``", "Standard deviation, both", p-value
   :widths: 30 22 12 12 12 12

   Brownian motion, final value, 0.008, -0.001, "0.99, 1.00", 0.23
   Geometric Brownian motion, final value, 1.054, 1.052, "0.21, 0.21", 0.32
   Fractional Brownian motion, final value, -0.000, 0.001, "0.99, 1.00", 0.59
   Vasicek process, final value, 0.611, 0.609, "0.17, 0.17", 0.09
   Cox-Ingersoll-Ross process, final value, 1.000, 1.002, "0.20, 0.20", 0.18
   CEV process, final value, 1.106, 1.107, "0.22, 0.22", 0.40
   Variance gamma process, final value, 0.104, 0.105, "1.01, 1.01", 0.81
   Poisson process, time of arrival 64, 32.03, 32.04, "4.00, 4.01", 0.87
   Jacobi process, final value, 0.474, 0.473, "0.12, 0.12", 0.52
   Verhulst process, final value, 2.306, 2.302, "0.42, 0.42", 0.25
   Gompertz process, final value, 2.187, 2.191, "0.35, 0.35", 0.47
   Kimura process, final value, 0.509, 0.509, "0.24, 0.24", 0.23
   Feller logistic process, final value, 1.742, 1.741, "0.23, 0.23", 0.32
   Normal inverse Gaussian process, final value, 0.101, 0.102, "0.50, 0.51", 0.74
   Merton jump diffusion, final logarithm, -0.093, -0.081, "0.58, 0.59", 0.15
   Compound Poisson process, value at time 2, -0.813, -0.806, "0.89, 0.90", 0.33
   Hawkes process, "arrivals in [0, 20]", 79.1, 78.5, "17.6, 17.4", 0.04
   Fractional Vasicek process, final value, 0.609, 0.610, "0.16, 0.16", 0.57
   Fractional Ornstein-Uhlenbeck process, final value, 0.220, 0.219, "0.16, 0.16", 0.56
   "Fractional Ornstein-Uhlenbeck process, H = 0.3", final value, 0.218, 0.220, "0.18, 0.18", 0.03
   Fractional Cox-Ingersoll-Ross process, final value, 1.011, 1.010, "0.20, 0.20", 0.77
   Fractional Jacobi process, final value, 0.476, 0.475, "0.11, 0.11", 0.18
   Fractional geometric Brownian motion, final logarithm, 0.048, 0.048, "0.20, 0.20", 0.89
   "Heston, price", final logarithm, 0.017, 0.017, "0.26, 0.26", 0.84
   "Heston, variance", final value, 0.047, 0.047, "0.03, 0.03", 0.09
   "Bates, price", final logarithm, -0.011, -0.008, "0.36, 0.37", 0.45
   "SABR, forward", final value, 0.998, 0.999, "0.20, 0.20", 0.92
   "SABR, volatility", final value, 0.200, 0.200, "0.08, 0.08", 0.52
   Correlated Brownian motions  sum of, final values, 0.009, 0.006, "1.79, 1.79", 0.95
   "Cox-Ingersoll-Ross, exact", final value, 1.000, 1.000, "0.20, 0.20", 0.28
   Ho-Lee, final value, 0.506, 0.499, "0.31, 0.31", 0.87
   "Hull-White two-factor, rate", final value, 0.279, 0.284, "0.16, 0.16", 0.19
   "Hull-White two-factor, factor", final value, -0.001, -0.002, "0.14, 0.14", 0.24
   AR(2) process, value 300, 0.010, -0.007, "2.25, 2.29", 0.05
   MA(2) process, value 300, 0.015, -0.000, "1.77, 1.77", 0.17
   "ARIMA(1, 1, 1) process", value 300, 0.3, -0.0, "44.8, 44.9", 0.44
   "SARIMA(1, 0, 1)(1, 1, 1) process, period 4", value 300, 0.1, -0.0, "22.3, 22.2", 0.42
   ARCH(1) process, value 300, 0.007, 0.006, "1.01, 1.00", 0.81
   "GARCH(1, 1) process", value 300, 0.002, 0.001, "1.01, 1.00", 0.96
   "GARCH(2, 2) process", value 300, -0.005, 0.004, "1.00, 1.00", 0.28
   "EGARCH(1, 1) process", value 300, 0.001, -0.001, "0.63, 0.63", 0.64
   "Threshold GARCH(1, 1) process", value 300, 0.002, -0.002, "1.00, 1.00", 0.54
   "AGARCH of ``stochastic-rs``, threshold GARCH here", value 300, 0.002, -0.005, "0.99, 1.00", 0.42

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
:doc:`/project/changelog`.

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
