Simulating Many Realizations
============================

Monte Carlo methods need many independent realizations of a process.
``sample_paths(n, paths)`` generates them in one call and returns a
two-dimensional array with one realization per row.

.. code-block:: python

    from samplepaths import BrownianMotion

    bm = BrownianMotion(drift=1, scale=2, t=10)
    realizations = bm.sample_paths(n=256, paths=10_000)

    print(realizations.shape)
    # (10000, 257)

Each row is what ``sample(n)`` returns: ``n + 1`` values for a process on a
time interval, ``n`` values for a noise or a discrete-time process. After the
call, ``times`` holds the time values shared by all rows.

Why not a loop?
---------------

A loop over ``sample()`` works for every process:

.. code-block:: python

    import numpy as np
    from samplepaths import BrownianMotion

    bm = BrownianMotion(drift=1, scale=2, t=10)
    realizations = np.array([bm.sample(256) for _ in range(1000)])

``sample_paths()`` gives realizations with the same distribution, faster.
Most processes generate all realizations in one pass, which is 2 to 13
times faster than the loop for 1000 realizations of 256 increments, and 50
to 70 times faster for the :doc:`models with a changing variance
<timeseries>`, which otherwise advance one value at a time in Python:

.. list-table::
   :header-rows: 1
   :widths: 40 20 20 20

   * - Process
     - Loop
     - ``sample_paths()``
     - Gain
   * - Cox-Ingersoll-Ross process
     - 30.0 ms
     - 2.2 ms
     - 13.5x
   * - Vasicek process
     - 14.6 ms
     - 1.8 ms
     - 8.3x
   * - Pink noise
     - 15.0 ms
     - 2.2 ms
     - 6.7x
   * - Brownian bridge
     - 6.9 ms
     - 1.6 ms
     - 4.4x
   * - Geometric Brownian motion
     - 7.6 ms
     - 2.0 ms
     - 3.8x
   * - Markov chain
     - 17.0 ms
     - 4.5 ms
     - 3.8x
   * - Brownian motion
     - 5.2 ms
     - 1.4 ms
     - 3.7x

See :doc:`performance` for all processes and for how the times were measured.

A worked example
----------------

The price of a European call option is the discounted mean of its payoff.
Under the Black-Scholes model the price follows a geometric Brownian motion,
so the mean can be estimated from simulated prices:

.. code-block:: python

    import numpy as np
    from samplepaths import GeometricBrownianMotion

    rate, volatility, maturity = 0.05, 0.2, 1.0
    spot, strike = 100.0, 105.0

    gbm = GeometricBrownianMotion(
        drift=rate,
        volatility=volatility,
        t=maturity,
        rng=np.random.default_rng(1),
    )
    prices = gbm.sample_paths_with_initial(n=64, paths=100_000, initial=spot)

    payoffs = np.maximum(prices[:, -1] - strike, 0.0)
    estimate = np.exp(-rate * maturity) * payoffs.mean()

    print(round(estimate, 2))  # close to the exact price, 8.02

Because the array holds whole paths, quantities which depend on the path are
as easy, for example the probability that the price ever exceeds 130:

.. code-block:: python

    import numpy as np
    from samplepaths import GeometricBrownianMotion

    gbm = GeometricBrownianMotion(drift=0.05, volatility=0.2, rng=np.random.default_rng(1))
    prices = gbm.sample_paths_with_initial(n=252, paths=20_000, initial=100.0)

    print((prices.max(axis=1) > 130).mean())

Using several threads
---------------------

``sample_paths()`` can generate the realizations on several threads. Pass the
number of threads as ``workers``:

.. code-block:: python

    from samplepaths import FractionalBrownianMotion

    fbm = FractionalBrownianMotion(hurst=0.7)
    realizations = fbm.sample_paths(n=64, paths=100_000, workers=8)

Each thread generates a share of the realizations with its own random number
generator, derived from the generator of the process. ``numpy`` does not hold
the interpreter while it generates and transforms the numbers, so the threads
run in parallel. On a machine with 14 cores:

.. list-table::
   :header-rows: 1
   :widths: 40 20 20 20

   * - 100,000 realizations of 64 increments
     - 1 worker
     - 14 workers
     - Gain
   * - Gaussian noise
     - 23.2 ms
     - 4.0 ms
     - 5.8x
   * - Brownian motion
     - 33.8 ms
     - 5.7 ms
     - 5.9x
   * - Fractional Brownian motion
     - 81.2 ms
     - 11.3 ms
     - 7.2x
   * - Variance gamma process
     - 104.7 ms
     - 14.9 ms
     - 7.0x
   * - Vasicek process
     - 44.3 ms
     - 8.9 ms
     - 5.0x
   * - Cox-Ingersoll-Ross process
     - 52.9 ms
     - 20.3 ms
     - 2.6x

Three things to know:

* **Threads pay off for large batches.** A thread is only started for every
  50,000 values to generate, so a small batch runs on one thread whatever
  ``workers`` is.
* **The result depends on the number of workers.** A seed gives the same
  array every time for a given ``workers``, and a different array for another
  number of workers, because the random numbers are split differently.
* **Not every process gains.** The processes generated one realization after
  the other ignore ``workers``. The diffusion processes which are not linear,
  such as the Cox-Ingersoll-Ross process, advance step by step in Python and
  use at most four threads, and only for 20,000 realizations or more per
  thread. The :doc:`time series models <timeseries>` generate their random
  numbers on the threads and then advance on one of them, which is enough
  for 100,000 realizations of 64 values to take 8 instead of 27
  milliseconds for an autoregressive process and 16 instead of 35 for a
  GARCH process.

Functions passed as parameters, such as a time-dependent mean, are called from
several threads and must not depend on shared state.

Starting from an initial value
------------------------------

``sample_paths()`` starts the geometric Brownian motion and the diffusion
processes from 1, as ``sample()`` does. Use
``sample_paths_with_initial(n, paths, initial)`` to choose the initial value.
It takes ``workers`` as well.

.. code-block:: python

    from samplepaths import VasicekProcess

    vasicek = VasicekProcess(speed=1.0, mean=0.05, vol=0.02, t=10)
    rates = vasicek.sample_paths_with_initial(n=1000, paths=5000, initial=0.03)

Reproducibility
---------------

A seeded generator makes ``sample_paths()`` reproducible: the same seed, the
same arguments and the same version of the library give the same array.

``sample_paths()`` draws its random numbers in a different order from
``sample()``. With the same seed, its rows are therefore not the realizations
which repeated calls to ``sample()`` would return, although they have the
same distribution. See :doc:`random`.

Notes on specific processes
---------------------------

* **Moran process.** A realization ends when it reaches zero or the maximum
  value, so realizations have different lengths. ``sample_paths()`` returns
  rows of ``n`` values and fills the rest of a row with the state in which
  the process was absorbed, where it stays.
* **Mixed Poisson process.** Every realization draws its own rate.
* **Chinese restaurant, Dirichlet, Moran and mixed Poisson processes.** These
  are generated one realization after the other, in the time of a loop.
* **Memory.** The result holds ``paths * (n + 1)`` values of 8 bytes: 100,000
  realizations of 1000 increments take 800 MB. Generate them in several
  calls if you only need a summary of each batch.
