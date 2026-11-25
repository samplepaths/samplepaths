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
     - 37 named processes in Python on ``numpy``; typed; reproducible
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

For one realization, compiled code wins: by a factor of 1 to 3 where
``samplepaths`` calls ``numpy`` a few times, and by up to 14 for the
diffusion processes, which ``samplepaths`` advances step by step in Python.
All of these take less than half a millisecond.

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
   * - Vasicek process
     - 7.8
     - 8.9
     - 1.1
   * - CEV process
     - 18.5
     - 20.3
     - 1.1
   * - Geometric Brownian motion
     - 7.6
     - 9.0
     - 1.2
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
   * - Gaussian noise
     - 15.1
     - 12.4
     - 0.8
   * - Fractional Brownian motion
     - 45.5
     - 37.6
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
   * - CEV process
     - 55.3
     - 212.8
     - 3.9
   * - Cox-Ingersoll-Ross process
     - 27.2
     - 213.9
     - 7.9

For large batches the two are close: ``samplepaths`` is between 0.6 and 1.4
times the time of ``stochastic-rs`` for eight of the ten processes. The
exceptions are the CEV and the Cox-Ingersoll-Ross processes, whose
volatility depends on the value of the process. They cannot be written as
array operations over time, and for long paths ``samplepaths`` is 4 to 8
times slower.

For small batches the fixed cost of a call dominates. For 1000 realizations
of 256 increments ``samplepaths`` takes 0.5 to 2.6 milliseconds and is 1.4 to
4.7 times slower than ``stochastic-rs`` in parallel.

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
* **Coverage.** ``stochastic-rs`` has about 70 models which ``samplepaths``
  does not have yet: jump and Lévy processes, stochastic volatility models,
  interest rate models, fractional diffusions, time series models and
  correlated processes. ``samplepaths`` has the Chinese restaurant, the
  Dirichlet and the Moran processes, the Markov chain, the Bessel processes,
  the Brownian bridge, excursion and meander, the colored noises and the
  multifractional Brownian motion, which ``stochastic-rs`` does not list.

Planned
-------

The missing models are being added in this order: jump processes and simple
diffusions, time series models, fractional models, then stochastic
volatility, interest rate and correlated models, which have several
components. Exact sampling for the Cox-Ingersoll-Ross process, which would
remove the step-by-step loop, is planned with them.
