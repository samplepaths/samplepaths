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

``stochastic-rs`` has a Rust core, so it is the package to measure against.
:doc:`/project/comparison/speed` has the times of both for one realization and for
large batches on 14 cores. In short:

* For 100,000 realizations of 64 increments, ``samplepaths`` takes between
  0.5 and 1.4 times the time of ``stochastic-rs`` for 16 of the 19 processes
  compared, and is faster for seven of them.
* For long paths, the processes which advance step by step take 1.6 to 8
  times as long.
* For one realization, compiled code wins by a factor of 1 to 14.

Do the two packages agree?
--------------------------

The two use different random number generators, so what can be compared is
the distribution of their results. :doc:`/project/comparison/agreement` compares 43
quantities with a Kolmogorov-Smirnov test and finds no difference beyond
what chance explains. The first comparison showed two processes of
``samplepaths`` to be wrong; both are fixed.

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

What may come next is on the :doc:`roadmap`. Option pricing, calibration and
statistical tests, which ``stochastic-rs`` offers beyond simulation, are not
planned; see :doc:`decisions`.

.. toctree::
   :hidden:

   comparison/speed
   comparison/agreement
