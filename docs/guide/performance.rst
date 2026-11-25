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

:doc:`/guide/performance/timings` gives, for every process, the time of one realization of 4096
increments with ``sample()`` and with ``sample_at()``, and of 1000
realizations of 256 increments with ``sample_paths()``.

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
  realization of 4096 values of a GARCH process takes 0.42 milliseconds, and
  1000 realizations of 256 values take 2.1.
* **Use the exact methods where they exist.** ``sample_exact()`` and
  ``sample_paths_exact()`` of the Vasicek, Ornstein-Uhlenbeck and
  Cox-Ingersoll-Ross processes draw each value from its exact distribution,
  so a few steps suffice: one step gives 100,000 exact final values in 0.8
  milliseconds for a Vasicek process and 1.6 for a Cox-Ingersoll-Ross
  process. See :doc:`exact`.
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

.. toctree::
   :hidden:

   performance/timings
