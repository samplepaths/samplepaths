Exact Sampling
==============

The Euler-Maruyama method approximates a diffusion process over each time
step, and the approximation depends on the size of the step. For some
processes the value after a step has a known distribution given the value
before. Drawing from that distribution is exact: the realization does not
depend on the grid, and a single step gives the true distribution of the
final value.

Which processes
---------------

.. list-table::
   :header-rows: 1
   :widths: 44 56

   * - Process
     - Value after a step, given the value before
   * - :py:class:`~samplepaths.diffusion.VasicekProcess`,
       :py:class:`~samplepaths.diffusion.OrnsteinUhlenbeckProcess` and
       :py:class:`~samplepaths.diffusion.ExtendedVasicekProcess`
     - normal
   * - :py:class:`~samplepaths.diffusion.CoxIngersollRossProcess`
     - a multiple of a noncentral chi-square variate

The parameters must be numbers, with a positive speed. The processes built
on them are exact in the same way: the variance of the
:doc:`Heston and Bates models <multivariate>` and the factors of the
two-factor Cox-Ingersoll-Ross model.

The methods
-----------

``sample_exact(n, initial)`` and ``sample_paths_exact(n, paths, initial)``
are the exact counterparts of ``sample_with_initial()`` and
``sample_paths_with_initial()``. ``sample()`` and ``sample_paths()`` keep
using the Euler-Maruyama method, so their results for a given seed do not
change.

.. code-block:: python

    from samplepaths import CoxIngersollRossProcess, VasicekProcess

    vasicek = VasicekProcess(speed=1.0, mean=0.03, vol=0.01, t=10)
    path = vasicek.sample_exact(120, initial=0.02)

    cir = CoxIngersollRossProcess(speed=1.0, mean=0.04, vol=0.6)
    final = cir.sample_paths_exact(1, paths=100_000, initial=0.04)[:, -1]

Compared with the Euler-Maruyama method:

* **One step is enough for the final value.** 100,000 exact final values
  take 0.8 milliseconds for a Vasicek process and 1.6 for a
  Cox-Ingersoll-Ross process.
* **A Cox-Ingersoll-Ross process is never negative.** When
  ``2 * speed * mean`` is less than ``vol ** 2`` the process reaches zero.
  The Euler-Maruyama method then steps below zero and returns ``nan``; with
  the parameters above it does so for most realizations. The exact method
  returns zero and carries on.
* **The cost of a step differs by process.** For the Vasicek family the
  exact method is faster: one realization of 4096 increments takes 0.06
  instead of 0.21 milliseconds, because no step is taken in Python. For the
  Cox-Ingersoll-Ross process a step costs about twice as much, 1.1 against
  0.46 milliseconds.

Sampling at chosen times
------------------------

Because the exact methods do not need equal steps, these processes can be
sampled at any increasing times with ``sample_at(times)`` and
``sample_at_with_initial(times, initial)``.

.. code-block:: python

    from samplepaths import CoxIngersollRossProcess

    cir = CoxIngersollRossProcess(speed=1.0, mean=0.04, vol=0.2)
    rates = cir.sample_at_with_initial([0, 0.25, 0.5, 1, 2, 5, 10], initial=0.03)

The same holds for the
:py:class:`~samplepaths.diffusion.HoLeeProcess`, whose noise is exact, and
for these :doc:`processes with several components <multivariate>`, which
return one row per component and one column per time:

.. list-table::
   :header-rows: 1
   :widths: 44 56

   * - Process
     - At the chosen times
   * - :py:class:`~samplepaths.multivariate.CorrelatedBrownianMotions`
     - exact
   * - :py:class:`~samplepaths.multivariate.TwoFactorCoxIngersollRossModel`
     - exact
   * - :py:class:`~samplepaths.multivariate.HestonModel` and
       :py:class:`~samplepaths.multivariate.BatesModel`
     - the variance is exact; the price is as accurate as the longest step
       is short

When the first time is not zero, the realization has no value for time zero
and starts with the value at the first time. A parameter given as a function
of time makes ``sample_exact()`` raise ``ValueError`` and ``sample_at()``
raise ``NotImplementedError``.
