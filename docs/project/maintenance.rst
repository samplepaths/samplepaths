Maintenance Notes
=================

What was learned while optimizing the package and comparing it with
``stochastic-rs``, written down so that it does not have to be learned
again.

Seeded results
--------------

Realizations of ``sample()`` and ``sample_at()`` for a given seed do not
change from one version to the next. Check it with:

.. code-block:: bash

   make check-seeded                 # against the last release
   make check-seeded VERSION=0.4.0   # against another one

The target installs the released version from PyPI in a temporary
environment, records the results of every process with it, and compares them
with the working tree, bit for bit. It also reports changes of
``sample_paths()``, ``sample_exact()`` and the variants with an initial
value, which are not promised to users and do not make it fail. The
``Quality`` workflow runs it on every pull request.

To keep results when making code faster:

* Do the same floating point operations in the same order.
  ``a * b * c`` is ``(a * b) * c``; writing it as ``a * (b * c)`` changes the
  last bit.
* Draw the same random numbers in the same order. One call for an array and
  several calls for its parts give the same numbers only when the parts are
  taken in order.
* Put a better algorithm in a new method. Exact sampling is
  ``sample_exact()`` for this reason.

A fix may change results. Say in the changelog which process changes and
why; the job of the workflow then fails for that pull request, as it should.

What makes sampling fast
------------------------

Measure before and after with ``make benchmark``; the ``Benchmarks`` workflow
fails a pull request which makes a method more than two times slower.

* **One realization, step by step.** Convert the noise to a list of Python
  floats and loop over it with local variables only: no method call and no
  attribute in the loop. This made one GARCH realization five times faster.
  Appending to a list is faster than writing into a list or an array created
  with its final size.
* **Do not build a list per step for constants.** Computing two constants
  for each of 4096 steps, when the steps are equal, cost a quarter of the
  time of an exact Cox-Ingersoll-Ross realization.
* **Many realizations, step by step.** Keep one array with a row per step
  and advance all realizations together. Rows are contiguous in memory;
  columns of an array with a row per realization are not.
* **Linear recurrences need no loop.** ``_recurrence.py`` solves
  ``X[k + 1] = decay X[k] + input[k]`` with a cumulative sum.
* **Threads help only when NumPy does the work.** Generating random numbers
  and whole-array operations release the interpreter lock. A loop of short
  NumPy calls does not: with 14 threads the Heston model was slower than
  with 6. The step-by-step models use at most six threads and one for every
  5000 realizations. ``scipy.signal.lfilter`` on several threads was several
  times slower than on one.

Comparing with stochastic-rs
----------------------------

``stochastic-rs`` has a Rust core and serves as the reference for speed and
for checking results. It is not a dependency:

.. code-block:: bash

   uv run --with stochastic-rs python docs/scripts/compare_rust.py
   uv run --with stochastic-rs python docs/scripts/compare_rust_outputs.py

* Its ``n`` is the number of points of a realization; here ``n`` is the
  number of increments.
* Parameters are often defined differently, and its initial values default to
  zero. The scripts document the mapping for each process. When two results
  disagree, read the Rust source of the process before suspecting either
  package: of the first disagreements, two were bugs here and one was a
  drift which ``stochastic-rs`` adjusts itself.
* The two use different random number generators, so only distributions can
  be compared, with a Kolmogorov-Smirnov test. Among 43 comparisons, two
  p-values below 0.05 are expected by chance, and which ones changes from run
  to run. A single run gave 0.0003 for one comparison which gives 0.71 with
  two million realizations. Before concluding that two results differ,
  repeat with more realizations and compare ``stochastic-rs`` with itself to
  see the spread.

Continuous integration
----------------------

* The rules of the ``master`` branch require the test jobs by name. Renaming
  a job of the ``Tests`` workflow blocks every merge until the rules list the
  new names.
* The workflows run for pull requests to ``master`` only. A pull request
  based on another branch runs nothing until it targets ``master``.
* Pull requests are merged by squashing. A branch built on another branch
  must be rebased on ``master`` after the first is merged, with
  ``git rebase --onto origin/master <old base> <branch>``.

Known limits
------------

* The fractional Cox-Ingersoll-Ross and Jacobi processes stay at the boundary
  of their domain for a Hurst parameter below 1/2. This is a property of the
  Euler method for them and is documented, not prevented.
* One realization of a model which advances step by step stays 4 to 14 times
  slower than compiled code. ``sample_paths()`` and the exact methods are
  the ways around it.
