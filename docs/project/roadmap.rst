Roadmap
=======

The work which is open, most useful first within each part. This page is
the list of record: when an item is done, remove it and add the change to
the :doc:`changelog`; when new work is found or put off, add it here.

Next
----

* **Release 0.6.0.** The changelog section is written. Merge the pull
  request which prepares it, then create the release ``v0.6.0`` on
  ``master``; see :doc:`development`.
* **Check the daily job with the latest dependencies.** It runs on a
  schedule or by hand only, so its first run is after the merge. Start it
  once from the Actions tab and read the result.
* **Split the files over 300 lines.** Files stay under 300 lines; these were
  longer before the rule and are split when next changed, or in a change of
  their own:

  * ``docs/project/comparison.rst``: the speed tables, the agreement table
    and the discussion can be three pages
  * ``docs/guide/performance.rst``: one page of tables per family, or the
    tables generated from the benchmark
  * ``docs/project/changelog.rst``: one page per minor version
  * ``docs/guide/continuous.rst``: the jump processes can have their own page
  * ``docs/scripts/make_images.py`` and
    ``docs/scripts/compare_rust_outputs.py``: one module per family
  * ``tests/test_sample_paths.py``: the cases apart from the tests
  * ``src/samplepaths/diffusion/diffusion_process.py``: the two loops of
    the Euler-Maruyama method can move to a private module

Sampling
--------

* **Sampling at chosen times for the SABR and the two-factor Hull-White
  models.** Both need their step loop written for unequal steps. See
  :doc:`decisions` for why they were left out.
* **Quasi-random and antithetic sampling.** For Monte Carlo estimates,
  reducing the variance gives more accuracy per realization than more speed.
  It needs a decision on the interface first: an argument of
  ``sample_paths()``, or a generator which the process accepts.
* **Exact methods for more processes.** The Heston price could use the exact
  scheme of Broadie and Kaya where accuracy matters more than time.
* **Faster single realizations of the general diffusion process.** One
  realization of a Cox-Ingersoll-Ross or a CEV process is 5 to 14 times
  slower than compiled code. Its loop already uses Python floats; the
  remaining cost is the parameters which may be functions of time.

Models
------

Models which ``stochastic-rs`` has and ``samplepaths`` does not, to add when
someone asks for them:

* rough volatility: the rough Bergomi and the rough Heston models
* further jump and Lévy processes: CGMY, tempered stable, bilateral gamma
* multi-factor volatility models, and correlated fractional processes

Not planned: term-structure models (HJM, LMM), option pricing, calibration
and statistical tests. See :doc:`decisions`.

Project
-------

* **The macOS test jobs waited for runners for hours on 2026-11-25**, with
  no incident reported by GitHub and with an empty queue. If it happens
  again, check the limits of the account on concurrent macOS jobs before
  anything else. The cause was not found.
* **Let the daily tests open an issue when they fail.** A scheduled run which
  fails only shows in the Actions tab.
* **A write-up of the comparison with stochastic-rs**, for readers who do
  not know the package: pure Python, level with compiled code on large
  batches, and checked against it.
