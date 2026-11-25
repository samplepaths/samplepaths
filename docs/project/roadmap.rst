Roadmap
=======

The work which is open, most useful first within each part. This page is
the list of record: when an item is done, remove it and add the change to
the :doc:`changelog`; when new work is found or put off, add it here.

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
