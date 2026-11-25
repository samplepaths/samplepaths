Decisions
=========

Choices which shaped the package, with their reasons, most general first.
Read this page before proposing to change one of them, and add an entry when
a choice is made which someone could later question: what was chosen, what
the alternative was, and why.

The package
-----------

**Pure Python on NumPy and SciPy.**
   No compiled extension of its own. The package installs wherever NumPy
   does, every process is one readable module, and for large batches it is
   level with a Rust implementation; see :doc:`comparison`. The price is a
   single realization of the step-by-step models, 4 to 14 times slower than
   compiled code.

**The name samplepaths.**
   The package is a fork of ``stochastic``. The names ``stochastic`` and
   ``pystochastic`` were taken on PyPI, and the fork needed a name of its
   own to be published.

**Simulation only.**
   No option pricing, calibration, statistical tests or plain probability
   distributions, which ``stochastic-rs`` has. They are other packages'
   work, and each would need its own validation.

**Python 3.11 and newer.**
   Python 3.10 is past its end of life, and its SciPy wheels no longer load
   everywhere.

The interface
-------------

**Results for a given seed never change, except for fixes.**
   ``sample()`` and ``sample_at()`` return the same values, bit for bit, in
   every version. Users rely on a seed to reproduce published results. A fix
   may change results and says so in the changelog.

**A better algorithm is a new method.**
   Exact sampling is ``sample_exact()``, not a change of ``sample()``, which
   keeps the Euler-Maruyama method. This follows from the rule above. The
   Euler-Maruyama method of the Cox-Ingersoll-Ross process still returns
   ``nan`` when a step goes below zero for the same reason.

**sample_paths() has the distribution of sample(), not its numbers.**
   Generating all realizations together draws the random numbers in another
   order. Asking for the same numbers would forbid most of the speed.

**Several components are one array with a component axis.**
   ``(components, n + 1)`` from ``sample()`` and
   ``(paths, components, n + 1)`` from ``sample_paths()``, with the names in
   ``components``. A tuple of arrays was the alternative; one array unpacks
   the same way, ``price, variance = model.sample(n)``, and keeps
   ``sample_paths()`` a single array like for every other process.

**Parameters of the hidden component are constructor arguments.**
   ``initial_variance`` of the Heston model and ``initial_volatility`` of the
   SABR model belong to the model, while the initial price is an argument of
   ``sample_with_initial()``, as for the other processes.

**Models keep the names they are known by.**
   ``HestonModel``, ``BatesModel``, ``SABRModel``, not ``...Process``: these
   are the names people search for.

The processes
-------------

**The Heston variance is drawn from its exact distribution.**
   The Euler method needs the variance to be cut off at zero and depends on
   the step. The exact noncentral chi-square transition does neither, and
   the price then uses the discretization of Andersen. It costs more per
   step than a normal variate.

**No Hull-White class with one factor.**
   It is the ``ExtendedVasicekProcess`` with a mean given as a function of
   time. A second class for the same equation would be two things to
   maintain.

**The fractional geometric Brownian motion is the Wick form.**
   Its mean grows at the drift for every Hurst parameter, which is what a
   price model needs, and it is exact. ``stochastic-rs`` applies the Euler
   method to the equation, which for a Hurst parameter above 1/2 tends to a
   process whose logarithm is higher by a known term.

**TGARCHProcess and AGARCHProcess follow the literature.**
   The threshold model is the one of Glosten, Jagannathan and Runkle, and the
   asymmetric model the one of Engle with a shift. ``stochastic-rs`` calls
   the first AGARCH.

**Linear series start from zero, variance models from their long-run state.**
   A linear model started from zero is what a filter computes and matches
   other packages. A variance model started away from its long-run variance
   would need values to be thrown away before it is usable.

**The fractional diffusions accept every Hurst parameter.**
   The Euler method is only known to converge above 1/2, and below it the
   Cox-Ingersoll-Ross and Jacobi versions stay at the boundary. This is
   documented instead of forbidden, because ``stochastic-rs`` allows it and
   the Ornstein-Uhlenbeck version is useful for rough volatility.

**No sample_at() for the SABR and the two-factor Hull-White models.**
   Their step loops assume equal steps. They were left out when
   ``sample_at()`` was added to the exact models, not ruled out; see the
   :doc:`roadmap`.

Performance
-----------

**Step-by-step models use at most six threads.**
   With more, the threads compete for the interpreter: the Heston model was
   slower on 14 threads than on 6. Models computed with whole-array
   operations use every thread.

**Loops are written out per class for one realization.**
   The general loop calls a method per step. Writing the loop of a class out
   with local variables made one GARCH realization five times faster. The
   duplication is accepted, and a test checks each written-out loop against
   the general one.

The project
-----------

**uv, one Makefile, mainstream tools.**
   All dependencies, those of the docs included, are dependency groups in
   ``pyproject.toml``; there are no requirements files. One top-level
   ``Makefile`` holds every command. ``ruff``, ``mypy`` and ``pytest`` with
   ordinary settings. One conventional toolchain is easier to keep than
   several.

**One class per file, and files under 300 lines.**
   The package is one module per process, and small files keep each of them
   readable.

**What a split produces gets a place of its own.**
   A file which is split keeps its name and its place. Its parts go into a
   private subpackage for code, such as ``_diffusion_process`` beside
   ``diffusion_process.py`` or ``_images`` beside ``make_images.py``, and
   into a subdirectory for pages and tests, such as ``comparison/`` beside
   ``comparison.rst``. The folder then shows what belongs together, and
   the parts do not crowd the folder of the files they came from. Tables
   with many rows are written as CSV tables, one line per row, which keeps
   the pages short.

**The docs are rewritten, not appended to.**
   Every change updates the pages it affects in the same pull request, and
   reorganizes them when needed, so that the documentation reads as a guide
   and not as a log of changes.

**Workflows call make targets; scripts are Python in .github/scripts.**
   What runs in CI can be run locally with the same command, and the scripts
   are linted and type checked like the rest.

**Tests and CodeQL run on pull requests and daily, not on merge.**
   A merge has already been tested as a pull request. The daily run catches
   what changes outside the repository.

**The version comes from the git tag.**
   It is written in no file, so a release cannot disagree with its tag.

**Knowledge is kept in the repository.**
   Rules, open work and decisions live in
   ``.github/copilot-instructions.md``, the :doc:`roadmap` and this page,
   not in the private notes of a tool or a person, so that every contributor
   and every coding agent starts from the same state.
