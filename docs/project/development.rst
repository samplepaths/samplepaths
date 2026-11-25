Development
===========

``samplepaths`` is developed on
`GitHub <https://github.com/samplepaths/samplepaths>`_. Bug reports, feature
requests and pull requests are welcome, in particular for additional
processes, generalizations and algorithms.

Setting up
----------

The project is managed with `uv <https://docs.astral.sh/uv/>`_, and the
common tasks are available as ``make`` targets.

.. code-block:: bash

   git clone https://github.com/samplepaths/samplepaths.git
   cd samplepaths
   make install

``make install`` creates a virtual environment in ``.venv`` with the package
and all development dependencies, exactly as pinned in ``uv.lock``.

Make targets
------------

Run ``make`` without arguments to list the targets.

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Target
     - Description
   * - ``make install``
     - create the virtual environment and install all dependencies
   * - ``make lint``
     - check the code style and formatting with ``ruff``
   * - ``make format``
     - format the code and apply the safe lint fixes
   * - ``make typecheck``
     - check the types with ``mypy``
   * - ``make test``
     - run the tests
   * - ``make coverage``
     - run the tests in parallel with a coverage report
   * - ``make check``
     - run every check that CI runs
   * - ``make docs``
     - build this documentation, treating warnings as errors
   * - ``make docs-live``
     - serve the documentation and rebuild it on changes
   * - ``make docs-images``
     - regenerate the figures shown in the documentation
   * - ``make benchmark``
     - measure the time of the sampling methods
   * - ``make version``
     - show the version derived from the git tags
   * - ``make build``
     - build the source distribution and the wheel
   * - ``make clean``
     - remove build, test and cache artifacts

Project layout
--------------

.. code-block:: text

   src/samplepaths/
       base/          base classes of the processes
       continuous/    continuous-time processes, one module per process
       diffusion/     diffusion processes
       discrete/      discrete-time processes
       noise/         noise processes
       utils/         helper and validation functions
       random.py      default random number generator
   tests/             one test module per source module, in the same layout
   docs/              this documentation
       guide/         the user guide
       examples/      the examples of the gallery, run at build time
       scripts/       the benchmark and the script which draws the figures

Every class lives in its own module, named after the class.

Dependencies are declared in ``pyproject.toml``. The development
dependencies are split into the groups ``test``, ``lint`` and ``docs``, and
the group ``dev`` includes all of them.

Code style
----------

* The code is formatted and linted with ``ruff`` and type checked with
  ``mypy`` in strict mode.
* Every public module, class, method and function has a docstring. The
  docstrings use Sphinx field lists (``:param name:``, ``:returns:`` and
  ``:raises Error:``). Types are not repeated in the docstrings; they are
  taken from the type hints.
* The parameters of a class are documented in the class docstring.
* ``typing.override`` is imported from ``samplepaths._compat``, which provides
  it on Python 3.11.

Adding a process
----------------

1. Add a module with the process class to the subpackage of its family.
   Derive from :py:class:`~samplepaths.base.BaseTimeProcess` for a process on
   a time interval or from
   :py:class:`~samplepaths.base.BaseSequenceProcess` for a discrete-time
   process, and implement ``sample(n)``.
2. Validate the parameters in property setters, using the functions of
   :py:mod:`samplepaths.utils.validation`.
3. Export the class from the ``__init__.py`` of the subpackage and from
   ``samplepaths/__init__.py``.
4. Add a test module covering invalid parameters, the string
   representations, the shape of the realizations and their values for a
   seeded generator.
5. Add the process to the guide of its family in ``docs/guide`` and a figure
   to ``docs/scripts/make_images.py``. Add an example to the folder of the
   family in ``docs/examples``. The API reference is generated
   automatically.
6. Override ``sample_paths(n, paths)`` when the process can generate many
   realizations in one pass, and add the process to the cases of
   ``tests/test_sample_paths.py``, which compares the result with repeated
   calls to ``sample()``. A process which derives from another process must
   override it, or it would inherit the realizations of its parent.

Documentation
-------------

The documentation is built with Sphinx. The pages in ``docs/reference/api``
are generated from the docstrings at build time, and the pages in
``docs/gallery`` from the examples in ``docs/examples``, which are run.
Neither is committed.

Keep the documentation in step with the code. A change to the behavior or to
the interface of the library updates, in the same pull request:

* the guide page of the feature in ``docs/guide``, including its examples
* the gallery in ``docs/examples`` when an example shows the change best
* ``docs/guide/performance.rst`` when sampling times change, with times
  measured by ``make benchmark``
* the "Unreleased" section of ``docs/project/changelog.rst``
* ``README.md`` when the change affects what a first-time reader should know

.. code-block:: bash

   make docs       # build into docs/_build/html
   make docs-live  # serve at http://127.0.0.1:8000 and rebuild on changes

Continuous integration
----------------------

Every pull request to ``master`` runs these workflows:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Workflow
     - What it does
   * - Tests
     - the tests on Linux, macOS and Windows with Python 3.11 to 3.14
   * - Quality
     - ``ruff``, ``mypy`` and a build of the documentation, which runs the
       examples of the gallery
   * - Benchmarks
     - measures the sampling times on the base branch and on the pull
       request, and fails when a method becomes more than two times slower
   * - CodeQL
     - static analysis of the code and of the workflows

Releasing
---------

The version of the package is not written in any file. It is read from the
git tag of the release: the tag ``v0.2.0`` gives the version ``0.2.0``.

To release, rename the "Unreleased" section of ``docs/project/changelog.rst``
to the new version and its date, merge that change, and create a GitHub
release on ``master`` with a new tag of the form ``vX.Y.Z``. Publishing the release runs the ``Publish`` workflow, which
builds the distributions, checks that their version matches the tag and
uploads them to PyPI using trusted publishing.

Between releases the version is derived from the most recent tag and the
number of commits since, for example ``0.2.1.dev3+g1a2b3c4``.
