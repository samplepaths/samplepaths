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
   notebooks/         notebooks comparing the processes with ``stochastic``

Dependencies are declared in ``pyproject.toml``. The development
dependencies are split into the groups ``test``, ``lint``, ``docs`` and
``notebooks``, and the group ``dev`` includes all of them.

Code style
----------

* The code is formatted and linted with ``ruff`` and type checked with
  ``mypy`` in strict mode.
* Every public module, class, method and function has a docstring. The
  docstrings use Sphinx field lists (``:param name:``, ``:returns:`` and
  ``:raises Error:``). Types are not repeated in the docstrings; they are
  taken from the type hints.
* The parameters of a class are documented in the class docstring.

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
   to ``docs/scripts/make_images.py``. The API reference is generated
   automatically.

Documentation
-------------

The documentation is built with Sphinx. The pages in ``docs/reference/api``
are generated from the docstrings at build time and are not committed.

.. code-block:: bash

   make docs       # build into docs/_build/html
   make docs-live  # serve at http://127.0.0.1:8000 and rebuild on changes

Releasing
---------

The version of the package is not written in any file. It is read from the
git tag of the release: the tag ``v0.2.0`` gives the version ``0.2.0``.

To release, create a GitHub release on ``master`` with a new tag of the form
``vX.Y.Z``. Publishing the release runs the ``Publish`` workflow, which
builds the distributions, checks that their version matches the tag and
uploads them to PyPI using trusted publishing.

Between releases the version is derived from the most recent tag and the
number of commits since, for example ``0.2.1.dev3+g1a2b3c4``.
