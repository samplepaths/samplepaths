The Refactor
============

``samplepaths`` started as a fork of
`stochastic <https://github.com/crflynn/stochastic>`_ 0.7.0. The mathematics
of the original package was kept: the same processes, the same sampling
algorithms and the same bibliographical sources. What changed is everything
around it. The code base was restructured, typed, re-tested and moved to a
current toolchain.

This page describes what the refactor did and why it matters when you use
the library. For the list of changes needed to port existing code, see
:doc:`migration`.

Runs on a current scientific Python stack
-----------------------------------------

The original package supported Python 3.8 to 3.10 and ``numpy`` 1.x, which
prevents installing it next to current releases of the scientific Python
stack.

``samplepaths`` requires Python 3.12 or newer, ``numpy`` 2.2 or newer and
``scipy`` 1.15 or newer, and is tested on Python 3.12, 3.13 and 3.14 on
Linux, macOS and Windows.

**What you get:** the library installs into a modern environment without
pinning old versions of ``numpy`` or ``scipy``.

Type hints throughout
---------------------

Every public class, method and function is annotated, and the package ships
a ``py.typed`` marker so type checkers use the annotations. The code base
itself is checked with ``mypy`` in strict mode, and every public class,
method and function has a docstring in one enforced style.

**What you get:** editors show accurate signatures and completions, and
``mypy`` or ``pyright`` can catch a wrong argument in your simulation code
before it runs.

One sampling interface for every process
----------------------------------------

In the original package the signature of ``sample()`` depended on the
process: some took an ``algorithm`` argument, others an ``initial`` value, a
``length`` or a ``start`` state.

In ``samplepaths``, ``sample(n)`` has the same signature on every process.
Anything beyond the number of steps has its own explicitly named method, such
as ``sample_with_initial()``, ``sample_with_length()`` or
``sample_hosking()``.

**What you get:** processes are interchangeable. Code that runs a Monte Carlo
simulation can accept any process without knowing which one it is:

.. code-block:: python

    import numpy as np
    from samplepaths import BrownianMotion
    from samplepaths import CoxIngersollRossProcess
    from samplepaths import PinkNoise
    from samplepaths.base import BaseProcess


    def final_values(process: BaseProcess, n: int, paths: int) -> np.ndarray:
        """Simulate a number of paths and collect the last value of each."""
        return np.array([process.sample(n)[-1] for _ in range(paths)])


    for process in (
        BrownianMotion(drift=1, scale=2),
        CoxIngersollRossProcess(speed=1, mean=1, vol=0.5),
        PinkNoise(),
    ):
        print(type(process).__name__, final_values(process, n=64, paths=100).mean())

Explicit, validated parameters
------------------------------

The continuous-time, diffusion and noise processes take keyword arguments
only, so a call such as ``BrownianMotion(drift=1, scale=2, t=10)`` states
what each number means and cannot silently swap two parameters.

Parameters are validated when they are set, both at instantiation and when
an attribute is changed later, using one shared set of validation helpers.

.. code-block:: python

    from samplepaths import BrownianMotion


    bm = BrownianMotion(drift=1, scale=2)

    try:
        bm.scale = -1
    except ValueError as error:
        print(error)

**What you get:** an invalid parameter fails immediately with a message
naming the parameter, instead of producing a wrong realization later.

A single random number generation model
---------------------------------------

The original package supported both the legacy ``numpy.random.RandomState``
and :py:class:`numpy.random.Generator`. ``samplepaths`` supports
:py:class:`numpy.random.Generator` only.

Every process accepts its own ``rng``, and processes without one share the
package default, which can be seeded or replaced. See :doc:`/guide/random`.

**What you get:** one consistent way to make a simulation reproducible, and
independent random streams per process when you need them.

A structure that is easier to extend
------------------------------------

Each process lives in its own module, and all processes derive from three
small base classes in :py:mod:`samplepaths.base`:

* ``BaseProcess`` handles random number generation and defines ``sample()``
* ``BaseSequenceProcess`` is the base of the discrete-time processes
* ``BaseTimeProcess`` is the base of the processes defined on ``[0, t]``

Every process can be imported from the subpackage of its family, such as
``samplepaths.continuous``, or directly from ``samplepaths``.

**What you get:** adding a process means adding one module and one test
module, without touching the others.

Stronger quality checks
-----------------------

The test suite was rewritten as parametrized tests with one test module per
process, covering valid and invalid parameters as well as sampling. Every
pull request runs the tests on three operating systems and three Python
versions, together with ``ruff``, ``mypy`` and a build of this
documentation. Dependencies are locked with ``uv`` so that builds are
reproducible.

**What you get:** changes to the library are verified on the platforms and
Python versions it supports before they are released.
