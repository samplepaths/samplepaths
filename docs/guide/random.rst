Random Number Generation
========================

``samplepaths`` uses :py:class:`numpy.random.Generator` for all random number
generation. The legacy ``numpy.random.RandomState`` is not supported.

There are two ways to control the generator used by a process:

* pass an ``rng`` to the process, which only affects that instance
* change the default generator of the package, which affects every process
  created without an ``rng``

The generator of a process
--------------------------

Every process accepts an ``rng`` argument and exposes the generator it uses
as the ``rng`` attribute.

.. code-block:: python

    import numpy as np
    from samplepaths import GaussianNoise

    gn = GaussianNoise(rng=np.random.default_rng(42))

    print(gn.sample(4))
    # [ 0.15235854 -0.51999205  0.3752256   0.47028236]

Give each process its own generator when you need independent random
streams, for example to keep one process reproducible while others change:

.. code-block:: python

    import numpy as np
    from samplepaths import BrownianMotion
    from samplepaths import PoissonProcess

    streams = np.random.SeedSequence(2026).spawn(2)

    prices = BrownianMotion(rng=np.random.default_rng(streams[0]))
    arrivals = PoissonProcess(rate=3, rng=np.random.default_rng(streams[1]))

The default generator
---------------------

A process created without an ``rng`` uses the default generator of the
package, :py:data:`samplepaths.random.generator`. The process looks it up
every time it samples, so changing the default affects existing instances.

Seeding the default generator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:py:func:`samplepaths.random.seed` replaces the default generator with a new
one created from a seed:

.. code-block:: python

    from samplepaths import GaussianNoise
    from samplepaths import random

    gn = GaussianNoise()

    random.seed(value=42)
    print(gn.sample(4))
    # [ 0.15235854 -0.51999205  0.3752256   0.47028236]

    random.seed(value=42)
    print(gn.sample(4))
    # [ 0.15235854 -0.51999205  0.3752256   0.47028236]

Replacing the default generator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:py:func:`samplepaths.random.use_generator` sets the default generator to one
of your choice, for example one using a different
`bit generator <https://numpy.org/doc/stable/reference/random/bit_generators/index.html>`__.
Calling it without an argument restores the original default.

.. code-block:: python

    from numpy.random import Generator
    from numpy.random import Philox
    from samplepaths import GaussianNoise
    from samplepaths import random

    gn = GaussianNoise()
    print(gn.rng is random.generator)
    # True

    custom = Generator(Philox(seed=7))
    random.use_generator(rng=custom)
    print(gn.rng is custom)
    # True

    random.use_generator()

Both functions take keyword arguments only. See
:py:mod:`samplepaths.random` in the API reference.

Many realizations
-----------------

``sample_paths()`` uses the generator of the process as well, and is
reproducible in the same way. It draws the random numbers of all realizations
together, in a different order from ``sample()``. With the same seed, its
rows are therefore not the realizations which repeated calls to ``sample()``
would return, although they have the same distribution.

.. code-block:: python

    import numpy as np
    from samplepaths import BrownianMotion

    first = BrownianMotion(rng=np.random.default_rng(42)).sample_paths(8, paths=3)
    second = BrownianMotion(rng=np.random.default_rng(42)).sample_paths(8, paths=3)

    print(np.array_equal(first, second))
    # True

Which one to use
----------------

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Goal
     - Use
   * - Make a whole script reproducible
     - ``random.seed(value=...)`` once at the start
   * - Make one process reproducible
     - ``rng=numpy.random.default_rng(seed)`` on that process
   * - Independent streams, for example in parallel workers
     - one ``rng`` per process, spawned from a
       :py:class:`numpy.random.SeedSequence`
