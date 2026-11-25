Quickstart
==========

This page walks through the whole interface of ``samplepaths`` in a few
minutes. Every process works the same way:

1. import the process class
2. create an instance with the parameters of the process
3. call ``sample(n)`` to generate a realization

Your first realization
----------------------

.. code-block:: python

    from samplepaths import BrownianMotion

    bm = BrownianMotion(drift=1, scale=2, t=10)
    path = bm.sample(1000)

    print(path.shape)
    # (1001,)

``sample(n)`` generates ``n`` increments. A continuous-time process starts
at time zero, so the realization holds ``n + 1`` values on the interval
``[0, t]``.

Every process can be imported from the top-level package, as above, or from
the subpackage of its family:

.. code-block:: python

    from samplepaths.continuous import BrownianMotion
    from samplepaths.diffusion import VasicekProcess
    from samplepaths.discrete import MarkovChain
    from samplepaths.noise import PinkNoise

Plotting a realization
----------------------

After sampling, the ``times`` attribute holds the time values of the
realization.

.. code-block:: python

    import matplotlib.pyplot as plt
    from samplepaths import BrownianMotion

    bm = BrownianMotion(drift=1, scale=2, t=10)

    for _ in range(3):
        path = bm.sample(1000)
        plt.plot(bm.times, path)

    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.show()

Reproducible results
--------------------

Pass a :py:class:`numpy.random.Generator` to make a process reproducible:

.. code-block:: python

    import numpy as np
    from samplepaths import BrownianMotion

    bm = BrownianMotion(rng=np.random.default_rng(42))
    print(bm.sample(4))
    # [ 0.          0.15235854 -0.36763351  0.00759208  0.47787444]

or seed the generator shared by all processes that were not given one:

.. code-block:: python

    from samplepaths import BrownianMotion
    from samplepaths import random

    random.seed(value=42)
    print(BrownianMotion().sample(4))
    # [ 0.          0.15235854 -0.36763351  0.00759208  0.47787444]

See :doc:`/guide/random` for details.

A Monte Carlo simulation
------------------------

Because every process has the same ``sample(n)`` method, simulation code
does not depend on the process it simulates. This example estimates the mean
of a geometric Brownian motion at time ``t``, which is
:math:`S_0 e^{\mu t}`:

.. code-block:: python

    import numpy as np
    from samplepaths import GeometricBrownianMotion

    gbm = GeometricBrownianMotion(
        drift=0.05,
        volatility=0.2,
        t=1,
        rng=np.random.default_rng(1),
    )

    finals = np.array([gbm.sample_with_initial(64, initial=100.0)[-1] for _ in range(10_000)])

    print(finals.mean())  # close to 100 * exp(0.05) = 105.13

Where to go next
----------------

* :doc:`/guide/usage` describes the sampling methods in detail
* :doc:`/guide/continuous`, :doc:`/guide/diffusion`, :doc:`/guide/discrete`
  and :doc:`/guide/noise`
  describe every process
* :doc:`/reference/index` documents every class, method and function
