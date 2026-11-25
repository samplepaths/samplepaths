samplepaths
============

|tests| |docs| |license|

.. |tests| image:: https://github.com/samplepaths/samplepaths/actions/workflows/tests.yml/badge.svg
    :target: https://github.com/samplepaths/samplepaths/actions/workflows/tests.yml

.. |docs| image:: https://readthedocs.org/projects/samplepaths/badge/?version=latest
    :target: https://samplepaths.readthedocs.io/

.. |license| image:: https://img.shields.io/badge/license-MIT-blue.svg
    :target: https://github.com/samplepaths/samplepaths/blob/master/LICENSE


A Python package for generating realizations of stochastic processes.

``samplepaths`` is an independently maintained fork of
`stochastic <https://github.com/crflynn/stochastic>`_ by Christopher Flynn,
modernized for current Python, ``numpy`` and ``scipy`` releases and shipped
with type hints. It is maintained by `Bassem Girgis <https://girgis.me>`_.

Installation
------------

Install ``samplepaths`` from its GitHub repository using pip

.. code-block:: shell

    pip install git+https://github.com/samplepaths/samplepaths.git

``samplepaths`` requires Python 3.12 or newer.

Dependencies
~~~~~~~~~~~~

``samplepaths`` uses ``numpy`` for many calculations and ``scipy`` for sampling
specific random variables.

Quickstart
----------

Every process is a class with a ``sample(n)`` method which generates a
realization as a ``numpy`` array.

.. code-block:: python

    from samplepaths import BrownianMotion

    bm = BrownianMotion(drift=1, scale=2, t=10)
    path = bm.sample(1000)  # 1001 values on [0, 10]
    times = bm.times  # the matching time values

Pass a ``numpy.random.Generator`` for reproducible results, or seed the
generator shared by all processes:

.. code-block:: python

    import numpy as np
    from samplepaths import BrownianMotion
    from samplepaths import random

    bm = BrownianMotion(rng=np.random.default_rng(42))

    random.seed(value=42)

Some processes can be sampled at times of your choice, and some have
additional sampling methods:

.. code-block:: python

    from samplepaths import BrownianMotion
    from samplepaths import FractionalGaussianNoise
    from samplepaths import GeometricBrownianMotion

    values = BrownianMotion(drift=1).sample_at([0, 3, 10, 11, 11.2, 20])
    prices = GeometricBrownianMotion(drift=0.05, volatility=0.2).sample_with_initial(252, initial=100.0)
    noise = FractionalGaussianNoise(hurst=0.6).sample_hosking(32)

Processes
---------

All processes can be imported from ``samplepaths`` or from the subpackage of
their family. Diffusion processes are approximated using the Euler-Maruyama
method.

* ``samplepaths.continuous``: BesselProcess, BrownianBridge,
  BrownianExcursion, BrownianMeander, BrownianMotion, CauchyProcess,
  FractionalBrownianMotion, GammaProcess, GeometricBrownianMotion,
  InverseGaussianProcess, MixedPoissonProcess,
  MultifractionalBrownianMotion, PoissonProcess, SquaredBesselProcess,
  VarianceGammaProcess, WienerProcess
* ``samplepaths.diffusion``: DiffusionProcess (generalized),
  ConstantElasticityVarianceProcess, CoxIngersollRossProcess,
  ExtendedVasicekProcess, OrnsteinUhlenbeckProcess, VasicekProcess
* ``samplepaths.discrete``: BernoulliProcess, ChineseRestaurantProcess,
  DirichletProcess, MarkovChain, MoranProcess, RandomWalk
* ``samplepaths.noise``: BlueNoise, BrownianNoise, ColoredNoise,
  FractionalGaussianNoise, GaussianNoise, PinkNoise, RedNoise, VioletNoise,
  WhiteNoise

Documentation
-------------

The documentation is at https://samplepaths.readthedocs.io/. It contains a
user guide for every family of processes and an API reference.

To build it locally, run ``make docs`` and open
``docs/_build/html/index.html``.

Development
-----------

.. code-block:: shell

    make install  # create the environment with uv
    make check    # lint, type check, test and build the docs

Run ``make`` to list all targets.

License
-------

``samplepaths`` is released under the MIT license. See the ``LICENSE`` file.
