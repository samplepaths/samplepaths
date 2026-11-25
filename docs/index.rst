samplepaths
===========

|tests| |docs| |pypi| |license|

.. |tests| image:: https://github.com/samplepaths/samplepaths/actions/workflows/tests.yml/badge.svg
    :target: https://github.com/samplepaths/samplepaths/actions/workflows/tests.yml

.. |docs| image:: https://readthedocs.org/projects/samplepaths/badge/?version=latest
    :target: https://samplepaths.readthedocs.io/

.. |pypi| image:: https://img.shields.io/pypi/v/samplepaths.svg
    :target: https://pypi.org/project/samplepaths/

.. |license| image:: https://img.shields.io/badge/license-MIT-blue.svg
    :target: https://github.com/samplepaths/samplepaths/blob/master/LICENSE


.. image:: /_static/banner.png
    :alt: Realizations of a Brownian motion with drift and their quantile bands
    :width: 100%

``samplepaths`` is a Python package for generating realizations (sample
paths) of stochastic processes as ``numpy`` arrays.

.. code-block:: python

    from samplepaths import BrownianMotion

    bm = BrownianMotion(drift=1, scale=2, t=10)
    path = bm.sample(1000)  # 1001 values on [0, 10]
    paths = bm.sample_paths(1000, paths=5000)  # 5000 realizations at once

It provides 37 processes in four families, all sharing one small interface:

* :doc:`guide/continuous`, such as Brownian motion, the Poisson process and the
  gamma process
* :doc:`guide/diffusion`, such as the Vasicek and Cox-Ingersoll-Ross models
* :doc:`guide/discrete`, such as Markov chains and random walks
* :doc:`guide/noise`, such as Gaussian, fractional Gaussian and colored noise

.. image:: /_static/families.png
    :alt: One example process of each of the four families
    :width: 100%

``samplepaths`` is an independently maintained fork of
`stochastic <https://github.com/crflynn/stochastic>`_ by Christopher Flynn,
modernized for current Python, ``numpy`` and ``scipy`` releases and shipped
with type hints. See :doc:`project/about` for what the fork adds and
:doc:`project/migration` for the changes needed to port existing code.

``samplepaths`` is maintained by `Bassem Girgis <https://girgis.me>`_.

.. toctree::
   :maxdepth: 2
   :caption: Getting started

   getting-started/installation
   getting-started/quickstart

.. toctree::
   :maxdepth: 2
   :caption: User guide

   guide/usage
   guide/paths
   guide/random
   guide/continuous
   guide/diffusion
   guide/discrete
   guide/noise
   guide/performance

.. toctree::
   :maxdepth: 2
   :caption: Reference

   reference/index
   reference/sources

.. toctree::
   :maxdepth: 2
   :caption: Project

   project/about
   project/changelog
   project/migration
   project/development
   project/license

Indices
-------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
