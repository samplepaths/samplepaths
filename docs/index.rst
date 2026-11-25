samplepaths
===========

|tests| |docs| |pypi| |pyversions| |weekly| |total| |license|

.. |tests| image:: https://github.com/samplepaths/samplepaths/actions/workflows/tests.yml/badge.svg
    :target: https://github.com/samplepaths/samplepaths/actions/workflows/tests.yml
    :alt: Tests

.. |docs| image:: https://readthedocs.org/projects/samplepaths/badge/?version=latest
    :target: https://samplepaths.readthedocs.io/
    :alt: Documentation

.. |pypi| image:: https://img.shields.io/pypi/v/samplepaths
    :target: https://pypi.org/project/samplepaths/
    :alt: PyPI version

.. |pyversions| image:: https://img.shields.io/pypi/pyversions/samplepaths
    :target: https://pypi.org/project/samplepaths/
    :alt: Python versions

.. |weekly| image:: https://img.shields.io/pypi/dw/samplepaths?label=downloads%2Fweek
    :target: https://pypistats.org/packages/samplepaths
    :alt: Weekly downloads

.. |total| image:: https://img.shields.io/pepy/dt/samplepaths?label=total%20downloads
    :target: https://pepy.tech/projects/samplepaths
    :alt: Total downloads

.. |license| image:: https://img.shields.io/pypi/l/samplepaths
    :target: https://github.com/samplepaths/samplepaths/blob/master/LICENSE
    :alt: License


.. image:: /_static/banner.png
    :alt: Realizations of a Brownian motion with drift and their quantile bands
    :width: 100%

``samplepaths`` is a Python package for generating realizations (sample
paths) of stochastic processes as ``numpy`` arrays. It is pure Python on
``numpy``, and for large simulations as fast as compiled code.

.. code-block:: python

    from samplepaths import BrownianMotion

    bm = BrownianMotion(drift=1, scale=2, t=10)
    path = bm.sample(1000)  # 1001 values on [0, 10]
    paths = bm.sample_paths(1000, paths=5000)  # 5000 realizations at once
    paths = bm.sample_paths(1000, paths=50_000, workers=8)  # on 8 threads

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
   gallery/index
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
   project/comparison
   project/changelog
   project/migration
   project/development
   project/license

Indices
-------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
