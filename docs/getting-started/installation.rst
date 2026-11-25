Installation
============

Requirements
------------

``samplepaths`` requires Python 3.11 or newer. It depends on two packages,
which are installed automatically:

* ``numpy`` 2.2 or newer, used for most calculations
* ``scipy`` 1.15 or newer, used to generate certain random variables

It is tested on Python 3.11, 3.12, 3.13 and 3.14 on Linux, macOS and Windows.

Installing from PyPI
--------------------

Install the latest release from
`PyPI <https://pypi.org/project/samplepaths/>`_ using ``pip``:

.. code-block:: bash

   pip install samplepaths

or add it to a project managed with ``uv``:

.. code-block:: bash

   uv add samplepaths

Installing from GitHub
----------------------

To use changes which are not released yet, install from the
`GitHub repository <https://github.com/samplepaths/samplepaths>`_:

.. code-block:: bash

   pip install git+https://github.com/samplepaths/samplepaths.git

Checking the installation
-------------------------

.. code-block:: python

    import samplepaths

    print(samplepaths.__version__)

Plotting
--------

``samplepaths`` returns plain ``numpy`` arrays and does not depend on a
plotting library. The plotting examples in this documentation use
``matplotlib``, which you can install separately:

.. code-block:: bash

   pip install matplotlib

Installing for development
--------------------------

To work on ``samplepaths`` itself, see :doc:`/project/development`.
