Processes with Several Components
=================================

The :py:mod:`samplepaths.multivariate` module provides processes made of
several quantities which evolve together: correlated Brownian motions, and
the models of a price or a rate together with its random volatility.

Overview
--------

.. list-table::
   :header-rows: 1
   :widths: 44 56

   * - Process
     - Components
   * - :py:class:`~samplepaths.multivariate.CorrelatedBrownianMotions`
     - any number of Brownian motions
   * - :py:class:`~samplepaths.multivariate.HestonModel`
     - a price and its variance
   * - :py:class:`~samplepaths.multivariate.BatesModel`
     - a price with jumps and its variance
   * - :py:class:`~samplepaths.multivariate.SABRModel`
     - a forward and its volatility

The shape of a realization
--------------------------

A realization is a two-dimensional array with one component per row. The
attribute ``components`` names the rows, and ``times`` holds the time values
they share.

.. code-block:: python

    from samplepaths import HestonModel

    heston = HestonModel(drift=0.05, speed=2.0, mean=0.04, vol=0.3, correlation=-0.7)
    price, variance = heston.sample(252)

    print(heston.components)
    # ('price', 'variance')

``sample_paths(n, paths)`` adds the realizations as the first axis: the
result has the shape ``(paths, components, n + 1)``.

.. code-block:: python

    paths = heston.sample_paths(252, paths=10_000)
    prices = paths[:, 0]  # shape (10000, 253)
    variances = paths[:, 1]

Correlated Brownian motions
---------------------------

.. image:: /_static/correlated_brownian_motions.png
    :scale: 50%

:py:class:`~samplepaths.multivariate.CorrelatedBrownianMotions` are standard
Brownian motions whose increments over the same time step are correlated.
Give one number for two motions, or a correlation matrix for more.

.. code-block:: python

    from samplepaths import CorrelatedBrownianMotions

    two = CorrelatedBrownianMotions(correlation=0.8)
    first, second = two.sample(1000)

    three = CorrelatedBrownianMotions(
        correlation=[[1.0, 0.5, 0.2], [0.5, 1.0, -0.3], [0.2, -0.3, 1.0]], t=2
    )
    motions = three.sample(1000)  # shape (3, 1001)

They are the building block of models with several sources of randomness.
Correlated geometric Brownian motions, for a basket of assets, are their
exponentials:

.. code-block:: python

    import numpy as np

    motions = two.sample_paths(252, paths=10_000)
    volatility = np.array([[0.2], [0.3]])
    prices = 100 * np.exp(-volatility**2 / 2 * two.times + volatility * motions)

Heston model
------------

.. image:: /_static/heston_model.png
    :scale: 50%

In a :py:class:`~samplepaths.multivariate.HestonModel` the variance of a
price follows a Cox-Ingersoll-Ross process:

.. math::

    dS_t &= \mu S_t dt + \sqrt{V_t} S_t dW^S_t \\
    dV_t &= \kappa (\theta - V_t) dt + \xi \sqrt{V_t} dW^V_t

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Parameter
     - Symbol
     - Meaning
   * - ``drift``
     - :math:`\mu`
     - the growth rate of the mean price
   * - ``speed``
     - :math:`\kappa`
     - how fast the variance reverts to its mean
   * - ``mean``
     - :math:`\theta`
     - the long-run variance
   * - ``vol``
     - :math:`\xi`
     - the volatility of the variance
   * - ``correlation``
     - :math:`\rho`
     - the correlation of the price and the variance; negative for stocks
   * - ``initial_variance``
     - :math:`V_0`
     - the variance at time zero; the long-run variance if not given

The variance is drawn from its exact distribution at each step, so it is
never negative and does not depend on the number of steps. The price uses
the discretization of Andersen, whose error is small even for a few steps
per year. ``sample_with_initial(n, initial)`` and
``sample_paths_with_initial(n, paths, initial)`` choose the initial price,
which is 1 otherwise.

.. code-block:: python

    from samplepaths import HestonModel

    heston = HestonModel(drift=0.05, speed=2.0, mean=0.04, vol=0.3, correlation=-0.7)
    paths = heston.sample_paths_with_initial(64, paths=100_000, initial=100.0, workers=8)
    final_prices = paths[:, 0, -1]

Bates model
-----------

.. image:: /_static/bates_model.png
    :scale: 50%

A :py:class:`~samplepaths.multivariate.BatesModel` is a Heston model whose
price also jumps, as in a
:py:class:`~samplepaths.continuous.MertonJumpDiffusion`. Jumps arrive at the
rate ``rate``, and the logarithm of a jump is normal with mean ``jump_mean``
and standard deviation ``jump_volatility``. The mean effect of the jumps is
taken out of the drift, so the mean price still grows at the rate ``drift``.

.. code-block:: python

    from samplepaths import BatesModel

    bates = BatesModel(
        drift=0.05, speed=2.0, mean=0.04, vol=0.3, correlation=-0.7,
        rate=0.5, jump_mean=-0.1, jump_volatility=0.15,
    )
    price, variance = bates.sample(252)

SABR model
----------

.. image:: /_static/sabr_model.png
    :scale: 50%

A :py:class:`~samplepaths.multivariate.SABRModel` describes a forward price
or rate :math:`F_t` with a lognormal volatility :math:`\sigma_t`:

.. math::

    dF_t &= \sigma_t F_t^\beta dW^F_t \\
    d\sigma_t &= \nu \sigma_t dW^\sigma_t

The exponent ``beta`` goes from 0, a normal model in which the forward can
become negative, to 1, a lognormal model. ``volvol`` is :math:`\nu`,
``initial_volatility`` is the volatility at time zero, often called
:math:`\alpha`, and ``correlation`` links the two Wiener processes.

.. code-block:: python

    from samplepaths import SABRModel

    sabr = SABRModel(volvol=0.4, beta=0.5, correlation=-0.3, initial_volatility=0.2)
    forward, volatility = sabr.sample_with_initial(252, initial=0.03)

The volatility is exact. For ``beta`` strictly between 0 and 1 the forward
uses the Euler method, and a forward which reaches zero stays there.

Many realizations
-----------------

All four generate their realizations together and accept ``workers``. With 14
threads, 100,000 realizations of 64 increments take 10 milliseconds for two
correlated Brownian motions, 48 for a SABR model and 61 for a Heston model.
The Heston, the Bates and, for ``beta`` between 0 and 1, the SABR models
advance step by step, and use a thread for every 5000 realizations at most.
See :doc:`paths` and :doc:`performance`.
