Migrating from stochastic
=========================

``samplepaths`` is a fork of
`stochastic <https://github.com/crflynn/stochastic>`_ version 0.7.0, written
by Christopher Flynn. The fork keeps the original processes and algorithms
and is developed independently of the original package. See :doc:`refactor`
for what the fork changed and why.

Changes from stochastic 0.7.0
-----------------------------

Code written for ``stochastic`` needs the following changes to run on
``samplepaths``.

* The import name is ``samplepaths`` instead of ``stochastic``, and the
  ``processes`` level of the package was removed:

  .. code-block:: python

      # stochastic
      from stochastic.processes.continuous import BrownianMotion

      # samplepaths
      from samplepaths.continuous import BrownianMotion
      from samplepaths import BrownianMotion  # also possible

* Python 3.12 or newer is required, together with ``numpy`` 2.2 and
  ``scipy`` 1.15 or newer.
* Only :py:class:`numpy.random.Generator` is supported for random number
  generation. Support for the legacy ``numpy.random.RandomState`` and the
  ``use_randomstate()`` function were removed.
* :py:func:`samplepaths.random.seed` and
  :py:func:`samplepaths.random.use_generator` take keyword arguments only,
  for example ``random.seed(value=42)``.
* The continuous-time, diffusion and noise process classes take keyword
  arguments only, for example ``BrownianMotion(drift=1, scale=1, t=1)``.
* The ``algorithm`` argument of ``sample()`` was replaced by one method per
  algorithm: ``sample_davies_harte()`` and ``sample_hosking()``.
* Optional arguments of ``sample()`` moved to dedicated methods, so that
  ``sample(n)`` has the same signature on every process:

  * ``sample_with_initial()`` and ``sample_at_with_initial()`` on
    :py:class:`~samplepaths.continuous.GeometricBrownianMotion`,
    and ``sample_with_initial()`` on the diffusion processes
  * ``sample_with_length()`` on
    :py:class:`~samplepaths.continuous.PoissonProcess` and
    :py:class:`~samplepaths.continuous.MixedPoissonProcess`
  * ``sample_with_endpoint()`` and ``sample_at_with_endpoint()`` on
    :py:class:`~samplepaths.continuous.BrownianMeander`
  * ``sample_with_start()`` on
    :py:class:`~samplepaths.discrete.MoranProcess`

* The ``times(n)`` method was replaced by the ``times`` attribute, which
  holds the time values of the most recent realization.
* ``sample_at()`` raises ``NotImplementedError`` on the processes which
  cannot be sampled at specified times.
* The package ships type hints (``py.typed``).

Behavior fixes
--------------

The following results differ from ``stochastic`` 0.7.0 because of corrected
defects:

* The drift of :py:class:`~samplepaths.continuous.BrownianMotion` realizations
  generated with ``sample()`` is proportional to time for every ``t``.
  Previously the drift term was only correct for ``t = 1``.
* :py:class:`~samplepaths.continuous.GeometricBrownianMotion` realizations
  follow :math:`S_0 \exp((\mu - \sigma^2 / 2) t + \sigma W_t)` for every
  ``t``.
* ``sample_partition()`` of
  :py:class:`~samplepaths.discrete.ChineseRestaurantProcess` numbers the
  first customer 0 instead of 1, consistently with the other customers.
