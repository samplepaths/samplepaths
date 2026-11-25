Time Series Models
==================

The :py:mod:`samplepaths.timeseries` module provides the models of time
series analysis. Like the :doc:`discrete-time processes <discrete>` they
evolve in steps and have no time interval: ``sample(n)`` generates ``n``
values. They come in two groups.

Overview
--------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Process
     - A value depends on
   * - :py:class:`~samplepaths.timeseries.AutoregressiveProcess`
     - the previous values
   * - :py:class:`~samplepaths.timeseries.MovingAverageProcess`
     - the previous values of the noise
   * - :py:class:`~samplepaths.timeseries.ARIMAProcess`
     - both, after differencing
   * - :py:class:`~samplepaths.timeseries.SARIMAProcess`
     - both, also at multiples of a season length
   * - :py:class:`~samplepaths.timeseries.ARCHProcess`
     - the squares of the previous values, through its variance
   * - :py:class:`~samplepaths.timeseries.GARCHProcess`
     - the squares of the previous values and the previous variances
   * - :py:class:`~samplepaths.timeseries.TGARCHProcess`
     - the same, with more weight on negative values
   * - :py:class:`~samplepaths.timeseries.AGARCHProcess`
     - the same, with the previous values shifted
   * - :py:class:`~samplepaths.timeseries.EGARCHProcess`
     - the sizes and the signs of the previous values, through the logarithm
       of its variance

The first four are **linear models**: Gaussian white noise passed through a
linear filter. The other five are **models with a changing variance**: the
values are uncorrelated, but calm and agitated periods alternate.

Coefficients are given as a sequence, or as a single number for a model of
order one. The order of a model is the number of coefficients.

Linear models
-------------

The linear models are driven by Gaussian white noise with standard deviation
``scale``. The series and the noise are zero before the first value, so a
realization takes a few steps to forget its start.

Autoregressive process
~~~~~~~~~~~~~~~~~~~~~~

.. image:: /_static/autoregressive_process.png
    :scale: 50%

In an :py:class:`~samplepaths.timeseries.AutoregressiveProcess`, or AR(p),
every value is a linear combination of the ``p`` previous values plus noise.

.. code-block:: python

    from samplepaths import AutoregressiveProcess

    ar = AutoregressiveProcess(coefficients=[0.6, 0.3], scale=1.0)
    series = ar.sample(500)

Moving average process
~~~~~~~~~~~~~~~~~~~~~~

.. image:: /_static/moving_average_process.png
    :scale: 50%

In a :py:class:`~samplepaths.timeseries.MovingAverageProcess`, or MA(q),
every value is a linear combination of the current and the ``q`` previous
values of the noise. Values more than ``q`` steps apart are independent.

.. code-block:: python

    from samplepaths import MovingAverageProcess

    ma = MovingAverageProcess(coefficients=[0.6, 0.3])
    series = ma.sample(500)

ARIMA process
~~~~~~~~~~~~~

.. image:: /_static/arima_process.png
    :scale: 50%

An :py:class:`~samplepaths.timeseries.ARIMAProcess` combines the two, and
integrates the result ``d`` times. With ``d=0`` it is an ARMA process, which
settles around zero; with ``d=1`` it wanders like a random walk.

.. code-block:: python

    from samplepaths import ARIMAProcess

    arma = ARIMAProcess(ar=[0.5, -0.2], ma=0.3)
    arima = ARIMAProcess(ar=0.5, ma=0.3, d=1)
    series = arima.sample(500)

Seasonal ARIMA process
~~~~~~~~~~~~~~~~~~~~~~

.. image:: /_static/sarima_process.png
    :scale: 50%

A :py:class:`~samplepaths.timeseries.SARIMAProcess` adds a second set of
coefficients which act at multiples of a season length ``period``, and a
seasonal differencing of order ``seasonal_d``. With monthly data and
``period=12``, one seasonal difference makes every month start from its value
of the year before.

.. code-block:: python

    from samplepaths import SARIMAProcess

    sarima = SARIMAProcess(ar=0.5, seasonal_ma=0.4, seasonal_d=1, period=12)
    months = sarima.sample(120)

Models with a changing variance
-------------------------------

A value of these models is :math:`X_t = \sigma_t z_t`, a standard normal
variate :math:`z_t` times a standard deviation which depends on the past.
They describe the returns of financial assets, whose volatility comes in
clusters. A realization starts from the long-run state of the model, so no
value needs to be discarded.

After ``sample()`` or ``sample_paths()``, the attribute ``volatility`` holds
the standard deviations :math:`\sigma_t` of the values just generated, in an
array of the same shape.

.. code-block:: python

    from samplepaths import GARCHProcess

    garch = GARCHProcess(omega=0.1, alpha=0.1, beta=0.8)
    returns = garch.sample(1000)
    volatility = garch.volatility  # the standard deviation of each value

The parameters of a model are only valid together: those of a GARCH process
must add up to less than one for the variance to settle. They are checked
when the process is created and when one of them is changed. A change which
would make the model invalid raises ``ValueError`` and is not applied.

ARCH process
~~~~~~~~~~~~

.. image:: /_static/arch_process.png
    :scale: 50%

In an :py:class:`~samplepaths.timeseries.ARCHProcess`, the variance is
``omega`` plus the squares of the previous values weighted by ``alpha``. A
large value is followed by a large variance.

.. code-block:: python

    from samplepaths import ARCHProcess

    arch = ARCHProcess(omega=0.5, alpha=[0.3, 0.2])
    returns = arch.sample(1000)

GARCH process
~~~~~~~~~~~~~

.. image:: /_static/garch_process.png
    :scale: 50%

A :py:class:`~samplepaths.timeseries.GARCHProcess` adds the previous
variances, weighted by ``beta``, which makes agitated periods last. Its
long-run variance is ``omega / (1 - sum(alpha) - sum(beta))``.

.. code-block:: python

    from samplepaths import GARCHProcess

    garch = GARCHProcess(omega=0.1, alpha=0.1, beta=0.8)  # long-run variance 1
    returns = garch.sample(1000)

Asymmetric models
~~~~~~~~~~~~~~~~~

Stock markets are more agitated after a fall than after a rise of the same
size. Three models reproduce this leverage effect, each in its own way.

.. image:: /_static/tgarch_process.png
    :scale: 50%

A :py:class:`~samplepaths.timeseries.TGARCHProcess`, the threshold or
GJR-GARCH model, gives the square of a negative value the additional weight
``gamma``.

.. image:: /_static/agarch_process.png
    :scale: 50%

An :py:class:`~samplepaths.timeseries.AGARCHProcess` subtracts ``shift``
from a value before squaring it, so that the variance is lowest after a value
equal to the shift instead of zero.

.. image:: /_static/egarch_process.png
    :scale: 50%

An :py:class:`~samplepaths.timeseries.EGARCHProcess` models the logarithm of
the variance, which keeps the variance positive whatever the signs of the
coefficients. ``alpha`` sets the effect of the size of a value and ``gamma``
the effect of its sign; a negative ``gamma`` makes falls raise the variance
more.

.. code-block:: python

    from samplepaths import AGARCHProcess, EGARCHProcess, TGARCHProcess

    threshold = TGARCHProcess(omega=0.1, alpha=0.05, gamma=0.1, beta=0.8)
    shifted = AGARCHProcess(omega=0.1, alpha=0.1, shift=0.5, beta=0.8)
    exponential = EGARCHProcess(omega=-0.1, alpha=0.2, gamma=-0.1, beta=0.9)

.. note::

   Names vary between packages. What ``stochastic-rs`` calls AGARCH is the
   threshold model, :py:class:`~samplepaths.timeseries.TGARCHProcess` here.

Many realizations
-----------------

``sample_paths(n, paths)`` advances all realizations together, which matters
most for the models with a changing variance: 1000 realizations of 256
values take about 2 milliseconds, against 28 for a loop over ``sample()``.
With ``workers``, the random numbers are generated on several threads. See
:doc:`paths` and :doc:`performance`.

.. code-block:: python

    from samplepaths import GARCHProcess

    garch = GARCHProcess(omega=0.1, alpha=0.1, beta=0.8)
    returns = garch.sample_paths(250, paths=100_000, workers=8)
    worst = returns.sum(axis=1).min()  # the worst year in 100,000
