Diffusion Processes
===================

The :py:mod:`samplepaths.diffusion` module provides processes defined by a
stochastic differential equation and sampled with the Euler-Maruyama method.

The general model
-----------------

All diffusion processes are special cases of
:py:class:`~samplepaths.diffusion.DiffusionProcess`, which solves

.. math::

    dX_t = \theta_t (\mu_t - X_t) dt + \sigma_t X_t^{\gamma_t} dW_t

where :math:`W_t` is a Wiener process. Its parameters are:

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - Parameter
     - Symbol
     - Meaning
   * - ``speed``
     - :math:`\theta_t`
     - how fast the process reverts to its mean
   * - ``mean``
     - :math:`\mu_t`
     - the level the process reverts to
   * - ``vol``
     - :math:`\sigma_t`
     - the volatility coefficient
   * - ``volexp``
     - :math:`\gamma_t`
     - the exponent applied to the process in the volatility term

Each parameter is either a number or a function of one argument, for
parameters that change over time.

.. code-block:: python

    from samplepaths import DiffusionProcess

    # mean reversion towards a level which rises over time
    dp = DiffusionProcess(
        speed=2.0,
        mean=lambda t: 1 + t,
        vol=0.3,
        volexp=0,
        t=5,
    )
    path = dp.sample(1000)

Named models
------------

The named models fix or rename some of these parameters:

.. list-table::
   :header-rows: 1
   :widths: 40 30 30

   * - Process
     - Parameters
     - Equation
   * - :py:class:`~samplepaths.diffusion.OrnsteinUhlenbeckProcess`
     - ``speed``, ``vol``
     - :math:`dX_t = -\theta X_t dt + \sigma dW_t`
   * - :py:class:`~samplepaths.diffusion.VasicekProcess`
     - ``speed``, ``mean``, ``vol``
     - :math:`dX_t = \theta (\mu - X_t) dt + \sigma dW_t`
   * - :py:class:`~samplepaths.diffusion.ExtendedVasicekProcess`
     - ``speed``, ``mean``, ``vol``, each a number or a function
     - :math:`dX_t = \theta_t (\mu_t - X_t) dt + \sigma_t dW_t`
   * - :py:class:`~samplepaths.diffusion.CoxIngersollRossProcess`
     - ``speed``, ``mean``, ``vol``
     - :math:`dX_t = \theta (\mu - X_t) dt + \sigma \sqrt{X_t} dW_t`
   * - :py:class:`~samplepaths.diffusion.ConstantElasticityVarianceProcess`
     - ``drift``, ``vol``, ``volexp``
     - :math:`dX_t = \mu X_t dt + \sigma X_t^\gamma dW_t`

.. image:: /_static/mean_reversion.png
    :alt: Vasicek realizations from two initial values reverting to the mean
    :width: 100%

The figure shows a Vasicek process started from two initial values. Both
revert to the mean, and the shaded bands hold 90% of the realizations.

.. code-block:: python

    from samplepaths import CoxIngersollRossProcess
    from samplepaths import VasicekProcess

    vasicek = VasicekProcess(speed=1.0, mean=0.05, vol=0.02, t=10)
    rates = vasicek.sample_with_initial(1000, initial=0.03)

    cir = CoxIngersollRossProcess(speed=1.0, mean=0.05, vol=0.1, t=10)
    rates = cir.sample_with_initial(1000, initial=0.03)

The initial value
-----------------

``sample(n)`` starts every diffusion process from an initial value of 1.
Use ``sample_with_initial(n, initial)`` to choose the initial value.

Accuracy
--------

The Euler-Maruyama method is an approximation which improves as the time
step ``t / n`` gets smaller. Keep these points in mind:

* Use enough increments. A fast mean reversion (large ``speed``) or a large
  volatility needs a smaller time step.
* The approximation can leave the domain of the exact process. A
  Cox-Ingersoll-Ross process is nonnegative, but its discretized path can
  step below zero, after which the square root is undefined and the
  realization contains ``nan``. A smaller time step makes this less likely.
* Diffusion processes cannot be sampled at specified times; ``sample_at()``
  raises ``NotImplementedError``.
