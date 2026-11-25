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

Each parameter is either a number or a function of one argument, the time,
for parameters that change over time.
Give a constant parameter as a number: a function is called at every step,
which makes sampling slower.

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

Diffusions on a domain
----------------------

Five more diffusion processes have a drift or a volatility which the general
model above does not cover. Each lives on a domain, the unit interval or the
positive numbers, and their parameters are numbers.

.. list-table::
   :header-rows: 1
   :widths: 34 26 40

   * - Process
     - Parameters
     - Equation
   * - :py:class:`~samplepaths.diffusion.JacobiProcess`
     - ``speed``, ``mean``, ``vol``
     - :math:`dX_t = \theta (\mu - X_t) dt + \sigma \sqrt{X_t (1 - X_t)} dW_t`
   * - :py:class:`~samplepaths.diffusion.KimuraProcess`
     - ``selection``, ``vol``
     - :math:`dX_t = s X_t (1 - X_t) dt + \sigma \sqrt{X_t (1 - X_t)} dW_t`
   * - :py:class:`~samplepaths.diffusion.VerhulstProcess`
     - ``rate``, ``capacity``, ``vol``
     - :math:`dX_t = r X_t (1 - X_t / K) dt + \sigma X_t dW_t`
   * - :py:class:`~samplepaths.diffusion.GompertzProcess`
     - ``growth``, ``decay``, ``vol``
     - :math:`dX_t = (a - b \ln X_t) X_t dt + \sigma X_t dW_t`
   * - :py:class:`~samplepaths.diffusion.FellerLogisticProcess`
     - ``speed``, ``level``, ``vol``
     - :math:`dX_t = \kappa (\theta - X_t) X_t dt + \sigma \sqrt{X_t} dW_t`

The Jacobi and the Kimura processes stay in ``[0, 1]`` and model
proportions: a correlation, or the frequency of an allele in a population.
The Verhulst, the Gompertz and the Feller logistic processes are positive
and model the size of a population which grows towards a limit.

.. code-block:: python

    from samplepaths import JacobiProcess
    from samplepaths import VerhulstProcess

    correlation = JacobiProcess(speed=2.0, mean=0.6, vol=0.4, t=5)
    path = correlation.sample_with_initial(1000, initial=0.2)

    population = VerhulstProcess(rate=1.5, capacity=1000.0, vol=0.1, t=10)
    sizes = population.sample_paths_with_initial(1000, paths=500, initial=10.0)

``sample()`` starts the Jacobi and the Kimura processes from 0.5 and the
others from 1. A step of the Euler-Maruyama method can leave the domain, so
every value is brought back into it: a value which would be negative becomes
zero, for example. For the Kimura and the Feller logistic processes the
boundaries are absorbing, as in the processes they approximate: an allele
which is lost stays lost, and a population which is extinct stays extinct.

Fractional diffusions
---------------------

Four diffusion processes are driven by a
:py:class:`fractional Brownian motion <samplepaths.continuous.FractionalBrownianMotion>`
:math:`B^H_t` instead of a Wiener process. The Hurst parameter ``hurst``
sets the memory of the noise: above 0.5 its increments are positively
correlated and the process has a long memory, below 0.5 they are negatively
correlated and the process is rough, as volatility is observed to be.

.. list-table::
   :header-rows: 1
   :widths: 44 56

   * - Process
     - Equation
   * - :py:class:`~samplepaths.diffusion.FractionalOrnsteinUhlenbeckProcess`
     - :math:`dX_t = -\theta X_t dt + \sigma dB^H_t`
   * - :py:class:`~samplepaths.diffusion.FractionalVasicekProcess`
     - :math:`dX_t = \theta (\mu - X_t) dt + \sigma dB^H_t`
   * - :py:class:`~samplepaths.diffusion.FractionalCoxIngersollRossProcess`
     - :math:`dX_t = \theta (\mu - X_t) dt + \sigma \sqrt{X_t} dB^H_t`
   * - :py:class:`~samplepaths.diffusion.FractionalJacobiProcess`
     - :math:`dX_t = \theta (\mu - X_t) dt + \sigma \sqrt{X_t (1 - X_t)} dB^H_t`

.. image:: /_static/fractional_ornstein_uhlenbeck_process.png
    :scale: 50%

.. code-block:: python

    from samplepaths import FractionalOrnsteinUhlenbeckProcess
    from samplepaths import FractionalVasicekProcess

    rough = FractionalOrnsteinUhlenbeckProcess(hurst=0.1, speed=0.5, vol=0.3)
    log_volatility = rough.sample(1000)

    rates = FractionalVasicekProcess(hurst=0.8, speed=1.0, mean=0.03, vol=0.01, t=10)
    paths = rates.sample_paths_with_initial(1000, paths=5000, initial=0.02)

Their parameters are numbers, and ``hurst`` can be changed after creation.
The realizations use exact increments of the fractional Brownian motion,
generated with the Davies-Harte method, in the Euler method. The method is
known to converge for a Hurst parameter above 0.5. Below 0.5 the two
processes with a square root in their volatility tend to stay at the
boundary of their domain once they reach it. The two linear processes,
Ornstein-Uhlenbeck and Vasicek, are computed for all steps at once, and use
every thread given by ``workers``.

The fractional counterpart of the geometric Brownian motion is exact and is
described with the :doc:`continuous-time processes <continuous>`.

Ho-Lee process
--------------

.. image:: /_static/ho_lee_process.png
    :scale: 50%

A :py:class:`~samplepaths.diffusion.HoLeeProcess` is a model of the
instantaneous interest rate which does not revert to a mean:

.. math::

    dX_t = \theta_t dt + \sigma dW_t

Its ``drift`` is a number or a function of time; as a function it lets the
model reproduce the interest rates observed today for every maturity. The
noise is exact. ``sample()`` starts from 0.

.. code-block:: python

    from samplepaths import HoLeeProcess

    rates = HoLeeProcess(drift=lambda t: 0.01 + 0.002 * t, vol=0.01, t=10)
    paths = rates.sample_paths_with_initial(120, paths=5000, initial=0.03)

The Hull-White model, which adds reversion to a mean, is the
:py:class:`~samplepaths.diffusion.ExtendedVasicekProcess` above with a mean
given as a function of time. Models of the rate with two factors are
described with the :doc:`processes with several components <multivariate>`.

Exact sampling
--------------

The Vasicek, the Ornstein-Uhlenbeck and the Cox-Ingersoll-Ross processes can
also be drawn from their exact distributions, on any grid of times, when
their parameters are numbers. See :doc:`exact`.

The initial value
-----------------

``sample(n)`` starts the diffusion processes of the general model from an
initial value of 1.
Use ``sample_with_initial(n, initial)`` to choose the initial value.

An initial value outside the domain of the volatility term gives a
realization of ``nan`` values, with a ``RuntimeWarning``. This is the case
for a negative initial value with a fractional exponent ``volexp``.

Many realizations
-----------------

``sample_paths(n, paths)`` and ``sample_paths_with_initial(n, paths,
initial)`` advance all realizations together, one time step at a time. This
is the fastest way to simulate a diffusion process: 1000 realizations of 256
increments of a Vasicek process take 1.8 milliseconds, instead of 14.6 with
a loop over ``sample()``.

.. code-block:: python

    from samplepaths import CoxIngersollRossProcess

    cir = CoxIngersollRossProcess(speed=1.0, mean=0.05, vol=0.1, t=10)
    rates = cir.sample_paths_with_initial(n=1000, paths=5000, initial=0.03)

    print(rates.shape)
    # (5000, 1001)

See :doc:`paths`.

Accuracy
--------

The Euler-Maruyama method is an approximation which improves as the time
step ``t / n`` gets smaller. Keep these points in mind:

* Use enough increments. A fast mean reversion (large ``speed``) or a large
  volatility needs a smaller time step.
* The approximation can leave the domain of the exact process. A
  Cox-Ingersoll-Ross process is nonnegative, but its discretized path can
  step below zero, after which the square root is undefined and the
  realization contains ``nan``. A smaller time step makes this less likely,
  and its exact method, described above, cannot leave the domain.
* Only the processes with an exact method, and the Ho-Lee process, can be
  sampled at specified times. For the others ``sample_at()`` raises
  ``NotImplementedError``.
