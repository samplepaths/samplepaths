"""Comparisons with stochastic-rs: continuous-time and diffusion processes."""

import math

import numpy as np
import stochastic_rs as rs

import samplepaths as sp

from .shared import FINAL, POINTS, STEPS, Add, final_values, looped


def add_process_cases(
    add: Add, rng: np.random.Generator, count: int, jump_rng: np.random.Generator
) -> None:
    """Compare the continuous-time and the diffusion processes.

    :param add: adds a comparison to the results
    :param rng: the generator of the processes of samplepaths
    :param count: the number of realizations to compare
    :param jump_rng: the generator of the jump sizes given to stochastic-rs
    """
    n = STEPS
    few = count // 10  # for the processes generated one realization at a time
    final = FINAL

    add(
        "Brownian motion",
        final,
        final_values(rs.PyBm(POINTS, 1.0), count),
        sp.BrownianMotion(rng=rng).sample_paths(n, count)[:, -1],
    )
    add(
        "Geometric Brownian motion",
        final,
        final_values(rs.PyGbm(0.05, 0.2, POINTS, 1.0, 1.0), count),
        sp.GeometricBrownianMotion(drift=0.05, volatility=0.2, rng=rng).sample_paths(n, count)[
            :, -1
        ],
    )
    add(
        "Fractional Brownian motion",
        final,
        final_values(rs.PyFbm(0.7, POINTS, 1.0), count),
        sp.FractionalBrownianMotion(hurst=0.7, rng=rng).sample_paths(n, count)[:, -1],
    )
    add(
        "Vasicek",
        final,
        final_values(rs.PyVasicek(1.5, 0.5, 0.3, POINTS, 1.0, 1.0), count),
        sp.VasicekProcess(speed=1.5, mean=0.5, vol=0.3, rng=rng).sample_paths(n, count)[:, -1],
    )
    add(
        "Cox-Ingersoll-Ross",
        final,
        final_values(rs.PyCir(1.0, 1.0, 0.3, POINTS, 1.0, 1.0), count),
        sp.CoxIngersollRossProcess(speed=1.0, mean=1.0, vol=0.3, rng=rng).sample_paths(n, count)[
            :, -1
        ],
    )
    add(
        "CEV",
        final,
        final_values(rs.PyCev(0.1, 0.2, 0.5, POINTS, 1.0, 1.0), count),
        sp.ConstantElasticityVarianceProcess(drift=0.1, vol=0.2, volexp=0.5, rng=rng).sample_paths(
            n, count
        )[:, -1],
    )
    add(
        "Variance gamma",
        final,
        final_values(rs.PyVg(0.1, 1.0, 1.0, POINTS, 0.0, 1.0), count),
        sp.VarianceGammaProcess(drift=0.1, variance=1.0, scale=1.0, rng=rng).sample_paths(n, count)[
            :, -1
        ],
    )
    add(
        "Poisson",
        f"time of arrival {n}",
        final_values(rs.PyPoisson(2.0, POINTS), count),
        sp.PoissonProcess(rate=2.0, rng=rng).sample_paths(n, count)[:, -1],
    )

    # stochastic-rs writes the Jacobi drift as alpha - beta x.
    add(
        "Jacobi",
        final,
        final_values(rs.PyJacobi(1.0, 2.0, 0.5, POINTS, 0.3, 1.0), count),
        sp.JacobiProcess(speed=2.0, mean=0.5, vol=0.5, rng=rng).sample_paths_with_initial(
            n, count, initial=0.3
        )[:, -1],
    )
    add(
        "Verhulst",
        final,
        final_values(rs.PyVerhulst(1.0, 10.0, 0.2, POINTS, 1.0, 1.0, False), count),
        sp.VerhulstProcess(rate=1.0, capacity=10.0, vol=0.2, rng=rng).sample_paths(n, count)[:, -1],
    )
    add(
        "Gompertz",
        final,
        final_values(rs.PyGompertz(1.0, 0.5, 0.2, POINTS, 1.0, 1.0), count),
        sp.GompertzProcess(growth=1.0, decay=0.5, vol=0.2, rng=rng).sample_paths(n, count)[:, -1],
    )
    add(
        "Kimura",
        final,
        final_values(rs.PyKimura(1.0, 0.5, POINTS, 0.3, 1.0), count),
        sp.KimuraProcess(selection=1.0, vol=0.5, rng=rng).sample_paths_with_initial(
            n, count, initial=0.3
        )[:, -1],
    )
    add(
        "Feller logistic",
        final,
        final_values(rs.PyFellerLogistic(1.0, 2.0, 0.3, POINTS, 1.0, 1.0), count),
        sp.FellerLogisticProcess(speed=1.0, level=2.0, vol=0.3, rng=rng).sample_paths(n, count)[
            :, -1
        ],
    )

    # Normal inverse Gaussian: stochastic-rs uses a drift theta, a volatility
    # sigma and a variance kappa of the clock; samplepaths uses alpha, beta
    # and delta.
    theta, sigma, kappa = 0.1, 0.5, 0.3
    beta = theta / sigma**2
    delta = sigma / math.sqrt(kappa)
    alpha = math.hypot(1.0 / (sigma * math.sqrt(kappa)), beta)
    add(
        "Normal inverse Gaussian",
        final,
        final_values(rs.PyNig(theta, sigma, kappa, POINTS, 0.0, 1.0), count),
        sp.NormalInverseGaussianProcess(alpha=alpha, beta=beta, delta=delta, rng=rng).sample_paths(
            n, count
        )[:, -1],
    )

    # Merton: stochastic-rs simulates the logarithm of the price, with a
    # compensator theta given by the caller.
    drift, volatility, rate, jump_mean, jump_volatility = 0.08, 0.2, 3.0, -0.1, 0.3
    mean_relative_jump = math.exp(jump_mean + jump_volatility**2 / 2) - 1
    merton = rs.PyMerton(
        drift,
        volatility,
        rate,
        mean_relative_jump,
        lambda: float(jump_rng.normal(jump_mean, jump_volatility)),
        POINTS,
        0.0,
        1.0,
    )
    add(
        "Merton jump diffusion",
        "final logarithm",
        looped(lambda: merton.sample()[-1], few),
        np.log(
            sp.MertonJumpDiffusion(
                drift=drift,
                volatility=volatility,
                rate=rate,
                jump_mean=jump_mean,
                jump_volatility=jump_volatility,
                rng=rng,
            ).sample_paths(n, few)[:, -1]
        ),
    )

    # Compound Poisson: stochastic-rs returns the jump times, the values and
    # the jump sizes up to a time.
    compound = rs.PyCompoundPoisson(lambda: float(jump_rng.normal(-0.1, 0.3)), 4.0, None, 2.0)
    add(
        "Compound Poisson",
        "value at time 2",
        looped(lambda: compound.sample()[1][-1], few),
        sp.CompoundPoissonProcess(
            rate=4.0,
            jumps=lambda generator, size: generator.normal(-0.1, 0.3, size),
            t=2.0,
            rng=rng,
        ).sample_paths(n, few)[:, -1],
    )

    # Hawkes: number of arrivals in [0, 20]. Both results start with time 0;
    # the one of samplepaths also ends with the first arrival after 20.
    hawkes = sp.HawkesProcess(baseline=2.0, excitation=1.5, decay=3.0, rng=rng)
    add(
        "Hawkes",
        "arrivals in [0, 20]",
        looped(lambda: len(rs.PyHawkes(2.0, 1.5, 3.0, None, 20.0).sample()) - 1, few),
        looped(lambda: len(hawkes.sample_with_length(20.0)) - 2, few),
    )
