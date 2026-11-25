"""Comparisons with stochastic-rs: fractional, multi-component and rate models."""

import math

import numpy as np
import stochastic_rs as rs

import samplepaths as sp

from .shared import FINAL, POINTS, STEPS, Add, final_components, final_values, looped


def add_model_cases(
    add: Add, rng: np.random.Generator, count: int, jump_rng: np.random.Generator
) -> None:
    """Compare the fractional models, the models with several components and the rate models.

    :param add: adds a comparison to the results
    :param rng: the generator of the processes of samplepaths
    :param count: the number of realizations to compare
    :param jump_rng: the generator of the jump sizes given to stochastic-rs
    """
    n = STEPS
    few = count // 10  # for the processes generated one realization at a time
    final = FINAL

    # Fractional models, with a Hurst parameter of 0.7. stochastic-rs writes
    # the Jacobi drift as alpha - beta x.
    hurst = 0.7
    add(
        "Fractional Vasicek",
        final,
        final_values(rs.PyFVasicek(hurst, 1.5, 0.5, 0.3, POINTS, 1.0, 1.0), count),
        sp.FractionalVasicekProcess(
            hurst=hurst, speed=1.5, mean=0.5, vol=0.3, rng=rng
        ).sample_paths(n, count)[:, -1],
    )
    add(
        "Fractional Ornstein-Uhlenbeck",
        final,
        final_values(rs.PyFou(hurst, 1.5, 0.0, 0.3, POINTS, 1.0, 1.0), count),
        sp.FractionalOrnsteinUhlenbeckProcess(
            hurst=hurst, speed=1.5, vol=0.3, rng=rng
        ).sample_paths(n, count)[:, -1],
    )
    add(
        "Rough Ornstein-Uhlenbeck",
        "final value, H = 0.3",
        final_values(rs.PyFou(0.3, 1.5, 0.0, 0.3, POINTS, 1.0, 1.0), count),
        sp.FractionalOrnsteinUhlenbeckProcess(hurst=0.3, speed=1.5, vol=0.3, rng=rng).sample_paths(
            n, count
        )[:, -1],
    )
    add(
        "Fractional Cox-Ingersoll-Ross",
        final,
        final_values(rs.PyFcir(hurst, 1.0, 1.0, 0.3, POINTS, 1.0, 1.0), count),
        sp.FractionalCoxIngersollRossProcess(
            hurst=hurst, speed=1.0, mean=1.0, vol=0.3, rng=rng
        ).sample_paths(n, count)[:, -1],
    )
    add(
        "Fractional Jacobi",
        final,
        final_values(rs.PyFJacobi(hurst, 1.0, 2.0, 0.5, POINTS, 0.3, 1.0), count),
        sp.FractionalJacobiProcess(
            hurst=hurst, speed=2.0, mean=0.5, vol=0.5, rng=rng
        ).sample_paths_with_initial(n, count, initial=0.3)[:, -1],
    )

    # Fractional geometric Brownian motion: stochastic-rs applies the Euler
    # method to the equation, which for a Hurst parameter above 1/2 tends to
    # exp(mu t + sigma B). samplepaths returns the solution with the Wick
    # product, whose logarithm is lower by sigma^2 t^(2H) / 2. The Euler
    # method needs a fine grid, so stochastic-rs gets 1024 increments.
    add(
        "Fractional geometric BM",
        "final logarithm",
        np.log(final_values(rs.PyFgbm(hurst, 0.05, 0.2, 1025, 1.0, 1.0), count)),
        np.log(
            sp.FractionalGeometricBrownianMotion(
                hurst=hurst, drift=0.05, volatility=0.2, rng=rng
            ).sample_paths(n, count)[:, -1]
        )
        + 0.2**2 / 2,
    )

    # Models with two components. stochastic-rs applies the Euler method to
    # the Heston and the Bates models, so it gets a fine grid; samplepaths
    # draws the variance from its exact distribution.
    fine = 1025
    their_price, their_variance = final_components(
        rs.PyHeston(2.0, 0.04, 0.3, -0.7, 0.05, fine, 1.0, 0.09, 1.0), count
    )
    ours = sp.HestonModel(
        drift=0.05, speed=2.0, mean=0.04, vol=0.3, correlation=-0.7, initial_variance=0.09, rng=rng
    ).sample_paths(n, count)
    add("Heston, price", "final logarithm", np.log(their_price), np.log(ours[:, 0, -1]))
    add("Heston, variance", final, their_variance, ours[:, 1, -1])

    # Bates: stochastic-rs takes the relative size of a jump, exp(Y) - 1, and
    # the variance drift as alpha - beta V.
    jump_mean, jump_volatility, rate = -0.1, 0.15, 2.0
    bates = rs.PyBates(
        rate,
        math.exp(jump_mean + jump_volatility**2 / 2) - 1,
        2.0 * 0.04,
        2.0,
        0.3,
        -0.7,
        lambda: math.exp(jump_rng.normal(jump_mean, jump_volatility)) - 1,
        fine,
        0.05,
        None,
        None,
        None,
        1.0,
        0.09,
        1.0,
    )
    add(
        "Bates, price",
        "final logarithm",
        looped(lambda: math.log(bates.sample()[0][-1]), few),
        np.log(
            sp.BatesModel(
                drift=0.05,
                speed=2.0,
                mean=0.04,
                vol=0.3,
                correlation=-0.7,
                rate=rate,
                jump_mean=jump_mean,
                jump_volatility=jump_volatility,
                initial_variance=0.09,
                rng=rng,
            ).sample_paths(n, few)[:, 0, -1]
        ),
    )

    # SABR: stochastic-rs calls alpha the volatility of the volatility.
    their_forward, their_volatility = final_components(
        rs.PySabr(0.4, 0.5, -0.3, 257, 1.0, 0.2, 1.0), count
    )
    ours = sp.SABRModel(
        volvol=0.4, beta=0.5, correlation=-0.3, initial_volatility=0.2, rng=rng
    ).sample_paths(256, count)
    add("SABR, forward", final, their_forward, ours[:, 0, -1])
    add("SABR, volatility", final, their_volatility, ours[:, 1, -1])

    # Correlated Brownian motions: the sum of the two has a variance of
    # 2 + 2 rho, so it tests the correlation as well.
    first, second = final_components(rs.PyCbms(0.6, POINTS, 1.0), count)
    ours = sp.CorrelatedBrownianMotions(correlation=0.6, rng=rng).sample_paths(n, count)
    add(
        "Correlated Brownian motions",
        "sum of final values",
        first + second,
        ours[:, :, -1].sum(axis=1),
    )

    # Interest rate models. The exact method of the Cox-Ingersoll-Ross
    # process is compared with the Euler method of stochastic-rs on a fine
    # grid. stochastic-rs generates the Ho-Lee and the two-factor Hull-White
    # models one realization at a time.
    add(
        "Cox-Ingersoll-Ross, exact",
        final,
        final_values(rs.PyCir(1.0, 1.0, 0.3, fine, 1.0, 1.0), count),
        sp.CoxIngersollRossProcess(speed=1.0, mean=1.0, vol=0.3, rng=rng).sample_paths_exact(
            8, count
        )[:, -1],
    )
    ho_lee = rs.PyHoLee(0.3, POINTS, None, 0.5, 1.0)
    add(
        "Ho-Lee",
        final,
        looped(lambda: ho_lee.sample()[-1], few),
        sp.HoLeeProcess(drift=0.5, vol=0.3, rng=rng).sample_paths(n, few)[:, -1],
    )
    hull_white = rs.PyHullWhite2F(lambda time: 0.5, 1.5, 0.3, 0.2, -0.4, 0.8, POINTS, 0.1, 1.0)
    theirs = np.array([[component[-1] for component in hull_white.sample()] for _ in range(few)])
    ours = sp.HullWhiteTwoFactorModel(
        speed=1.5, drift=0.5, vol=0.3, factor_speed=0.8, factor_vol=0.2, correlation=-0.4, rng=rng
    ).sample_paths_with_initial(n, few, initial=0.1)
    add("Hull-White two-factor, rate", final, theirs[:, 0], ours[:, 0, -1])
    add("Hull-White two-factor, factor", final, theirs[:, 1], ours[:, 1, -1])
