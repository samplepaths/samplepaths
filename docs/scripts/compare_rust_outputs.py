"""Compare the results of samplepaths and of stochastic-rs.

The two packages use different random number generators, so the same seed
does not give the same realization. What can be compared is the distribution
of the results. For every process which both packages provide, this script
generates many realizations with each and compares a summary of each
realization (its final value, for most processes) with a two-sample
Kolmogorov-Smirnov test::

    uv run --with stochastic-rs python docs/scripts/compare_rust_outputs.py

A p-value which is not small means that the test finds no difference between
the two distributions. With 33 comparisons, one p-value below 0.05 is
expected by chance.
"""

import math
import warnings
from collections.abc import Callable

import numpy as np
import numpy.typing as npt
import stochastic_rs as rs
from scipy.stats import ks_2samp

import samplepaths as sp

Array = npt.NDArray[np.float64]

#: Number of increments of the realizations on [0, 1].
STEPS = 64

#: stochastic-rs counts the points of a realization, not its increments.
POINTS = STEPS + 1

#: Number of values of the time series. stochastic-rs starts a series from
#: zeros and samplepaths from the long-run distribution, so the series are
#: long enough for the start to be forgotten.
SERIES = 300


def final_values(process: object, count: int) -> Array:
    """Return the final values of realizations of a stochastic-rs process."""
    return np.asarray(process.sample_par(count))[:, -1]  # type: ignore[attr-defined]


def looped(sample: Callable[[], float], count: int) -> Array:
    """Collect a summary of realizations generated one by one."""
    return np.array([sample() for _ in range(count)])


def cases(rng: np.random.Generator, count: int) -> list[tuple[str, str, Array, Array]]:
    """Return, for every shared process, a summary from each package.

    :returns: the name of the process, what is compared, the values of
        stochastic-rs and the values of samplepaths
    """
    n = STEPS
    few = count // 10  # for the processes generated one realization at a time
    results: list[tuple[str, str, Array, Array]] = []

    def add(name: str, what: str, theirs: Array, ours: Array) -> None:
        results.append((name, what, np.asarray(theirs, dtype=float), np.asarray(ours, dtype=float)))

    final = "final value"
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
    jump_rng = np.random.default_rng(1)
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

    # Time series: the last of SERIES values. The parameters have the same
    # meaning in both packages.
    last = f"value {SERIES}"
    add(
        "AR(2)",
        last,
        final_values(rs.PyARp([0.5, -0.3], 2.0, SERIES), count),
        sp.AutoregressiveProcess(coefficients=[0.5, -0.3], scale=2.0, rng=rng).sample_paths(
            SERIES, count
        )[:, -1],
    )
    add(
        "MA(2)",
        last,
        final_values(rs.PyMAq([0.6, 0.2], 1.5, SERIES), count),
        sp.MovingAverageProcess(coefficients=[0.6, 0.2], scale=1.5, rng=rng).sample_paths(
            SERIES, count
        )[:, -1],
    )
    add(
        "ARIMA(1, 1, 1)",
        last,
        final_values(rs.PyArima([0.5], [0.3], 1, 1.0, SERIES), count),
        sp.ARIMAProcess(ar=0.5, ma=0.3, d=1, rng=rng).sample_paths(SERIES, count)[:, -1],
    )
    add(
        "SARIMA(1, 0, 1)(1, 1, 1)4",
        last,
        final_values(rs.PySarima([0.5], [0.3], [0.2], [0.4], 0, 1, 4, 1.0, SERIES), count),
        sp.SARIMAProcess(
            ar=0.5, ma=0.3, seasonal_ar=0.2, seasonal_ma=0.4, seasonal_d=1, period=4, rng=rng
        ).sample_paths(SERIES, count)[:, -1],
    )
    add(
        "ARCH(1)",
        last,
        final_values(rs.PyArch(0.5, [0.5], SERIES), count),
        sp.ARCHProcess(omega=0.5, alpha=0.5, rng=rng).sample_paths(SERIES, count)[:, -1],
    )
    add(
        "GARCH(1, 1)",
        last,
        final_values(rs.PyGarch(0.1, [0.1], [0.8], SERIES), count),
        sp.GARCHProcess(omega=0.1, alpha=0.1, beta=0.8, rng=rng).sample_paths(SERIES, count)[:, -1],
    )
    add(
        "GARCH(2, 2)",
        last,
        final_values(rs.PyGarch(0.1, [0.05, 0.05], [0.5, 0.3], SERIES), count),
        sp.GARCHProcess(omega=0.1, alpha=[0.05, 0.05], beta=[0.5, 0.3], rng=rng).sample_paths(
            SERIES, count
        )[:, -1],
    )
    add(
        "EGARCH(1, 1)",
        last,
        final_values(rs.PyEgarch(-0.1, [0.2], [-0.1], [0.9], SERIES), count),
        sp.EGARCHProcess(omega=-0.1, alpha=0.2, gamma=-0.1, beta=0.9, rng=rng).sample_paths(
            SERIES, count
        )[:, -1],
    )
    add(
        "TGARCH(1, 1)",
        last,
        final_values(rs.PyTgarch(0.1, [0.05], [0.1], [0.8], SERIES), count),
        sp.TGARCHProcess(omega=0.1, alpha=0.05, gamma=0.1, beta=0.8, rng=rng).sample_paths(
            SERIES, count
        )[:, -1],
    )

    # What stochastic-rs calls AGARCH adds a term for the negative values,
    # which is the threshold model here. The AGARCHProcess of samplepaths
    # shifts the value instead, and has no counterpart in stochastic-rs.
    add(
        "AGARCH of stochastic-rs",
        last,
        final_values(rs.PyAgarch(0.1, [0.05], [0.1], [0.8], SERIES), count),
        sp.TGARCHProcess(omega=0.1, alpha=0.05, gamma=0.1, beta=0.8, rng=rng).sample_paths(
            SERIES, count
        )[:, -1],
    )

    return results


def main() -> None:
    """Run the comparison and print the results."""
    warnings.simplefilter("ignore")
    rng = np.random.default_rng(2026)

    print(
        f"{'process':28s} {'compared':20s} {'rust mean':>10s} {'ours mean':>10s} "
        f"{'rust std':>9s} {'ours std':>9s} {'p-value':>8s}"
    )
    for name, what, theirs, ours in cases(rng, count=40_000):
        p_value = ks_2samp(theirs, ours).pvalue
        print(
            f"{name:28s} {what:20s} {theirs.mean():10.4f} {ours.mean():10.4f} "
            f"{theirs.std():9.4f} {ours.std():9.4f} {p_value:8.3f}"
        )


if __name__ == "__main__":
    main()
