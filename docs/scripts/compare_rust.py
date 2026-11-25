"""Compare the sampling times of samplepaths and of stochastic-rs.

stochastic-rs is a package with a Rust core. It is not a dependency of this
project; the comparison installs it for one run::

    uv run --with stochastic-rs python docs/scripts/compare_rust.py

The processes compared are those which both packages provide with the same
parameters. Times are in milliseconds.
"""

import os
import time
import warnings
from collections.abc import Callable
from functools import partial

import numpy as np
import stochastic_rs as rs

import samplepaths as sp


def measure(function: Callable[[], object], minimum_time: float = 0.5) -> float:
    """Return the mean time of a call in milliseconds."""
    function()
    calls = 0
    start = time.perf_counter()
    while True:
        function()
        calls += 1
        elapsed = time.perf_counter() - start
        if elapsed > minimum_time and calls >= 3:
            return elapsed / calls * 1e3


def _loop(process: object, paths: int) -> list[object]:
    """Generate realizations of a stochastic-rs process one by one."""
    return [process.sample() for _ in range(paths)]  # type: ignore[attr-defined]


def cases(n: int) -> list[tuple[str, object, sp.base.BaseProcess]]:
    """Return the processes of both packages for n increments on [0, 1]."""
    rng = np.random.default_rng(0)
    return [
        ("Gaussian noise", rs.PyGn(n, 1.0), sp.GaussianNoise(rng=rng)),
        ("Brownian motion", rs.PyBm(n, 1.0), sp.BrownianMotion(rng=rng)),
        (
            "Geometric Brownian motion",
            rs.PyGbm(0.05, 0.2, n, 1.0, 1.0),
            sp.GeometricBrownianMotion(drift=0.05, volatility=0.2, rng=rng),
        ),
        (
            "Fractional Gaussian noise",
            rs.PyFgn(0.7, n, 1.0),
            sp.FractionalGaussianNoise(hurst=0.7, rng=rng),
        ),
        (
            "Fractional Brownian motion",
            rs.PyFbm(0.7, n, 1.0),
            sp.FractionalBrownianMotion(hurst=0.7, rng=rng),
        ),
        ("Variance gamma", rs.PyVg(0.0, 1.0, 1.0, n, 0.0, 1.0), sp.VarianceGammaProcess(rng=rng)),
        ("Poisson", rs.PyPoisson(2.0, n), sp.PoissonProcess(rate=2.0, rng=rng)),
        (
            "Vasicek",
            rs.PyVasicek(1.5, 0.5, 0.3, n, 1.0, 1.0),
            sp.VasicekProcess(speed=1.5, mean=0.5, vol=0.3, rng=rng),
        ),
        (
            "Cox-Ingersoll-Ross",
            rs.PyCir(1.0, 1.0, 0.3, n, 1.0, 1.0),
            sp.CoxIngersollRossProcess(speed=1.0, mean=1.0, vol=0.3, rng=rng),
        ),
        (
            "CEV",
            rs.PyCev(0.1, 0.2, 0.5, n, 1.0, 1.0),
            sp.ConstantElasticityVarianceProcess(drift=0.1, vol=0.2, volexp=0.5, rng=rng),
        ),
        (
            "Fractional Vasicek",
            rs.PyFVasicek(0.7, 1.5, 0.5, 0.3, n, 1.0, 1.0),
            sp.FractionalVasicekProcess(hurst=0.7, speed=1.5, mean=0.5, vol=0.3, rng=rng),
        ),
        (
            "Fractional Cox-Ingersoll-Ross",
            rs.PyFcir(0.7, 1.0, 1.0, 0.3, n, 1.0, 1.0),
            sp.FractionalCoxIngersollRossProcess(hurst=0.7, speed=1.0, mean=1.0, vol=0.3, rng=rng),
        ),
        # Time series: n is the number of values in both packages.
        (
            "AR(2)",
            rs.PyARp([0.5, -0.3], 1.0, n),
            sp.AutoregressiveProcess(coefficients=[0.5, -0.3], rng=rng),
        ),
        (
            "ARIMA(1, 1, 1)",
            rs.PyArima([0.5], [0.3], 1, 1.0, n),
            sp.ARIMAProcess(ar=0.5, ma=0.3, d=1, rng=rng),
        ),
        (
            "GARCH(1, 1)",
            rs.PyGarch(0.1, [0.1], [0.8], n),
            sp.GARCHProcess(omega=0.1, alpha=0.1, beta=0.8, rng=rng),
        ),
        (
            "EGARCH(1, 1)",
            rs.PyEgarch(-0.1, [0.2], [-0.1], [0.9], n),
            sp.EGARCHProcess(omega=-0.1, alpha=0.2, gamma=-0.1, beta=0.9, rng=rng),
        ),
    ]


def main() -> None:
    """Run the comparison and print the results."""
    warnings.simplefilter("ignore")
    cores = os.cpu_count() or 1

    n = 4096
    print(f"One realization of {n} increments")
    print(f"{'process':28s} {'rust':>9s} {'ours':>9s} {'ours/rust':>10s}")
    for name, theirs, ours in cases(n):
        rust = measure(theirs.sample)  # type: ignore[attr-defined]
        python = measure(lambda ours=ours: ours.sample(n))
        print(f"{name:28s} {rust:9.3f} {python:9.3f} {python / rust:9.1f}x")

    for paths, n in ((1000, 256), (100_000, 64), (20_000, 1024)):
        print(f"\n{paths} realizations of {n} increments, {cores} cores")
        print(
            f"{'process':28s} {'rust loop':>10s} {'rust par':>9s} {'ours w=1':>9s} "
            f"{f'ours w={cores}':>10s} {'ours/rust':>10s}"
        )
        for name, theirs, ours in cases(n):
            loop = measure(partial(_loop, theirs, paths)) if paths <= 1000 else float("nan")
            parallel = measure(partial(theirs.sample_par, paths))  # type: ignore[attr-defined]
            one = measure(partial(ours.sample_paths, n, paths))
            many = measure(partial(ours.sample_paths, n, paths, workers=cores))
            best = min(one, many)
            print(
                f"{name:28s} {loop:10.2f} {parallel:9.2f} {one:9.2f} {many:10.2f} "
                f"{best / parallel:9.1f}x"
            )


if __name__ == "__main__":
    main()
