"""Processes and parameters shared by the tests of sample_paths.

Also read by ``.github/scripts/seeded_results.py``, so this module only
needs ``numpy`` and ``samplepaths``.
"""

from collections.abc import Callable
from typing import Any

import numpy as np

import samplepaths
from samplepaths.base import BaseProcess

T5 = [
    [0.1, 0.2, 0.3, 0.25, 0.15],
    [0.3, 0.3, 0.1, 0.2, 0.1],
    [0.2, 0.2, 0.2, 0.2, 0.2],
    [0.05, 0.05, 0.4, 0.4, 0.1],
    [0.5, 0.1, 0.1, 0.1, 0.2],
]

# One or more parameter sets per process, away from the defaults where the
# defaults would hide a term (a zero drift, for example).
CASES: list[tuple[str, dict[str, Any]]] = [
    ("BesselProcess", {"dim": 3, "t": 2.0}),
    ("CompoundPoissonProcess", {"rate": 3.0, "t": 2.0}),
    (
        "CompoundPoissonProcess",
        {"rate": 3.0, "jumps": lambda rng, size: rng.exponential(2.0, size), "t": 2.0},
    ),
    ("HawkesProcess", {"baseline": 2.0, "excitation": 1.5, "decay": 3.0}),
    (
        "KouJumpDiffusion",
        {"drift": 0.1, "volatility": 0.2, "rate": 3.0, "up_probability": 0.4, "t": 2.0},
    ),
    (
        "MertonJumpDiffusion",
        {"drift": 0.1, "volatility": 0.2, "rate": 3.0, "jump_mean": -0.1, "t": 2.0},
    ),
    ("NormalInverseGaussianProcess", {"alpha": 2.0, "beta": 0.5, "delta": 1.5, "mu": 0.1}),
    ("BrownianBridge", {"b": 1.5, "t": 2.0}),
    ("BrownianExcursion", {"t": 2.0}),
    ("BrownianMeander", {"t": 2.0}),
    ("BrownianMotion", {"drift": 0.7, "scale": 1.3, "t": 2.0}),
    ("CauchyProcess", {"t": 2.0}),
    ("FractionalBrownianMotion", {"hurst": 0.7, "t": 2.0}),
    ("FractionalBrownianMotion", {"hurst": 0.5}),
    ("GammaProcess", {"mean": 1.5, "variance": 2.0, "t": 2.0}),
    ("GeometricBrownianMotion", {"drift": 0.3, "volatility": 0.4, "t": 2.0}),
    ("InverseGaussianProcess", {"mean": lambda t: 2 * t + t * t, "scale": 1.5, "t": 2.0}),
    ("MixedPoissonProcess", {"rate_func": None, "rate_args": (1, 5)}),
    ("MultifractionalBrownianMotion", {"hurst": lambda t: 0.3 + 0.2 * t, "t": 2.0}),
    ("PoissonProcess", {"rate": 2.5}),
    ("SquaredBesselProcess", {"dim": 3, "t": 2.0}),
    ("VarianceGammaProcess", {"drift": 0.2, "variance": 1.5, "scale": 0.8, "t": 2.0}),
    ("WienerProcess", {"t": 2.0}),
    ("ConstantElasticityVarianceProcess", {"drift": 0.1, "vol": 0.2, "volexp": 0.5, "t": 2.0}),
    ("CoxIngersollRossProcess", {"speed": 1.0, "mean": 1.0, "vol": 0.3, "t": 2.0}),
    ("DiffusionProcess", {"speed": 2.0, "mean": lambda t: 1 + t, "vol": 0.3, "t": 2.0}),
    ("DiffusionProcess", {"speed": 1.0, "mean": 1.0, "vol": 0.2, "volexp": 1.0, "t": 2.0}),
    ("DiffusionProcess", {"vol": 0.2, "volexp": lambda t: 0.5 if t > 0.5 else 1.0}),
    ("ExtendedVasicekProcess", {"speed": lambda t: 1 + t, "mean": 0.5, "vol": 0.3, "t": 2.0}),
    ("OrnsteinUhlenbeckProcess", {"speed": 1.5, "vol": 0.3, "t": 2.0}),
    ("VasicekProcess", {"speed": 1.5, "mean": 0.5, "vol": 0.3, "t": 2.0}),
    ("FellerLogisticProcess", {"speed": 1.0, "level": 2.0, "vol": 0.3, "t": 2.0}),
    ("GompertzProcess", {"growth": 1.0, "decay": 0.5, "vol": 0.2, "t": 2.0}),
    ("JacobiProcess", {"speed": 2.0, "mean": 0.4, "vol": 0.5, "t": 2.0}),
    ("KimuraProcess", {"selection": 1.0, "vol": 0.5, "t": 2.0}),
    ("VerhulstProcess", {"rate": 1.5, "capacity": 20.0, "vol": 0.2, "t": 2.0}),
    ("AGARCHProcess", {"omega": 0.1, "alpha": 0.1, "shift": 0.5, "beta": 0.8}),
    ("ARCHProcess", {"omega": 0.5, "alpha": [0.3, 0.2]}),
    ("ARIMAProcess", {"ar": 0.5, "ma": 0.3, "d": 1}),
    ("AutoregressiveProcess", {"coefficients": [0.5, -0.3], "scale": 2.0}),
    ("EGARCHProcess", {"omega": -0.1, "alpha": 0.2, "gamma": -0.1, "beta": 0.9}),
    ("GARCHProcess", {"omega": 0.1, "alpha": [0.05, 0.05], "beta": [0.5, 0.3]}),
    ("MovingAverageProcess", {"coefficients": [0.6, 0.2]}),
    (
        "SARIMAProcess",
        {"ar": 0.5, "ma": 0.3, "seasonal_ar": 0.2, "seasonal_d": 1, "period": 4},
    ),
    ("TGARCHProcess", {"omega": 0.1, "alpha": 0.05, "gamma": 0.1, "beta": 0.8}),
    ("FractionalCoxIngersollRossProcess", {"hurst": 0.7, "speed": 1.0, "mean": 1.0, "vol": 0.3}),
    ("FractionalGeometricBrownianMotion", {"hurst": 0.7, "drift": 0.05, "volatility": 0.2}),
    ("FractionalJacobiProcess", {"hurst": 0.7, "speed": 2.0, "mean": 0.5, "vol": 0.5}),
    ("FractionalOrnsteinUhlenbeckProcess", {"hurst": 0.3, "speed": 1.5, "vol": 0.3}),
    ("FractionalVasicekProcess", {"hurst": 0.7, "speed": 1.5, "mean": 0.5, "vol": 0.3}),
    (
        "BatesModel",
        {"drift": 0.1, "speed": 2.0, "vol": 0.3, "rate": 3.0, "initial_variance": 0.09, "t": 2.0},
    ),
    ("CorrelatedBrownianMotions", {"correlation": [[1, 0.5, 0.2], [0.5, 1, -0.3], [0.2, -0.3, 1]]}),
    ("HestonModel", {"drift": 0.1, "speed": 2.0, "vol": 0.3, "initial_variance": 0.09, "t": 2.0}),
    ("SABRModel", {"volvol": 0.4, "beta": 0.5, "correlation": -0.3}),
    ("SABRModel", {"volvol": 0.4, "beta": 0.0, "correlation": 0.3}),
    ("SABRModel", {"volvol": 0.4, "beta": 1.0}),
    ("HoLeeProcess", {"drift": lambda t: 0.5 + t, "vol": 0.3, "t": 2.0}),
    (
        "HullWhiteTwoFactorModel",
        {"speed": 1.5, "drift": 0.5, "vol": 0.3, "factor_vol": 0.2, "correlation": -0.4},
    ),
    (
        "TwoFactorCoxIngersollRossModel",
        {"speed": (2.0, 0.2), "mean": (0.02, 0.03), "vol": (0.1, 0.05), "initial": (0.05, 0.01)},
    ),
    ("BernoulliProcess", {"p": 0.3}),
    ("ChineseRestaurantProcess", {"discount": 0.5, "strength": 2.0}),
    ("DirichletProcess", {"alpha": 5}),
    ("MarkovChain", {"transition": T5, "initial": [0.1, 0.1, 0.5, 0.2, 0.1]}),
    ("MoranProcess", {"maximum": 6}),
    ("RandomWalk", {"steps": [-1, 0, 2], "weights": [1, 1, 2]}),
    ("BlueNoise", {"t": 2.0}),
    ("BrownianNoise", {"t": 2.0}),
    ("ColoredNoise", {"beta": 1.5, "t": 2.0}),
    ("FractionalGaussianNoise", {"hurst": 0.3, "t": 2.0}),
    ("FractionalGaussianNoise", {"hurst": 0.5}),
    ("GaussianNoise", {"t": 2.0}),
    ("PinkNoise", {"t": 2.0}),
    ("RedNoise", {"t": 2.0}),
    ("VioletNoise", {"t": 2.0}),
    ("WhiteNoise", {"t": 2.0}),
]

IDS = [f"{name}-{index}" for index, (name, _) in enumerate(CASES)]


def make(name: str, kwargs: dict[str, Any], seed: int) -> BaseProcess:
    rng = np.random.default_rng(seed)
    if name == "MixedPoissonProcess":
        # Draw the random rate from the generator of the process.
        kwargs = {**kwargs, "rate_func": rng.uniform}
    process: BaseProcess = getattr(samplepaths, name)(rng=rng, **kwargs)
    return process


def one_by_one(process: BaseProcess, n: int, paths: int) -> Any:
    """Generate realizations with sample, padded like sample_paths."""
    rows = []
    for _ in range(paths):
        row = process.sample(n)
        if isinstance(process, samplepaths.MoranProcess):
            # The process stays in the state in which it was absorbed.
            row = np.concatenate((row, np.full(n - len(row), row[-1])))
        rows.append(row)
    return np.array(rows)


def flat(realizations: Any) -> Any:
    """Put the components of every realization one after the other."""
    return realizations.reshape(len(realizations), -1)


def _value_at(index: int) -> Callable[[Any], Any]:
    return lambda realizations: realizations[:, index]


def statistics(length: int) -> dict[str, Callable[[Any], Any]]:
    """Quantities to compare: values at several times and path functionals."""
    columns = sorted({1 if length > 1 else 0, length // 2, length - 1})
    quantities = {f"value at index {column}": _value_at(column) for column in columns}
    quantities["largest value"] = lambda a: np.max(a, axis=1)
    quantities["change over the second half"] = lambda a: a[:, -1] - a[:, length // 2]
    quantities["sum of squared changes"] = lambda a: np.sum(np.diff(a, axis=1) ** 2, axis=1)
    return quantities
