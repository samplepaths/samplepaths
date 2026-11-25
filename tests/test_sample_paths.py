"""Tests of sample_paths, which generates several realizations at once."""

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
from scipy.stats import ks_2samp

import samplepaths
from samplepaths.base import BaseProcess, BaseTimeProcess

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


def _column(index: int) -> Callable[[Any], Any]:
    return lambda realizations: realizations[:, index]


def _statistics(length: int) -> dict[str, Callable[[Any], Any]]:
    """Quantities to compare: values at several times and path functionals."""
    columns = sorted({1 if length > 1 else 0, length // 2, length - 1})
    statistics = {f"value at index {column}": _column(column) for column in columns}
    statistics["largest value"] = lambda a: np.max(a, axis=1)
    statistics["change over the second half"] = lambda a: a[:, -1] - a[:, length // 2]
    statistics["sum of squared changes"] = lambda a: np.sum(np.diff(a, axis=1) ** 2, axis=1)
    return statistics


def test_cases_cover_every_process() -> None:
    names = {name for name in samplepaths.__all__ if isinstance(getattr(samplepaths, name), type)}

    assert {name for name, _ in CASES} == names


@pytest.mark.parametrize("paths", [1, 5, 40])
@pytest.mark.parametrize("n", [1, 2, 7, 8])
@pytest.mark.parametrize("name, kwargs", CASES, ids=IDS)
def test_sample_paths_shape(name: str, kwargs: dict[str, Any], n: int, paths: int) -> None:
    process = make(name, kwargs, seed=1)
    if n == 1 and isinstance(process, samplepaths.ColoredNoise):
        # Colored noise has a zero mean, which leaves nothing for one value.
        with pytest.raises(ValueError):
            process.sample_paths(n, paths)
        return

    single = process.sample(n)

    realizations = process.sample_paths(n, paths)

    expected_length = n if name == "MoranProcess" else single.shape[-1]
    expected_shape = single.shape[:-1] + (expected_length,)
    assert realizations.shape == (paths, *expected_shape)
    assert realizations.dtype == single.dtype
    if isinstance(process, BaseTimeProcess):
        assert process.n == n
        np.testing.assert_allclose(process.times, np.linspace(0, process.t, n + 1))


@pytest.mark.parametrize("paths", [5, 40])
@pytest.mark.parametrize("name, kwargs", CASES, ids=IDS)
def test_sample_paths_reproducible(name: str, kwargs: dict[str, Any], paths: int) -> None:
    first = make(name, kwargs, seed=7).sample_paths(12, paths)
    second = make(name, kwargs, seed=7).sample_paths(12, paths)

    np.testing.assert_array_equal(first, second)
    if name not in ("BernoulliProcess", "MarkovChain", "MoranProcess", "ChineseRestaurantProcess"):
        # The realizations are distinct from each other.
        assert len(np.unique(first, axis=0)) == paths


@pytest.mark.parametrize("paths", [20, 3000])
@pytest.mark.parametrize("name, kwargs", CASES, ids=IDS)
def test_sample_paths_distribution(name: str, kwargs: dict[str, Any], paths: int) -> None:
    # The realizations of sample_paths have the distribution of those of
    # sample. Compare the two at several times, and compare a quantity which
    # depends on the whole path, for the code paths used for few and for many
    # realizations.
    n = 12
    repeats = 3000 // paths

    process = make(name, kwargs, seed=11)
    together = np.concatenate([process.sample_paths(n, paths) for _ in range(repeats)])
    separate = one_by_one(make(name, kwargs, seed=13), n, len(together))

    assert together.shape == separate.shape

    for label, statistic in _statistics(flat(together).shape[1]).items():
        result = ks_2samp(statistic(flat(together)), statistic(flat(separate)))
        assert result.pvalue > 1e-4, f"{label}: p-value {result.pvalue}"


@pytest.mark.parametrize("paths", [5, 200])
def test_sample_paths_with_initial(paths: int) -> None:
    gbm = samplepaths.GeometricBrownianMotion(drift=0.3, volatility=1e-12, t=2.0)
    realizations = gbm.sample_paths_with_initial(10, paths, initial=3.0)

    assert realizations.shape == (paths, 11)
    np.testing.assert_allclose(
        realizations,
        np.tile(3.0 * np.exp(0.3 * np.linspace(0, 2.0, 11)), (paths, 1)),
        rtol=1e-6,
    )

    vasicek = samplepaths.VasicekProcess(speed=1.0, mean=0.5, vol=1e-12)
    realizations = vasicek.sample_paths_with_initial(200, paths, initial=2.0)

    assert realizations.shape == (paths, 201)
    np.testing.assert_allclose(realizations[:, 0], 2.0)
    # Without noise the process decays towards its mean.
    np.testing.assert_allclose(realizations[:, -1], 0.5 + 1.5 * np.exp(-1.0), rtol=1e-2)


@pytest.mark.parametrize(
    "paths, error_type",
    [
        (0, ValueError),
        (-2, ValueError),
        (1.5, TypeError),
    ],
)
@pytest.mark.parametrize("name, kwargs", CASES, ids=IDS)
def test_sample_paths_invalid_paths(
    name: str,
    kwargs: dict[str, Any],
    paths: Any,
    error_type: type,
) -> None:
    process = make(name, kwargs, seed=1)

    with pytest.raises(error_type):
        process.sample_paths(8, paths)


def test_markov_chain_sample_paths_same_for_few_and_many_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Both code paths use the same uniform variates in the same way.
    import samplepaths.discrete.markov_chain as module

    kwargs: dict[str, Any] = {"transition": T5, "initial": [0.1, 0.1, 0.5, 0.2, 0.1]}
    paths = 2 * module._VECTORIZE_FROM_PATHS

    together = samplepaths.MarkovChain(rng=np.random.default_rng(3), **kwargs)
    vectorized = together.sample_paths(30, paths)

    monkeypatch.setattr(module, "_VECTORIZE_FROM_PATHS", 10 * paths)
    in_turn = samplepaths.MarkovChain(rng=np.random.default_rng(3), **kwargs)

    np.testing.assert_array_equal(in_turn.sample_paths(30, paths), vectorized)


@pytest.mark.parametrize("volexp", [0.0, 1.0, 0.5])
def test_diffusion_sample_paths_same_for_few_and_many_paths(
    volexp: float,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Both code paths apply the same method to the same noise.
    import samplepaths.diffusion.diffusion_process as module

    kwargs: dict[str, Any] = {
        "speed": lambda t: 1 + t,
        "mean": 1.0,
        "vol": 0.3,
        "volexp": volexp,
        "t": 2.0,
    }
    paths = 2 * module._VECTORIZE_FROM_PATHS

    together = samplepaths.DiffusionProcess(rng=np.random.default_rng(3), **kwargs)
    vectorized = together.sample_paths_with_initial(50, paths, initial=1.5)

    monkeypatch.setattr(module, "_VECTORIZE_FROM_PATHS", 10 * paths)
    in_turn = samplepaths.DiffusionProcess(rng=np.random.default_rng(3), **kwargs)

    np.testing.assert_allclose(
        in_turn.sample_paths_with_initial(50, paths, initial=1.5),
        vectorized,
        rtol=1e-12,
    )


@pytest.fixture
def small_batches_use_threads(monkeypatch: pytest.MonkeyPatch) -> None:
    """Let sample_paths start threads for the small batches of the tests."""
    import samplepaths.base.base_process as base_module
    import samplepaths.diffusion.diffusion_process as diffusion_module

    monkeypatch.setattr(base_module, "_VALUES_PER_WORKER", 1)
    monkeypatch.setattr(diffusion_module, "_STEPPING_PATHS_PER_WORKER", 1)


@pytest.mark.usefixtures("small_batches_use_threads")
@pytest.mark.parametrize("workers", [2, 5])
@pytest.mark.parametrize("name, kwargs", CASES, ids=IDS)
def test_sample_paths_workers_shape_and_reproducibility(
    name: str,
    kwargs: dict[str, Any],
    workers: int,
) -> None:
    n = 12
    paths = 43  # not a multiple of the number of workers

    process = make(name, kwargs, seed=7)
    first = process.sample_paths(n, paths, workers=workers)
    second = make(name, kwargs, seed=7).sample_paths(n, paths, workers=workers)
    single = make(name, kwargs, seed=7).sample_paths(n, paths)

    assert first.shape == single.shape
    assert first.dtype == single.dtype
    np.testing.assert_array_equal(first, second)
    if isinstance(process, BaseTimeProcess):
        assert process.n == n
        np.testing.assert_allclose(process.times, np.linspace(0, process.t, n + 1))


@pytest.mark.usefixtures("small_batches_use_threads")
@pytest.mark.parametrize("name, kwargs", CASES, ids=IDS)
def test_sample_paths_workers_distribution(name: str, kwargs: dict[str, Any]) -> None:
    # Realizations generated on several threads have the distribution of
    # those generated one by one.
    n = 12
    paths = 3000

    together = make(name, kwargs, seed=11).sample_paths(n, paths, workers=4)
    separate = one_by_one(make(name, kwargs, seed=13), n, paths)

    for label, statistic in _statistics(flat(together).shape[1]).items():
        result = ks_2samp(statistic(flat(together)), statistic(flat(separate)))
        assert result.pvalue > 1e-4, f"{label}: p-value {result.pvalue}"


@pytest.mark.usefixtures("small_batches_use_threads")
def test_sample_paths_workers_use_independent_generators() -> None:
    realizations = samplepaths.BrownianMotion(rng=np.random.default_rng(1)).sample_paths(
        16, 40, workers=4
    )

    # No realization is repeated across the threads.
    assert len(np.unique(realizations, axis=0)) == 40


@pytest.mark.usefixtures("small_batches_use_threads")
@pytest.mark.parametrize(
    "name",
    ["ChineseRestaurantProcess", "DirichletProcess", "MoranProcess", "MixedPoissonProcess"],
)
def test_sample_paths_workers_ignored_by_sequential_processes(name: str) -> None:
    kwargs = dict(next(case for case in CASES if case[0] == name)[1])

    threaded = make(name, kwargs, seed=5).sample_paths(10, 20, workers=4)
    single = make(name, kwargs, seed=5).sample_paths(10, 20)

    np.testing.assert_array_equal(threaded, single)


def test_sample_paths_small_batches_use_one_worker() -> None:
    # Too few values to be worth a thread: the result is that of one worker.
    threaded = samplepaths.BrownianMotion(rng=np.random.default_rng(2)).sample_paths(
        8, 10, workers=4
    )
    single = samplepaths.BrownianMotion(rng=np.random.default_rng(2)).sample_paths(8, 10)

    np.testing.assert_array_equal(threaded, single)


@pytest.mark.parametrize(
    "workers, error_type",
    [
        (0, ValueError),
        (-1, ValueError),
        (1.5, TypeError),
    ],
)
def test_sample_paths_invalid_workers(workers: Any, error_type: type) -> None:
    with pytest.raises(error_type):
        samplepaths.BrownianMotion().sample_paths(8, 10, workers=workers)

    with pytest.raises(error_type):
        samplepaths.VasicekProcess().sample_paths_with_initial(8, 10, initial=1.0, workers=workers)


@pytest.mark.usefixtures("small_batches_use_threads")
@pytest.mark.parametrize("workers", [1, 3])
def test_sample_paths_with_initial_workers(workers: int) -> None:
    gbm = samplepaths.GeometricBrownianMotion(drift=0.3, volatility=1e-12, t=2.0)
    realizations = gbm.sample_paths_with_initial(10, 50, initial=3.0, workers=workers)

    np.testing.assert_allclose(
        realizations,
        np.tile(3.0 * np.exp(0.3 * np.linspace(0, 2.0, 11)), (50, 1)),
        rtol=1e-6,
    )

    vasicek = samplepaths.VasicekProcess(speed=1.0, mean=0.5, vol=1e-12)
    realizations = vasicek.sample_paths_with_initial(200, 50, initial=2.0, workers=workers)

    np.testing.assert_allclose(realizations[:, 0], 2.0)
    np.testing.assert_allclose(realizations[:, -1], 0.5 + 1.5 * np.exp(-1.0), rtol=1e-2)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"speed": 1.5, "mean": 0.5, "vol": 0.3, "t": 2.0},
        # a decay small enough for the recurrence to be solved in blocks
        {"speed": 20.0, "mean": 1.0, "vol": 0.5, "t": 10.0},
        # a negative speed, where the process grows
        {"speed": -0.8, "mean": 1.0, "vol": 0.2, "t": 3.0},
        {"speed": 0.0, "mean": 1.0, "vol": 0.2},
        {"speed": 2.0, "mean": lambda t: np.sin(t), "vol": lambda t: 0.1 + t, "t": 2.0},
    ],
)
def test_linear_diffusion_sample_paths_matches_the_step_loop(
    kwargs: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Many realizations of a linear diffusion are computed in closed form,
    # few of them step by step. Both use the same noise.
    import samplepaths.diffusion.diffusion_process as module

    n = 1000
    paths = 2 * module._VECTORIZE_FROM_PATHS

    closed_form = samplepaths.ExtendedVasicekProcess(rng=np.random.default_rng(3), **kwargs)
    together = closed_form.sample_paths_with_initial(n, paths, initial=1.5)

    monkeypatch.setattr(module, "_VECTORIZE_FROM_PATHS", 10 * paths)
    stepping = samplepaths.ExtendedVasicekProcess(rng=np.random.default_rng(3), **kwargs)
    in_turn = stepping.sample_paths_with_initial(n, paths, initial=1.5)

    np.testing.assert_allclose(together, in_turn, rtol=1e-9, atol=1e-11)
