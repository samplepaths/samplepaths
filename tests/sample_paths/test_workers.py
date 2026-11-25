"""Tests of sample_paths, on several threads."""

from typing import Any

import numpy as np
import pytest
from scipy.stats import ks_2samp

import samplepaths
from samplepaths.base import BaseTimeProcess

from .cases import CASES, IDS, flat, make, one_by_one, statistics


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

    for label, statistic in statistics(flat(together).shape[1]).items():
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
