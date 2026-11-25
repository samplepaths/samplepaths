"""Tests of sample_paths, which generates several realizations at once."""

from typing import Any

import numpy as np
import pytest
from scipy.stats import ks_2samp

import samplepaths
from samplepaths.base import BaseTimeProcess

from .cases import CASES, IDS, T5, flat, make, one_by_one, statistics


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

    for label, statistic in statistics(flat(together).shape[1]).items():
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
