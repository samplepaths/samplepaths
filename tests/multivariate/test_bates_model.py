import math
from typing import Any

import numpy as np
import pytest

from samplepaths.multivariate import BatesModel as ProcessType

KWARGS: dict[str, Any] = {
    "drift": 0.05,
    "speed": 2.0,
    "mean": 0.04,
    "vol": 0.3,
    "correlation": -0.7,
    "initial_variance": 0.09,
    "rate": 2.0,
    "jump_mean": -0.1,
    "jump_volatility": 0.15,
}


@pytest.mark.parametrize(
    "kwargs, error_type",
    [
        ({"drift": "a"}, TypeError),
        ({"speed": 0}, ValueError),
        ({"speed": "a"}, TypeError),
        ({"mean": 0}, ValueError),
        ({"vol": 0}, ValueError),
        ({"vol": "a"}, TypeError),
        ({"correlation": 1.5}, ValueError),
        ({"correlation": "a"}, TypeError),
        ({"initial_variance": -0.1}, ValueError),
        ({"initial_variance": "a"}, TypeError),
        ({"t": 0}, ValueError),
        ({"rate": -1}, ValueError),
        ({"rate": "a"}, TypeError),
        ({"jump_mean": "a"}, TypeError),
        ({"jump_volatility": -1}, ValueError),
        ({"jump_volatility": "a"}, TypeError),
    ],
)
def test_invalid_parameters(kwargs: dict[str, Any], error_type: type) -> None:
    with pytest.raises(error_type):
        ProcessType(**{**KWARGS, **kwargs})


def test_str_repr() -> None:
    instance = ProcessType(**KWARGS)

    assert str(instance) == (
        "Bates model with drift=0.05, speed=2.0, mean=0.04, vol=0.3, correlat"
        "ion=-0.7, rate=2.0, jump_mean=-0.1, jump_volatility=0.15 on [0, 1.0]"
    )
    assert repr(instance) == (
        "BatesModel(drift=0.05, speed=2.0, mean=0.04, vol=0.3, correlation=-0.7, ra"
        "te=2.0, jump_mean=-0.1, jump_volatility=0.15, initial_variance=0.09, t=1.0)"
    )


@pytest.mark.parametrize("n", [1, 2, 100])
def test_sample_shape(n: int) -> None:
    instance = ProcessType(**KWARGS)
    samples = instance.sample(n)

    assert instance.components == ("price", "variance")
    assert samples.shape == (2, n + 1)
    assert instance.times.shape == (n + 1,)
    assert instance.sample_paths(n, 5).shape == (5, 2, n + 1)


@pytest.mark.parametrize("n, error_type", [(0, ValueError), (-1, ValueError), (1.5, TypeError)])
def test_sample_invalid_n(n: Any, error_type: type) -> None:
    with pytest.raises(error_type):
        ProcessType(**KWARGS).sample(n)


def test_sample_values() -> None:
    instance = ProcessType(**KWARGS, rng=np.random.default_rng(1234567890))
    samples = instance.sample(4)

    np.testing.assert_allclose(
        samples[0], [1.00000000, 1.23884092, 1.29007652, 1.35011353, 1.27077202], rtol=1e-6
    )
    np.testing.assert_allclose(
        samples[1], [0.09000000, 0.03441432, 0.04389870, 0.02619696, 0.02254614], rtol=1e-6
    )


def test_sample_at_shape() -> None:
    instance = ProcessType(**KWARGS)

    with_zero = instance.sample_at(np.array([0.0, 0.5, 2.0]))
    without_zero = instance.sample_at([0.5, 2.0])
    assert with_zero.shape == (2, 3)
    assert without_zero.shape == (2, 2)


@pytest.mark.parametrize("times", [[], [1.0, 0.5], [-1.0, 1.0], [0.5, 0.5], [[0.0, 1.0]]])
def test_sample_at_invalid_times(times: Any) -> None:
    with pytest.raises(ValueError):
        ProcessType(**KWARGS).sample_at(times)


def test_sample_at_is_reproducible() -> None:
    first = ProcessType(**KWARGS, rng=np.random.default_rng(5)).sample_at([0.1, 0.4, 3.0])
    again = ProcessType(**KWARGS, rng=np.random.default_rng(5)).sample_at([0.1, 0.4, 3.0])

    np.testing.assert_array_equal(first, again)


def test_sample_at_values() -> None:
    from scipy.stats import ks_2samp

    # An uneven grid gives the distribution of an even one.
    times = np.linspace(0.0, 1.0, 17) ** 2
    instance = ProcessType(**KWARGS, rng=np.random.default_rng(1))
    uneven = np.array([instance.sample_at_with_initial(times, 10.0) for _ in range(20_000)])
    even = instance.sample_paths_with_initial(64, 20_000, initial=10.0)

    assert np.all(uneven[:, 0, 0] == 10.0)
    assert np.all(uneven[:, 1, 0] == instance.initial_variance)
    assert ks_2samp(uneven[:, 0, -1], even[:, 0, -1]).pvalue > 1e-3
    assert ks_2samp(uneven[:, 1, -1], even[:, 1, -1]).pvalue > 1e-3
    np.testing.assert_allclose(uneven[:, 0, -1].mean(), 10.0 * math.exp(0.05), rtol=0.01)


def test_sample_at_invalid_initial() -> None:
    with pytest.raises(ValueError):
        ProcessType(**KWARGS).sample_at_with_initial([0.0, 1.0], initial=0.0)


def test_threads_give_reproducible_realizations() -> None:
    first = ProcessType(**KWARGS, rng=np.random.default_rng(5)).sample_paths(16, 20_000, workers=4)
    again = ProcessType(**KWARGS, rng=np.random.default_rng(5)).sample_paths(16, 20_000, workers=4)

    assert first.shape == (20_000, 2, 17)
    np.testing.assert_array_equal(first, again)


def test_sample_with_initial() -> None:
    instance = ProcessType(**KWARGS)

    assert instance.sample_with_initial(10, initial=50.0)[0, 0] == 50.0
    paths = instance.sample_paths_with_initial(10, 40, initial=50.0, workers=2)
    assert paths.shape == (40, 2, 11)
    assert np.all(paths[:, 0, 0] == 50.0)
    with pytest.raises(TypeError):
        instance.sample_with_initial(10, initial="a")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        instance.sample_with_initial(10, initial=-1.0)


def test_mean_relative_jump() -> None:
    instance = ProcessType(**KWARGS)

    np.testing.assert_allclose(instance.mean_relative_jump, math.exp(-0.1 + 0.15**2 / 2) - 1)


def test_mean_price_grows_at_the_drift_despite_the_jumps() -> None:
    instance = ProcessType(**{**KWARGS, "rate": 3.0, "jump_mean": -0.2, "jump_volatility": 0.3})
    instance.rng = np.random.default_rng(1)
    prices = instance.sample_paths(16, 800_000, workers=4)[:, 0, -1]

    np.testing.assert_allclose(prices.mean(), math.exp(0.05), rtol=0.005)


def test_mean_and_variance_of_the_logarithm_of_the_price() -> None:
    instance = ProcessType(**KWARGS, rng=np.random.default_rng(1))
    logs = np.log(instance.sample_paths(64, 200_000, workers=4)[:, 0, -1])

    integral = 0.04 + (0.09 - 0.04) * (1 - math.exp(-2.0)) / 2.0
    compensator = 2.0 * (math.exp(-0.1 + 0.15**2 / 2) - 1)
    np.testing.assert_allclose(
        logs.mean(), 0.05 - compensator - integral / 2 + 2.0 * -0.1, atol=0.004
    )
    np.testing.assert_allclose(logs.var(), integral + 2.0 * (0.1**2 + 0.15**2), rtol=0.05)


def test_without_jumps_is_a_heston_model() -> None:
    from samplepaths.multivariate import HestonModel

    kwargs = {key: KWARGS[key] for key in ("drift", "speed", "mean", "vol", "correlation")}
    bates = ProcessType(**kwargs, rate=0.0, rng=np.random.default_rng(1)).sample_paths(32, 50_000)
    heston = HestonModel(**kwargs, rng=np.random.default_rng(2)).sample_paths(32, 50_000)

    np.testing.assert_allclose(
        np.log(bates[:, 0, -1]).mean(), np.log(heston[:, 0, -1]).mean(), atol=0.005
    )
    np.testing.assert_allclose(
        np.log(bates[:, 0, -1]).std(), np.log(heston[:, 0, -1]).std(), rtol=0.03
    )


def test_the_variance_does_not_jump() -> None:
    instance = ProcessType(**{**KWARGS, "rate": 50.0}, rng=np.random.default_rng(1))
    variance = instance.sample_paths(64, 100_000)[:, 1, -1]

    np.testing.assert_allclose(variance.mean(), 0.04 + 0.05 * math.exp(-2.0), rtol=0.02)
