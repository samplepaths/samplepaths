import math
from typing import Any

import numpy as np
import pytest

from samplepaths.multivariate import HestonModel as ProcessType

KWARGS: dict[str, Any] = {
    "drift": 0.05,
    "speed": 2.0,
    "mean": 0.04,
    "vol": 0.3,
    "correlation": -0.7,
    "initial_variance": 0.09,
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
    ],
)
def test_invalid_parameters(kwargs: dict[str, Any], error_type: type) -> None:
    with pytest.raises(error_type):
        ProcessType(**{**KWARGS, **kwargs})


def test_str_repr() -> None:
    instance = ProcessType(**KWARGS)

    assert str(instance) == (
        "Heston model with drift=0.05, speed=2.0, mean=0.04, vol=0.3, correlation=-0.7 on [0, 1.0]"
    )
    assert repr(instance) == (
        "HestonModel(drift=0.05, speed=2.0, mean=0.04, vol=0"
        ".3, correlation=-0.7, initial_variance=0.09, t=1.0)"
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
        samples[0], [1.00000000, 1.08570830, 1.08360677, 1.08688889, 1.08715313], rtol=1e-6
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
        instance.sample_with_initial(10, initial=0.0)


def test_initial_variance_defaults_to_the_long_run_variance() -> None:
    instance = ProcessType(mean=0.05)

    assert instance.initial_variance == 0.05
    assert instance.sample(3)[1, 0] == 0.05
    instance.initial_variance = 0.01
    assert instance.sample(3)[1, 0] == 0.01
    instance.initial_variance = None
    assert instance.initial_variance == 0.05


@pytest.mark.parametrize("n", [4, 64])
def test_variance_has_its_exact_moments_whatever_the_step(n: int) -> None:
    instance = ProcessType(**KWARGS, rng=np.random.default_rng(1))
    variance = instance.sample_paths(n, 200_000, workers=4)[:, 1, -1]

    speed, mean, vol, start = 2.0, 0.04, 0.3, 0.09
    decay = math.exp(-speed)
    expected_mean = mean + (start - mean) * decay
    expected_variance = (
        start * vol**2 / speed * (decay - decay**2) + mean * vol**2 / (2 * speed) * (1 - decay) ** 2
    )
    np.testing.assert_allclose(variance.mean(), expected_mean, rtol=0.01)
    np.testing.assert_allclose(variance.var(), expected_variance, rtol=0.03)
    assert variance.min() >= 0


def test_variance_stays_nonnegative_when_it_can_reach_zero() -> None:
    # With 2 speed mean < vol^2 the variance reaches zero.
    instance = ProcessType(speed=0.5, mean=0.02, vol=1.0, rng=np.random.default_rng(1))
    paths = instance.sample_paths(64, 200_000, workers=4)

    assert paths[:, 1].min() >= 0
    assert np.all(np.isfinite(paths))
    assert np.all(paths[:, 0] > 0)
    np.testing.assert_allclose(paths[:, 1, -1].mean(), 0.02, rtol=0.05)
    assert np.mean(paths[:, 1, -1] < 1e-6) > 0.5


def test_mean_price_grows_at_the_drift() -> None:
    instance = ProcessType(**KWARGS, t=2.0, rng=np.random.default_rng(1))
    prices = instance.sample_paths_with_initial(32, 400_000, initial=10.0, workers=4)[:, 0]

    np.testing.assert_allclose(prices[:, -1].mean(), 10.0 * math.exp(0.1), rtol=0.005)
    np.testing.assert_allclose(prices[:, 16].mean(), 10.0 * math.exp(0.05), rtol=0.005)


def test_mean_logarithm_of_the_price() -> None:
    # E[ln S] = drift t - half the expected integral of the variance.
    instance = ProcessType(**KWARGS, rng=np.random.default_rng(1))
    logs = np.log(instance.sample_paths(64, 200_000, workers=4)[:, 0, -1])

    integral = 0.04 + (0.09 - 0.04) * (1 - math.exp(-2.0)) / 2.0
    np.testing.assert_allclose(logs.mean(), 0.05 - integral / 2, atol=0.003)


@pytest.mark.parametrize("correlation", [-0.7, 0.0, 0.7, 1.0, -1.0])
def test_price_and_variance_move_together_as_the_correlation_says(correlation: float) -> None:
    instance = ProcessType(**{**KWARGS, "correlation": correlation}, rng=np.random.default_rng(1))
    paths = instance.sample_paths(64, 50_000)
    returns = np.log(paths[:, 0, -1])
    measured = np.corrcoef(returns, paths[:, 1, -1])[0, 1]

    assert np.all(np.isfinite(paths))
    if correlation == 0:
        assert abs(measured) < 0.03
    else:
        assert math.copysign(1, measured) == math.copysign(1, correlation)
        assert abs(measured) > 0.4


def test_few_and_many_realizations_have_the_same_distribution() -> None:
    few = np.concatenate(
        [
            ProcessType(**KWARGS, rng=np.random.default_rng(seed)).sample_paths(16, 4)
            for seed in range(2000)
        ]
    )
    many = ProcessType(**KWARGS, rng=np.random.default_rng(1)).sample_paths(16, 8000)

    np.testing.assert_allclose(few[:, 1, -1].mean(), many[:, 1, -1].mean(), rtol=0.03)
    np.testing.assert_allclose(few[:, 1, -1].std(), many[:, 1, -1].std(), rtol=0.05)
    np.testing.assert_allclose(np.log(few[:, 0, -1]).std(), np.log(many[:, 0, -1]).std(), rtol=0.05)
