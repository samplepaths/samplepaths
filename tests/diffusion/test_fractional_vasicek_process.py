from typing import Any

import numpy as np
import pytest

from samplepaths.diffusion import FractionalVasicekProcess as ProcessType

KWARGS: dict[str, Any] = {"hurst": 0.7, "speed": 1.5, "mean": 0.5, "vol": 0.3}


@pytest.mark.parametrize(
    "kwargs, error_type",
    [
        ({"hurst": 0.0}, ValueError),
        ({"hurst": 1.0}, ValueError),
        ({"hurst": "a"}, TypeError),
        ({"t": 0}, ValueError),
        ({"speed": 0}, ValueError),
        ({"speed": "a"}, TypeError),
        ({"vol": -1}, ValueError),
        ({"vol": "a"}, TypeError),
        ({"mean": "a"}, TypeError),
    ],
)
def test_invalid_parameters(kwargs: dict[str, Any], error_type: type) -> None:
    with pytest.raises(error_type):
        ProcessType(**{**KWARGS, **kwargs})


def test_str_repr() -> None:
    instance = ProcessType(**KWARGS)

    assert str(instance) == (
        "Fractional Vasicek process with Hurst 0.7, speed=1.5, mean=0.5, vol=0.3 on [0, 1.0]"
    )
    assert repr(instance) == (
        "FractionalVasicekProcess(hurst=0.7, speed=1.5, mean=0.5, vol=0.3, t=1.0)"
    )


def test_hurst_can_be_changed() -> None:
    instance = ProcessType(**KWARGS)
    instance.hurst = 0.3

    assert instance.hurst == 0.3
    assert "0.3" in repr(instance)
    with pytest.raises(ValueError):
        instance.hurst = 1.5


@pytest.mark.parametrize("n", [1, 2, 100])
def test_sample_shape(n: int) -> None:
    instance = ProcessType(**KWARGS)
    samples = instance.sample(n)

    assert samples.shape == (n + 1,)
    assert samples[0] == 1.0
    assert instance.times.shape == (n + 1,)
    assert instance.times[-1] == instance.t


@pytest.mark.parametrize("n, error_type", [(0, ValueError), (-1, ValueError), (1.5, TypeError)])
def test_sample_invalid_n(n: Any, error_type: type) -> None:
    with pytest.raises(error_type):
        ProcessType(**KWARGS).sample(n)


def test_sample_values() -> None:
    instance = ProcessType(**KWARGS, rng=np.random.default_rng(1234567890))

    np.testing.assert_allclose(
        instance.sample(5),
        [1.00000000, 0.70149507, 0.46916770, 0.31791672, 0.24864488, 0.29111956],
        rtol=1e-6,
    )


def test_sample_with_initial() -> None:
    instance = ProcessType(**KWARGS)

    assert instance.sample_with_initial(10, initial=0.2)[0] == 0.2
    paths = instance.sample_paths_with_initial(10, 40, initial=0.2, workers=2)
    assert paths.shape == (40, 11)
    assert np.all(paths[:, 0] == 0.2)
    with pytest.raises(TypeError):
        instance.sample_with_initial(10, initial="a")  # type: ignore[arg-type]


def test_sample_at_is_not_supported() -> None:
    with pytest.raises(NotImplementedError):
        ProcessType(**KWARGS).sample_at(np.array([0.0, 0.5, 1.0]))


def test_follows_a_new_generator() -> None:
    instance = ProcessType(**KWARGS, rng=np.random.default_rng(1))
    first = instance.sample(8)
    instance.rng = np.random.default_rng(1)

    np.testing.assert_array_equal(instance.sample(8), first)


def test_follows_a_new_interval() -> None:
    instance = ProcessType(**KWARGS, rng=np.random.default_rng(1))
    instance.t = 4.0
    instance.sample(8)

    assert instance.times[-1] == 4.0


def test_threads_give_reproducible_realizations() -> None:
    first = ProcessType(**KWARGS, rng=np.random.default_rng(5)).sample_paths(16, 20_000, workers=4)
    again = ProcessType(**KWARGS, rng=np.random.default_rng(5)).sample_paths(16, 20_000, workers=4)

    np.testing.assert_array_equal(first, again)


def test_steps_follow_the_euler_method(monkeypatch: pytest.MonkeyPatch) -> None:
    # The realization is computed as a weighted cumulative sum; it must equal
    # the recurrence X[k + 1] = X[k] + speed (mean - X[k]) dt + vol dB[k].
    instance = ProcessType(**KWARGS, t=2.0)
    noise = np.random.default_rng(3).normal(size=(4, 50))
    values = instance._solve(noise, 2.0)

    speed, mean, vol, delta_t = 1.5, 0.5, 0.3, 2.0 / 50
    expected = np.empty((4, 51))
    expected[:, 0] = 2.0
    for k in range(50):
        previous = expected[:, k]
        expected[:, k + 1] = previous + speed * (mean - previous) * delta_t + vol * noise[:, k]
    np.testing.assert_allclose(values, expected, rtol=1e-10)


@pytest.mark.parametrize("t, n", [(400.0, 3000), (5000.0, 400), (30.0, 10)])
def test_long_intervals_and_large_steps_stay_exact(t: float, n: int) -> None:
    instance = ProcessType(**KWARGS, t=t)
    noise = np.random.default_rng(3).normal(size=n)
    values = instance._solve(noise, 2.0)

    speed, mean, vol, delta_t = 1.5, 0.5, 0.3, t / n
    x = 2.0
    for k in range(min(n, 12)):
        x = x + speed * (mean - x) * delta_t + vol * noise[k]
        np.testing.assert_allclose(values[k + 1], x, rtol=1e-8)


def test_mean_reverts() -> None:
    # The Euler method gives the mean exactly: a geometric approach to the
    # long-run mean.
    instance = ProcessType(**KWARGS, rng=np.random.default_rng(1))
    paths = instance.sample_paths_with_initial(64, 40_000, initial=2.0)

    expected = 0.5 + (2.0 - 0.5) * (1 - 1.5 / 64) ** np.arange(65)
    np.testing.assert_allclose(paths.mean(axis=0), expected, atol=0.01)


def test_a_larger_hurst_parameter_spreads_the_values_more_at_long_horizons() -> None:
    spreads = []
    for hurst in (0.3, 0.7):
        instance = ProcessType(**{**KWARGS, "hurst": hurst}, t=1.0, rng=np.random.default_rng(1))
        instance.speed = 0.01
        spreads.append(instance.sample_paths(64, 20_000)[:, 16].std())

    # Over a quarter of the interval the spread is close to vol * 0.25**hurst.
    np.testing.assert_allclose(spreads, [0.3 * 0.25**0.3, 0.3 * 0.25**0.7], rtol=0.05)


def test_with_hurst_one_half_is_the_vasicek_process() -> None:
    from samplepaths.diffusion import VasicekProcess

    fractional = ProcessType(**{**KWARGS, "hurst": 0.5}, rng=np.random.default_rng(1))
    paths = fractional.sample_paths(64, 40_000)[:, -1]
    reference = VasicekProcess(
        speed=1.5, mean=0.5, vol=0.3, rng=np.random.default_rng(2)
    ).sample_paths(64, 40_000)[:, -1]

    np.testing.assert_allclose(paths.mean(), reference.mean(), atol=0.01)
    np.testing.assert_allclose(paths.std(), reference.std(), rtol=0.03)
