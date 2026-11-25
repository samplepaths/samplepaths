from typing import Any

import numpy as np
import pytest

from samplepaths.continuous import FractionalGeometricBrownianMotion as ProcessType

KWARGS: dict[str, Any] = {"hurst": 0.7, "drift": 0.05, "volatility": 0.2}


@pytest.mark.parametrize(
    "kwargs, error_type",
    [
        ({"hurst": 0.0}, ValueError),
        ({"hurst": 1.0}, ValueError),
        ({"hurst": "a"}, TypeError),
        ({"t": 0}, ValueError),
        ({"drift": "a"}, TypeError),
        ({"volatility": -1}, ValueError),
        ({"volatility": "a"}, TypeError),
    ],
)
def test_invalid_parameters(kwargs: dict[str, Any], error_type: type) -> None:
    with pytest.raises(error_type):
        ProcessType(**{**KWARGS, **kwargs})


def test_str_repr() -> None:
    instance = ProcessType(**KWARGS)

    assert str(instance) == (
        "Fractional geometric Brownian motion with Hurst"
        " 0.7, drift 0.05 and volatility 0.2 on [0, 1.0]."
    )
    assert repr(instance) == (
        "FractionalGeometricBrownianMotion(hurst=0.7, drift=0.05, volatility=0.2, t=1.0)"
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
        [1.00000000, 0.91292230, 0.81943980, 0.74054273, 0.68535298, 0.67358433],
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


def test_mean_grows_at_the_drift_for_every_hurst_parameter() -> None:
    for hurst in (0.3, 0.5, 0.8):
        instance = ProcessType(
            hurst=hurst, drift=0.5, volatility=0.4, t=2.0, rng=np.random.default_rng(1)
        )
        paths = instance.sample_paths_with_initial(32, 200_000, initial=3.0, workers=4)

        np.testing.assert_allclose(paths[:, -1].mean(), 3.0 * np.exp(1.0), rtol=0.01)
        np.testing.assert_allclose(paths[:, 16].mean(), 3.0 * np.exp(0.5), rtol=0.01)


def test_logarithm_is_a_fractional_brownian_motion_with_a_trend() -> None:
    instance = ProcessType(
        hurst=0.8, drift=0.1, volatility=0.4, t=2.0, rng=np.random.default_rng(1)
    )
    logs = np.log(instance.sample_paths(32, 100_000))

    times = instance.times
    np.testing.assert_allclose(logs.mean(axis=0), 0.1 * times - 0.08 * times**1.6, atol=0.01)
    np.testing.assert_allclose(logs.std(axis=0), 0.4 * times**0.8, rtol=0.02, atol=1e-12)


def test_with_hurst_one_half_is_a_geometric_brownian_motion() -> None:
    from samplepaths.continuous import GeometricBrownianMotion

    fractional = ProcessType(hurst=0.5, drift=0.1, volatility=0.3, rng=np.random.default_rng(1))
    reference = GeometricBrownianMotion(drift=0.1, volatility=0.3, rng=np.random.default_rng(2))
    ours = np.log(fractional.sample_paths(32, 100_000)[:, -1])
    theirs = np.log(reference.sample_paths(32, 100_000)[:, -1])

    np.testing.assert_allclose(ours.mean(), theirs.mean(), atol=0.005)
    np.testing.assert_allclose(ours.std(), theirs.std(), rtol=0.02)


def test_without_volatility_is_the_exponential_trend() -> None:
    instance = ProcessType(hurst=0.7, drift=0.5, volatility=0.0)

    np.testing.assert_allclose(instance.sample(4), np.exp(0.5 * instance.times))


def test_invalid_initial() -> None:
    with pytest.raises(ValueError):
        ProcessType(**KWARGS).sample_with_initial(4, initial=0.0)
    with pytest.raises(ValueError):
        ProcessType(**KWARGS).sample_paths_with_initial(4, 3, initial=-1.0)
