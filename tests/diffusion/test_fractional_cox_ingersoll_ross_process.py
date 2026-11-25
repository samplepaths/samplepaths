from typing import Any

import numpy as np
import pytest

from samplepaths.diffusion import FractionalCoxIngersollRossProcess as ProcessType

KWARGS: dict[str, Any] = {"hurst": 0.7, "speed": 1.0, "mean": 1.0, "vol": 0.3}


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
        ({"mean": -1}, ValueError),
        ({"mean": "a"}, TypeError),
    ],
)
def test_invalid_parameters(kwargs: dict[str, Any], error_type: type) -> None:
    with pytest.raises(error_type):
        ProcessType(**{**KWARGS, **kwargs})


def test_str_repr() -> None:
    instance = ProcessType(**KWARGS)

    assert str(instance) == (
        "Fractional Cox-Ingersoll-Ross process with Hurs"
        "t 0.7, speed=1.0, mean=1.0, vol=0.3 on [0, 1.0]"
    )
    assert repr(instance) == (
        "FractionalCoxIngersollRossProcess(hurst=0.7, speed=1.0, mean=1.0, vol=0.3, t=1.0)"
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
        [1.00000000, 0.85149507, 0.72259229, 0.64163955, 0.61406730, 0.66544764],
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


def test_values_stay_in_the_domain() -> None:
    instance = ProcessType(**{**KWARGS, "vol": 3.0}, rng=np.random.default_rng(1))
    paths = instance.sample_paths(64, 2000)

    assert paths.min() >= 0.0
    assert paths.max() > 1.0
    assert np.all(np.isfinite(paths))
    single = instance.sample(500)
    assert single.min() >= 0.0


def test_mean_reverts() -> None:
    instance = ProcessType(**{**KWARGS, "speed": 8.0}, rng=np.random.default_rng(1))
    paths = instance.sample_paths_with_initial(128, 20_000, initial=3.0)

    np.testing.assert_allclose(paths[:, -1].mean(), 1.0, atol=0.02)


def test_few_and_many_realizations_use_the_same_method() -> None:
    few = ProcessType(**KWARGS, rng=np.random.default_rng(1)).sample_paths(32, 31)
    many = ProcessType(**KWARGS, rng=np.random.default_rng(1)).sample_paths(32, 3000)

    assert few.shape == (31, 33)
    np.testing.assert_allclose(few[:, -1].mean(), many[:, -1].mean(), atol=0.2)


def test_with_hurst_one_half_is_the_cox_ingersoll_ross_process() -> None:
    from samplepaths.diffusion import CoxIngersollRossProcess

    fractional = ProcessType(**{**KWARGS, "hurst": 0.5}, rng=np.random.default_rng(1))
    paths = fractional.sample_paths(64, 40_000)[:, -1]
    reference = CoxIngersollRossProcess(
        speed=1.0, mean=1.0, vol=0.3, rng=np.random.default_rng(2)
    ).sample_paths(64, 40_000)[:, -1]

    np.testing.assert_allclose(paths.mean(), reference.mean(), atol=0.01)
    np.testing.assert_allclose(paths.std(), reference.std(), rtol=0.03)
