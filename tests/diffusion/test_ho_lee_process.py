from typing import Any

import numpy as np
import pytest

from samplepaths.diffusion import HoLeeProcess as ProcessType

KWARGS: dict[str, Any] = {"drift": 0.5, "vol": 0.3}


@pytest.mark.parametrize(
    "kwargs, error_type",
    [
        ({"drift": "a"}, TypeError),
        ({"drift": lambda a, b: a}, ValueError),
        ({"vol": -1}, ValueError),
        ({"vol": "a"}, TypeError),
        ({"t": 0}, ValueError),
    ],
)
def test_invalid_parameters(kwargs: dict[str, Any], error_type: type) -> None:
    with pytest.raises(error_type):
        ProcessType(**{**KWARGS, **kwargs})


def test_str_repr() -> None:
    instance = ProcessType(**KWARGS)

    assert str(instance).startswith("Ho-Lee process with drift=")
    assert str(instance).endswith("vol=0.3 on [0, 1.0]")
    assert repr(instance).startswith("HoLeeProcess(drift=")


def test_drift_is_a_function_of_the_time() -> None:
    assert ProcessType(**KWARGS).drift(3.0) == 0.5
    assert ProcessType(drift=lambda t: 2 * t).drift(3.0) == 6.0


@pytest.mark.parametrize("n", [1, 2, 100])
def test_sample_shape(n: int) -> None:
    instance = ProcessType(**KWARGS)
    samples = instance.sample(n)

    assert samples.shape == (n + 1,)
    assert samples[0] == 0.0
    assert instance.times.shape == (n + 1,)


@pytest.mark.parametrize("n, error_type", [(0, ValueError), (-1, ValueError), (1.5, TypeError)])
def test_sample_invalid_n(n: Any, error_type: type) -> None:
    with pytest.raises(error_type):
        ProcessType(**KWARGS).sample(n)


def test_sample_values() -> None:
    instance = ProcessType(**KWARGS, rng=np.random.default_rng(1234567890))

    np.testing.assert_allclose(
        instance.sample(5),
        [0.00000000, -0.17360444, -0.07115588, -0.06944532, 0.13912669, 0.31900268],
        rtol=1e-6,
    )


def test_sample_with_initial() -> None:
    instance = ProcessType(**KWARGS)

    assert instance.sample_with_initial(10, initial=0.03)[0] == 0.03
    paths = instance.sample_paths_with_initial(10, 40, initial=0.03, workers=2)
    assert paths.shape == (40, 11)
    assert np.all(paths[:, 0] == 0.03)
    with pytest.raises(TypeError):
        instance.sample_with_initial(10, initial="a")  # type: ignore[arg-type]


def test_sample_at_is_not_supported() -> None:
    with pytest.raises(NotImplementedError):
        ProcessType(**KWARGS).sample_at(np.array([0.0, 0.5, 1.0]))


def test_is_a_brownian_motion_with_drift_for_a_constant_drift() -> None:
    instance = ProcessType(**KWARGS, t=2.0, rng=np.random.default_rng(1))
    paths = instance.sample_paths_with_initial(16, 200_000, initial=0.1, workers=4)

    np.testing.assert_allclose(paths.mean(axis=0), 0.1 + 0.5 * instance.times, atol=0.005)
    np.testing.assert_allclose(
        paths.std(axis=0), 0.3 * np.sqrt(instance.times), rtol=0.01, atol=1e-12
    )


def test_drift_given_as_a_function_is_added_step_by_step() -> None:
    instance = ProcessType(drift=lambda t: 2 * t, vol=0.0, t=2.0)

    # Steps of 0.5 with the drift at the end of each step: 1, 2, 3 and 4.
    np.testing.assert_allclose(
        instance.sample_with_initial(4, initial=1.0), [1.0, 1.5, 2.5, 4.0, 6.0]
    )
    np.testing.assert_allclose(instance.sample_paths(4, 3), [[0.0, 0.5, 1.5, 3.0, 5.0]] * 3)
    fine = instance.sample(4000)
    np.testing.assert_allclose(fine[-1], 4.0, rtol=1e-3)
