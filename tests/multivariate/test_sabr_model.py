import math
from typing import Any

import numpy as np
import pytest

from samplepaths.multivariate import SABRModel as ProcessType

KWARGS: dict[str, Any] = {
    "volvol": 0.4,
    "beta": 0.5,
    "correlation": -0.3,
    "initial_volatility": 0.2,
}


@pytest.mark.parametrize(
    "kwargs, error_type",
    [
        ({"volvol": -1}, ValueError),
        ({"volvol": "a"}, TypeError),
        ({"beta": -0.1}, ValueError),
        ({"beta": 1.1}, ValueError),
        ({"beta": "a"}, TypeError),
        ({"correlation": -1.5}, ValueError),
        ({"correlation": "a"}, TypeError),
        ({"initial_volatility": 0}, ValueError),
        ({"initial_volatility": "a"}, TypeError),
        ({"t": 0}, ValueError),
    ],
)
def test_invalid_parameters(kwargs: dict[str, Any], error_type: type) -> None:
    with pytest.raises(error_type):
        ProcessType(**{**KWARGS, **kwargs})


def test_str_repr() -> None:
    instance = ProcessType(**KWARGS)

    assert str(instance) == ("SABR model with volvol=0.4, beta=0.5, correlation=-0.3 on [0, 1.0]")
    assert repr(instance) == (
        "SABRModel(volvol=0.4, beta=0.5, correlation=-0.3, initial_volatility=0.2, t=1.0)"
    )


@pytest.mark.parametrize("n", [1, 2, 100])
def test_sample_shape(n: int) -> None:
    instance = ProcessType(**KWARGS)
    samples = instance.sample(n)

    assert instance.components == ("forward", "volatility")
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
        samples[0], [1.00000000, 1.11797359, 1.13558345, 1.11680110, 1.13383905], rtol=1e-6
    )
    np.testing.assert_allclose(
        samples[1], [0.20000000, 0.13037983, 0.12826546, 0.10859016, 0.12513966], rtol=1e-6
    )


def test_sample_at_is_not_supported() -> None:
    with pytest.raises(NotImplementedError):
        ProcessType(**KWARGS).sample_at(np.array([0.0, 0.5, 1.0]))


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


def test_volatility_is_a_geometric_brownian_motion_without_drift() -> None:
    instance = ProcessType(**KWARGS, t=2.0, rng=np.random.default_rng(1))
    volatility = instance.sample_paths(8, 400_000, workers=4)[:, 1, -1]

    np.testing.assert_allclose(volatility.mean(), 0.2, rtol=0.005)
    np.testing.assert_allclose(np.log(volatility).mean(), math.log(0.2) - 0.4**2, atol=0.005)
    np.testing.assert_allclose(np.log(volatility).std(), 0.4 * math.sqrt(2.0), rtol=0.01)


@pytest.mark.parametrize("beta", [0.0, 0.5, 1.0])
def test_forward_has_no_drift(beta: float) -> None:
    instance = ProcessType(**{**KWARGS, "beta": beta}, rng=np.random.default_rng(1))
    forwards = instance.sample_paths_with_initial(64, 200_000, initial=1.0, workers=4)[:, 0]

    np.testing.assert_allclose(forwards[:, -1].mean(), 1.0, atol=0.003)
    np.testing.assert_allclose(forwards[:, 32].mean(), 1.0, atol=0.003)


def test_without_volatility_of_the_volatility_and_exponent_is_a_brownian_motion() -> None:
    instance = ProcessType(volvol=0.0, beta=0.0, correlation=0.0, initial_volatility=0.3)
    instance.rng = np.random.default_rng(1)
    paths = instance.sample_paths_with_initial(16, 100_000, initial=-2.0)

    assert np.all(paths[:, 1] == 0.3)
    np.testing.assert_allclose(paths[:, 0, -1].mean(), -2.0, atol=0.005)
    np.testing.assert_allclose(paths[:, 0, -1].std(), 0.3, rtol=0.01)


def test_with_exponent_one_the_forward_is_lognormal_without_volatility_of_the_volatility() -> None:
    instance = ProcessType(volvol=0.0, beta=1.0, correlation=0.0, initial_volatility=0.3)
    instance.rng = np.random.default_rng(1)
    logs = np.log(instance.sample_paths(16, 100_000)[:, 0, -1])

    np.testing.assert_allclose(logs.mean(), -0.045, atol=0.005)
    np.testing.assert_allclose(logs.std(), 0.3, rtol=0.01)


def test_a_forward_which_reaches_zero_stays_there() -> None:
    instance = ProcessType(volvol=0.5, beta=0.3, correlation=0.0, initial_volatility=1.5)
    instance.rng = np.random.default_rng(1)
    forwards = instance.sample_paths_with_initial(64, 2000, initial=0.2)[:, 0]

    assert forwards.min() == 0.0
    reached = np.argmax(forwards == 0, axis=1)
    for row, first in zip(
        forwards[forwards.min(axis=1) == 0][:50], reached[forwards.min(axis=1) == 0], strict=False
    ):
        assert np.all(row[first:] == 0)
    single = instance.sample_with_initial(500, initial=0.01)[0]
    assert single.min() >= 0


def test_correlation_of_the_forward_and_the_volatility() -> None:
    for correlation in (-0.8, 0.8):
        instance = ProcessType(
            **{**KWARGS, "correlation": correlation}, rng=np.random.default_rng(1)
        )
        paths = instance.sample_paths(16, 50_000)
        measured = np.corrcoef(paths[:, 0, -1], paths[:, 1, -1])[0, 1]

        assert math.copysign(1, measured) == math.copysign(1, correlation)
        assert abs(measured) > 0.4


def test_few_and_many_realizations_use_the_same_method(monkeypatch: pytest.MonkeyPatch) -> None:
    import samplepaths.multivariate.sabr_model as module

    paths = 2 * module._VECTORIZE_FROM_PATHS
    together = ProcessType(**KWARGS, rng=np.random.default_rng(3)).sample_paths(50, paths)

    monkeypatch.setattr(module, "_VECTORIZE_FROM_PATHS", 10 * paths)
    in_turn = ProcessType(**KWARGS, rng=np.random.default_rng(3)).sample_paths(50, paths)

    np.testing.assert_allclose(in_turn, together, rtol=1e-10)
