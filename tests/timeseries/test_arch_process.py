from typing import Any

import numpy as np
import pytest

from samplepaths.timeseries import ARCHProcess as ProcessType

KWARGS: dict[str, Any] = {"omega": 0.5, "alpha": 0.5}


@pytest.mark.parametrize(
    "kwargs, error_type",
    [
        ({"omega": 0}, ValueError),
        ({"omega": -1}, ValueError),
        ({"omega": "a"}, TypeError),
        ({"alpha": "a"}, TypeError),
        ({"alpha": -0.1}, ValueError),
        ({"alpha": 1.0}, ValueError),
        ({"alpha": [0.6, 0.5]}, ValueError),
    ],
)
def test_invalid_parameters(kwargs: dict[str, Any], error_type: type) -> None:
    with pytest.raises(error_type):
        ProcessType(**{**KWARGS, **kwargs})


def test_an_invalid_change_is_undone() -> None:
    instance = ProcessType(**KWARGS)

    value: Any = 1.5
    with pytest.raises(ValueError):
        instance.alpha = value

    # The parameters are only valid together, so the previous value is kept.
    assert instance.alpha == (0.5,)
    assert instance.sample(5).shape == (5,)


def test_a_single_number_is_one_coefficient() -> None:
    assert ProcessType(**KWARGS).alpha == (0.5,)


def test_str_repr() -> None:
    instance = ProcessType(**KWARGS)

    assert str(instance) == ("ARCH(1) process with omega=0.5 and alpha=(0.5,).")
    assert repr(instance) == ("ARCHProcess(omega=0.5, alpha=(0.5,))")


@pytest.mark.parametrize("n", [1, 2, 100])
def test_sample_shape(n: int) -> None:
    instance = ProcessType(**KWARGS)
    samples = instance.sample(n)

    assert samples.shape == (n,)
    assert samples.dtype == np.float64
    assert instance.volatility.shape == (n,)
    assert np.all(instance.volatility > 0)


@pytest.mark.parametrize(
    "n, error_type",
    [
        (0, ValueError),
        (-1, ValueError),
        (1.5, TypeError),
    ],
)
def test_sample_invalid_n(n: Any, error_type: type) -> None:
    with pytest.raises(error_type):
        ProcessType(**KWARGS).sample(n)


def test_sample_values() -> None:
    instance = ProcessType(**KWARGS, rng=np.random.default_rng(1234567890))

    np.testing.assert_allclose(
        instance.sample(6),
        [0.02931140, -0.51825330, 0.64450554, 0.50084424, 0.21633897, -0.37511060],
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        instance.volatility,
        [1.60605964, 0.70741047, 0.79642529, 0.84124532, 0.79083657, 0.72346477],
        rtol=1e-6,
    )


def test_volatility_before_sampling() -> None:
    with pytest.raises(RuntimeError):
        _ = ProcessType(**KWARGS).volatility


def test_values_are_the_volatility_times_standard_normal_variates() -> None:
    instance = ProcessType(**KWARGS, rng=np.random.default_rng(3))
    paths = instance.sample_paths(200, 20_000)
    innovations = paths / instance.volatility

    assert instance.volatility.shape == paths.shape
    np.testing.assert_allclose(innovations.mean(), 0.0, atol=0.01)
    np.testing.assert_allclose(innovations.var(), 1.0, rtol=0.01)


def test_few_and_many_realizations_use_the_same_method(monkeypatch: pytest.MonkeyPatch) -> None:
    # Few realizations are advanced with Python floats, many with arrays.
    import samplepaths.timeseries.conditional_variance_process as module

    paths = 2 * module._VECTORIZE_FROM_PATHS
    together = ProcessType(**KWARGS, rng=np.random.default_rng(3))
    values = together.sample_paths(50, paths)

    monkeypatch.setattr(module, "_VECTORIZE_FROM_PATHS", 10 * paths)
    in_turn = ProcessType(**KWARGS, rng=np.random.default_rng(3))

    np.testing.assert_allclose(in_turn.sample_paths(50, paths), values, rtol=1e-10)
    np.testing.assert_allclose(in_turn.volatility, together.volatility, rtol=1e-10)


def test_starts_and_stays_at_the_long_run_variance() -> None:
    # The long-run variance is omega / (1 - sum(alpha)).
    paths = ProcessType(**KWARGS, rng=np.random.default_rng(1)).sample_paths(400, 40_000)

    for index in (0, 10, -1):
        np.testing.assert_allclose(paths[:, index].var(), 1.0, rtol=0.05)
    np.testing.assert_allclose(paths.mean(), 0.0, atol=0.01)


def test_large_values_cluster() -> None:
    # The squares of consecutive values are positively correlated.
    paths = ProcessType(**KWARGS, rng=np.random.default_rng(1)).sample_paths(200, 40_000)

    assert np.corrcoef(paths[:, -1] ** 2, paths[:, -2] ** 2)[0, 1] > 0.05


def test_effect_of_the_sign_of_a_value() -> None:
    # The variance depends on the square of a value, not on its sign.
    instance = ProcessType(**KWARGS, rng=np.random.default_rng(1))
    paths = instance.sample_paths(200, 40_000)
    previous = paths[:, -2]
    variance = instance.volatility[:, -1] ** 2

    after_negative = variance[previous < 0].mean()
    after_positive = variance[previous > 0].mean()
    assert abs(after_negative - after_positive) < 0.05 * after_positive


def test_variance_follows_its_recurrence() -> None:
    instance = ProcessType(omega=0.5, alpha=[0.3, 0.2], rng=np.random.default_rng(3))
    values = instance.sample(50)
    variance = instance.volatility**2

    expected = 0.5 + 0.3 * values[1:-1] ** 2 + 0.2 * values[:-2] ** 2
    np.testing.assert_allclose(variance[2:], expected)
