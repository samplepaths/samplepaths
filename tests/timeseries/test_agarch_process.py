from typing import Any

import numpy as np
import pytest

from samplepaths.timeseries import AGARCHProcess as ProcessType

KWARGS: dict[str, Any] = {"omega": 0.1, "alpha": 0.1, "shift": 0.5, "beta": 0.8}


@pytest.mark.parametrize(
    "kwargs, error_type",
    [
        ({"omega": 0}, ValueError),
        ({"alpha": "a"}, TypeError),
        ({"alpha": -0.1}, ValueError),
        ({"shift": "a"}, TypeError),
        ({"beta": -0.1}, ValueError),
        ({"beta": 0.9}, ValueError),
    ],
)
def test_invalid_parameters(kwargs: dict[str, Any], error_type: type) -> None:
    with pytest.raises(error_type):
        ProcessType(**{**KWARGS, **kwargs})


def test_an_invalid_change_is_undone() -> None:
    instance = ProcessType(**KWARGS)

    value: Any = 0.95
    with pytest.raises(ValueError):
        instance.beta = value

    # The parameters are only valid together, so the previous value is kept.
    assert instance.beta == (0.8,)
    assert instance.sample(5).shape == (5,)


def test_a_single_number_is_one_coefficient() -> None:
    assert ProcessType(**KWARGS).alpha == (0.1,)


def test_str_repr() -> None:
    instance = ProcessType(**KWARGS)

    assert str(instance) == (
        "AGARCH(1, 1) process with omega=0.1, alpha=(0.1,), shift=0.5 and beta=(0.8,)."
    )
    assert repr(instance) == ("AGARCHProcess(omega=0.1, alpha=(0.1,), shift=0.5, beta=(0.8,))")


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
        [0.02497627, -0.93270098, 1.02424894, 0.70671081, 0.30357509, -0.54108854],
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        instance.volatility,
        [1.36852497, 1.27312734, 1.26567997, 1.18703005, 1.10973204, 1.04358154],
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
    # The long-run variance is (omega + shift^2 sum(alpha)) / (1 - sum(alpha) - sum(beta)).
    paths = ProcessType(**KWARGS, rng=np.random.default_rng(1)).sample_paths(400, 40_000)

    for index in (0, 10, -1):
        np.testing.assert_allclose(paths[:, index].var(), 1.25, rtol=0.05)
    np.testing.assert_allclose(paths.mean(), 0.0, atol=0.01)


def test_large_values_cluster() -> None:
    # The squares of consecutive values are positively correlated.
    paths = ProcessType(**KWARGS, rng=np.random.default_rng(1)).sample_paths(200, 40_000)

    assert np.corrcoef(paths[:, -1] ** 2, paths[:, -2] ** 2)[0, 1] > 0.05


def test_effect_of_the_sign_of_a_value() -> None:
    # With a positive shift, a negative value raises the variance more.
    instance = ProcessType(**KWARGS, rng=np.random.default_rng(1))
    paths = instance.sample_paths(200, 40_000)
    previous = paths[:, -2]
    variance = instance.volatility[:, -1] ** 2

    after_negative = variance[previous < 0].mean()
    after_positive = variance[previous > 0].mean()
    assert after_negative > 1.05 * after_positive


def test_variance_follows_its_recurrence() -> None:
    instance = ProcessType(**KWARGS, rng=np.random.default_rng(3))
    values = instance.sample(50)
    variance = instance.volatility**2

    expected = 0.1 + 0.1 * (values[:-1] - 0.5) ** 2 + 0.8 * variance[:-1]
    np.testing.assert_allclose(variance[1:], expected)


def test_without_shift_is_a_garch_process() -> None:
    from samplepaths.timeseries import GARCHProcess

    shifted = ProcessType(omega=0.1, alpha=0.1, shift=0.0, beta=0.8, rng=np.random.default_rng(3))
    garch = GARCHProcess(omega=0.1, alpha=0.1, beta=0.8, rng=np.random.default_rng(3))

    np.testing.assert_allclose(shifted.sample(30), garch.sample(30))
