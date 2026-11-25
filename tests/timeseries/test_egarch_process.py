from typing import Any

import numpy as np
import pytest

from samplepaths.timeseries import EGARCHProcess as ProcessType

KWARGS: dict[str, Any] = {"omega": -0.1, "alpha": 0.2, "gamma": -0.1, "beta": 0.9}


@pytest.mark.parametrize(
    "kwargs, error_type",
    [
        ({"omega": "a"}, TypeError),
        ({"alpha": "a"}, TypeError),
        ({"gamma": [0.1, 0.1]}, ValueError),
        ({"beta": 1.0}, ValueError),
        ({"beta": -1.2}, ValueError),
        ({"beta": [0.6, 0.5]}, ValueError),
    ],
)
def test_invalid_parameters(kwargs: dict[str, Any], error_type: type) -> None:
    with pytest.raises(error_type):
        ProcessType(**{**KWARGS, **kwargs})


def test_an_invalid_change_is_undone() -> None:
    instance = ProcessType(**KWARGS)

    value: Any = 1.5
    with pytest.raises(ValueError):
        instance.beta = value

    # The parameters are only valid together, so the previous value is kept.
    assert instance.beta == (0.9,)
    assert instance.sample(5).shape == (5,)


def test_a_single_number_is_one_coefficient() -> None:
    assert ProcessType(**KWARGS).alpha == (0.2,)


def test_str_repr() -> None:
    instance = ProcessType(**KWARGS)

    assert str(instance) == (
        "EGARCH(1, 1) process with omega=-0.1, alpha=(0.2,), gamma=(-0.1,) and beta=(0.9,)."
    )
    assert repr(instance) == ("EGARCHProcess(omega=-0.1, alpha=(0.2,), gamma=(-0.1,), beta=(0.9,))")


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
        [0.01387798, -0.50332291, 0.56587268, 0.39460230, 0.17094300, -0.30237002],
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        instance.volatility,
        [0.76041618, 0.68703065, 0.69925747, 0.66279556, 0.62488962, 0.58317216],
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


def test_logarithm_of_the_variance_starts_and_stays_at_its_long_run_level() -> None:
    # The long-run level of the logarithm of the variance is
    # omega / (1 - sum(beta)).
    instance = ProcessType(**KWARGS, rng=np.random.default_rng(1))
    instance.sample_paths(400, 40_000)
    log_variance = 2 * np.log(instance.volatility)

    for index in (0, 10, -1):
        np.testing.assert_allclose(log_variance[:, index].mean(), -1.0, atol=0.03)


def test_large_values_cluster() -> None:
    # The squares of consecutive values are positively correlated.
    paths = ProcessType(**KWARGS, rng=np.random.default_rng(1)).sample_paths(200, 40_000)

    assert np.corrcoef(paths[:, -1] ** 2, paths[:, -2] ** 2)[0, 1] > 0.05


def test_effect_of_the_sign_of_a_value() -> None:
    # With a negative gamma, a negative value raises the variance more.
    instance = ProcessType(**KWARGS, rng=np.random.default_rng(1))
    paths = instance.sample_paths(200, 40_000)
    previous = paths[:, -2]
    variance = instance.volatility[:, -1] ** 2

    after_negative = variance[previous < 0].mean()
    after_positive = variance[previous > 0].mean()
    assert after_negative > 1.05 * after_positive


def test_logarithm_of_the_variance_follows_its_recurrence() -> None:
    instance = ProcessType(**KWARGS, rng=np.random.default_rng(3))
    values = instance.sample(50)
    log_variance = 2 * np.log(instance.volatility)
    innovations = values / instance.volatility

    expected = (
        -0.1
        + 0.2 * (np.abs(innovations[:-1]) - np.sqrt(2 / np.pi))
        - 0.1 * innovations[:-1]
        + 0.9 * log_variance[:-1]
    )
    np.testing.assert_allclose(log_variance[1:], expected)


def test_negative_coefficients_are_allowed() -> None:
    # The variance is an exponential, so it is positive whatever the signs.
    instance = ProcessType(omega=-0.5, alpha=-0.3, gamma=0.2, beta=-0.5)

    assert np.all(instance.sample(200) ** 2 >= 0)
    assert np.all(instance.volatility > 0)


def test_extreme_parameters_stay_finite() -> None:
    instance = ProcessType(
        omega=50.0, alpha=40.0, gamma=0.0, beta=0.99, rng=np.random.default_rng(1)
    )

    assert np.all(np.isfinite(instance.sample(200)))
