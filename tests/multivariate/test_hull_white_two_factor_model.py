import math
from typing import Any

import numpy as np
import pytest

from samplepaths.multivariate import HullWhiteTwoFactorModel as ProcessType

KWARGS: dict[str, Any] = {
    "speed": 1.5,
    "drift": 0.5,
    "vol": 0.3,
    "factor_speed": 0.8,
    "factor_vol": 0.2,
    "correlation": -0.4,
}


@pytest.mark.parametrize(
    "kwargs, error_type",
    [
        ({"speed": 0}, ValueError),
        ({"speed": "a"}, TypeError),
        ({"drift": "a"}, TypeError),
        ({"vol": -1}, ValueError),
        ({"vol": "a"}, TypeError),
        ({"factor_speed": 0}, ValueError),
        ({"factor_vol": -1}, ValueError),
        ({"correlation": 2}, ValueError),
        ({"correlation": "a"}, TypeError),
        ({"t": 0}, ValueError),
    ],
)
def test_invalid_parameters(kwargs: dict[str, Any], error_type: type) -> None:
    with pytest.raises(error_type):
        ProcessType(**{**KWARGS, **kwargs})


def test_str_repr() -> None:
    instance = ProcessType(**KWARGS)

    assert str(instance).startswith("Two-factor Hull-White model with speed=1")
    assert repr(instance).startswith("HullWhiteTwoFactorModel(speed=")
    assert repr(instance).endswith("-0.4, t=1.0)")


@pytest.mark.parametrize("n", [1, 2, 100])
def test_sample_shape(n: int) -> None:
    instance = ProcessType(**KWARGS)
    samples = instance.sample(n)

    assert instance.components == ("rate", "factor")
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
        samples[0], [0.00000000, -0.18089907, 0.04871035, 0.07886627, 0.31777512], rtol=1e-6
    )
    np.testing.assert_allclose(
        samples[1], [0.00000000, 0.13613878, 0.13325292, 0.08838602, 0.09019300], rtol=1e-6
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

    realization = instance.sample_with_initial(10, initial=0.03)
    assert realization[0, 0] == 0.03
    assert realization[1, 0] == 0.0
    paths = instance.sample_paths_with_initial(10, 40, initial=0.03, workers=2)
    assert paths.shape == (40, 2, 11)
    assert np.all(paths[:, 0, 0] == 0.03)
    with pytest.raises(TypeError):
        instance.sample_with_initial(10, initial="a")  # type: ignore[arg-type]


def test_steps_follow_the_euler_method() -> None:
    instance = ProcessType(**KWARGS, t=2.0, rng=np.random.default_rng(3))
    rate, factor = instance.sample_with_initial(50, initial=0.1)

    # Recover the increments of the two Wiener processes from the factor,
    # then check the recurrence of the rate.
    delta_t = 2.0 / 50
    noise_factor = (factor[1:] - (1 - 0.8 * delta_t) * factor[:-1]) / 0.2
    noise_rate = (rate[1:] - rate[:-1] - (0.5 + factor[:-1] - 1.5 * rate[:-1]) * delta_t) / 0.3
    np.testing.assert_allclose(noise_factor.std(), math.sqrt(delta_t), rtol=0.4)
    np.testing.assert_allclose(noise_rate.std(), math.sqrt(delta_t), rtol=0.4)


def test_without_noise_the_rate_goes_to_its_level() -> None:
    instance = ProcessType(speed=2.0, drift=0.06, vol=0.0, factor_vol=0.0, t=10.0)
    rate, factor = instance.sample_with_initial(2000, initial=0.01)

    assert np.all(factor == 0)
    np.testing.assert_allclose(rate[-1], 0.03, rtol=1e-3)


def test_drift_given_as_a_function() -> None:
    constant = ProcessType(**KWARGS, rng=np.random.default_rng(1)).sample(20)
    function = ProcessType(**{**KWARGS, "drift": lambda t: 0.5}, rng=np.random.default_rng(1))

    assert function.drift(2.0) == 0.5
    np.testing.assert_allclose(function.sample(20), constant)


def test_moments() -> None:
    instance = ProcessType(**KWARGS, rng=np.random.default_rng(1))
    paths = instance.sample_paths_with_initial(256, 100_000, initial=0.1, workers=4)
    rate, factor = paths[:, 0, -1], paths[:, 1, -1]

    # The factor is an Ornstein-Uhlenbeck process started at zero.
    np.testing.assert_allclose(factor.mean(), 0.0, atol=0.003)
    np.testing.assert_allclose(factor.var(), 0.2**2 * (1 - math.exp(-1.6)) / 1.6, rtol=0.03)
    # The mean of the rate does not depend on the factor.
    np.testing.assert_allclose(
        rate.mean(), 0.5 / 1.5 + (0.1 - 0.5 / 1.5) * math.exp(-1.5), atol=0.003
    )


def test_correlation_of_the_two_noises() -> None:
    for correlation in (-0.9, 0.9):
        instance = ProcessType(
            **{**KWARGS, "correlation": correlation}, rng=np.random.default_rng(1)
        )
        paths = instance.sample_paths(4, 100_000)
        # After one step the rate and the factor only hold the first noises.
        measured = np.corrcoef(paths[:, 0, 1], paths[:, 1, 1])[0, 1]

        np.testing.assert_allclose(measured, correlation, atol=0.01)
