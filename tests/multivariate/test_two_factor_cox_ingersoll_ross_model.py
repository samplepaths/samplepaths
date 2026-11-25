import math
from typing import Any

import numpy as np
import pytest

from samplepaths.multivariate import TwoFactorCoxIngersollRossModel as ProcessType

KWARGS: dict[str, Any] = {
    "speed": (2.0, 0.2),
    "mean": (0.02, 0.03),
    "vol": (0.1, 0.05),
    "initial": (0.05, 0.01),
}


@pytest.mark.parametrize(
    "kwargs, error_type",
    [
        ({"speed": (0, 1)}, ValueError),
        ({"speed": "a"}, TypeError),
        ({"speed": (1, 2, 3)}, ValueError),
        ({"mean": (-1, 1)}, ValueError),
        ({"mean": "a"}, TypeError),
        ({"vol": (-1, 1)}, ValueError),
        ({"vol": (1, 2, 3)}, ValueError),
        ({"initial": (-1, 1)}, ValueError),
        ({"initial": "a"}, TypeError),
        ({"t": 0}, ValueError),
    ],
)
def test_invalid_parameters(kwargs: dict[str, Any], error_type: type) -> None:
    with pytest.raises(error_type):
        ProcessType(**{**KWARGS, **kwargs})


def test_str_repr() -> None:
    instance = ProcessType(**KWARGS)

    assert str(instance).startswith("Two-factor Cox-Ingersoll-Ross model with")
    assert repr(instance).startswith("TwoFactorCoxIngersollRossModel")
    assert repr(instance).endswith(".01), t=1.0)")


@pytest.mark.parametrize("n", [1, 2, 100])
def test_sample_shape(n: int) -> None:
    instance = ProcessType(**KWARGS)
    samples = instance.sample(n)

    assert instance.components == ("rate", "first factor", "second factor")
    assert samples.shape == (3, n + 1)
    assert instance.times.shape == (n + 1,)
    assert instance.sample_paths(n, 5).shape == (5, 3, n + 1)


@pytest.mark.parametrize("n, error_type", [(0, ValueError), (-1, ValueError), (1.5, TypeError)])
def test_sample_invalid_n(n: Any, error_type: type) -> None:
    with pytest.raises(error_type):
        ProcessType(**KWARGS).sample(n)


def test_sample_values() -> None:
    instance = ProcessType(**KWARGS, rng=np.random.default_rng(1234567890))
    samples = instance.sample(4)

    np.testing.assert_allclose(
        samples[0], [0.06000000, 0.03662354, 0.03778425, 0.03067824, 0.03217795], rtol=1e-6
    )
    np.testing.assert_allclose(
        samples[1], [0.05000000, 0.02785935, 0.02791424, 0.02131079, 0.01945357], rtol=1e-6
    )
    np.testing.assert_allclose(
        samples[2], [0.01000000, 0.00876419, 0.00987001, 0.00936745, 0.01272438], rtol=1e-6
    )


def test_sample_at_is_not_supported() -> None:
    with pytest.raises(NotImplementedError):
        ProcessType(**KWARGS).sample_at(np.array([0.0, 0.5, 1.0]))


def test_threads_give_reproducible_realizations() -> None:
    first = ProcessType(**KWARGS, rng=np.random.default_rng(5)).sample_paths(16, 20_000, workers=4)
    again = ProcessType(**KWARGS, rng=np.random.default_rng(5)).sample_paths(16, 20_000, workers=4)

    assert first.shape == (20_000, 3, 17)
    np.testing.assert_array_equal(first, again)


def test_a_number_is_used_for_both_factors() -> None:
    instance = ProcessType(speed=1.5, mean=0.02, vol=0.1)

    assert instance.speed == (1.5, 1.5)
    assert instance.mean == (0.02, 0.02)
    assert instance.vol == (0.1, 0.1)


def test_initial_values_default_to_the_means() -> None:
    instance = ProcessType(mean=(0.02, 0.03))

    assert instance.initial == (0.02, 0.03)
    np.testing.assert_allclose(instance.sample(3)[:, 0], [0.05, 0.02, 0.03])
    instance.initial = (0.01, 0.04)
    np.testing.assert_allclose(instance.sample(3)[:, 0], [0.05, 0.01, 0.04])
    instance.initial = None
    assert instance.initial == (0.02, 0.03)


def test_rate_is_the_sum_of_the_factors() -> None:
    paths = ProcessType(**KWARGS, rng=np.random.default_rng(1)).sample_paths(16, 1000)

    np.testing.assert_allclose(paths[:, 0], paths[:, 1] + paths[:, 2])
    assert paths.min() >= 0


@pytest.mark.parametrize("n", [2, 32])
def test_factors_have_their_exact_means_whatever_the_step(n: int) -> None:
    instance = ProcessType(**KWARGS, rng=np.random.default_rng(1))
    final = instance.sample_paths(n, 200_000, workers=4)[:, 1:, -1]

    expected = [0.02 + 0.03 * math.exp(-2.0), 0.03 - 0.02 * math.exp(-0.2)]
    np.testing.assert_allclose(final.mean(axis=0), expected, rtol=0.01)


def test_factors_are_independent() -> None:
    final = ProcessType(**KWARGS, rng=np.random.default_rng(1)).sample_paths(8, 100_000)[:, 1:, -1]

    assert abs(np.corrcoef(final[:, 0], final[:, 1])[0, 1]) < 0.01
