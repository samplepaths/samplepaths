from typing import Any

import numpy as np
import pytest

from samplepaths.continuous import CompoundPoissonProcess as ProcessType


@pytest.mark.parametrize(
    "name, value, error_type",
    [
        ("rate", 0, ValueError),
        ("rate", -1, ValueError),
        ("rate", "a", TypeError),
        ("jumps", 1.0, TypeError),
        ("t", 0, ValueError),
    ],
)
def test_invalid_parameter(name: str, value: Any, error_type: type) -> None:
    with pytest.raises(error_type):
        ProcessType(**{name: value})


def test_str_repr() -> None:
    instance = ProcessType(rate=3.0, t=2.0)

    assert str(instance) == "Compound Poisson process with rate 3.0 on [0, 2.0]."
    assert repr(instance) == "CompoundPoissonProcess(rate=3.0, t=2.0)"


@pytest.mark.parametrize("n", [1, 2, 100])
def test_sample_shape(n: int) -> None:
    samples = ProcessType(rate=3.0, t=2.0).sample(n)

    assert samples.shape == (n + 1,)
    assert samples[0] == 0.0


def test_sample_values() -> None:
    instance = ProcessType(rate=3.0, t=2.0, rng=np.random.default_rng(1234567890))

    np.testing.assert_allclose(
        instance.sample(5),
        [0.00000000, -0.78052644, 1.17337971, 2.84378613, 2.27808994, 3.54583911],
        rtol=1e-6,
    )


def test_sample_at_values() -> None:
    instance = ProcessType(rate=3.0, rng=np.random.default_rng(1234567890))
    samples = instance.sample_at(np.array([0.0, 0.5, 2.0, 2.5]))

    np.testing.assert_allclose(
        samples, [0.00000000, -1.41762623, -1.65489456, -0.77511546], rtol=1e-6
    )


def test_sample_at_without_zero() -> None:
    samples = ProcessType(rate=3.0).sample_at(np.array([0.5, 2.0, 2.5]))

    assert samples.shape == (3,)


def test_moments_with_custom_jumps() -> None:
    # With exponential jumps of mean m, the value at time t has mean
    # rate t m and variance rate t 2 m^2.
    instance = ProcessType(
        rate=4.0,
        jumps=lambda rng, size: rng.exponential(2.0, size),
        t=3.0,
        rng=np.random.default_rng(1),
    )
    final = instance.sample_paths(32, 100_000)[:, -1]

    np.testing.assert_allclose(final.mean(), 4.0 * 3.0 * 2.0, rtol=0.01)
    np.testing.assert_allclose(final.var(), 4.0 * 3.0 * 8.0, rtol=0.03)


def test_is_constant_between_jumps() -> None:
    # With a low rate most increments hold no jump.
    samples = ProcessType(rate=0.5, t=1.0, rng=np.random.default_rng(2)).sample(1000)

    assert np.mean(np.diff(samples) == 0.0) > 0.99


def test_jump_function_must_return_one_size_per_jump() -> None:
    instance = ProcessType(rate=50.0, jumps=lambda rng, size: rng.normal(size=3))

    with pytest.raises(ValueError):
        instance.sample(100)
