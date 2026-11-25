import math
from typing import Any

import numpy as np
import pytest

from samplepaths.continuous import NormalInverseGaussianProcess as ProcessType


@pytest.mark.parametrize(
    "kwargs, error_type",
    [
        ({"alpha": 0}, ValueError),
        ({"alpha": "a"}, TypeError),
        ({"beta": "a"}, TypeError),
        # the asymmetry must be smaller than the tail heaviness
        ({"alpha": 1.0, "beta": 1.0}, ValueError),
        ({"alpha": 1.0, "beta": -2.0}, ValueError),
        ({"delta": 0}, ValueError),
        ({"mu": "a"}, TypeError),
        ({"t": 0}, ValueError),
    ],
)
def test_invalid_parameter(kwargs: dict[str, Any], error_type: type) -> None:
    with pytest.raises(error_type):
        ProcessType(**kwargs)


def test_parameters_are_validated_when_changed() -> None:
    instance = ProcessType(alpha=2.0, beta=0.5)

    with pytest.raises(ValueError):
        instance.beta = 2.5
    with pytest.raises(ValueError):
        instance.alpha = 0.25


def test_str_repr() -> None:
    instance = ProcessType(alpha=2.0, beta=0.5, delta=1.5, mu=0.1, t=2.0)

    assert str(instance) == (
        "Normal inverse Gaussian process with alpha=2.0, beta=0.5, delta=1.5, mu=0.1 on [0, 2.0]"
    )
    assert repr(instance) == (
        "NormalInverseGaussianProcess(alpha=2.0, beta=0.5, delta=1.5, mu=0.1, t=2.0)"
    )


@pytest.mark.parametrize("n", [1, 2, 100])
def test_sample_shape(n: int) -> None:
    samples = ProcessType().sample(n)

    assert samples.shape == (n + 1,)
    assert samples[0] == 0.0


def test_sample_values() -> None:
    instance = ProcessType(
        alpha=2.0, beta=0.5, delta=1.5, mu=0.1, t=2.0, rng=np.random.default_rng(1234567890)
    )

    np.testing.assert_allclose(
        instance.sample(5),
        [0.00000000, -0.27104126, -0.01178704, -0.00706325, 0.20116579, -0.02316684],
        rtol=1e-6,
    )


def test_sample_at_values() -> None:
    instance = ProcessType(
        alpha=2.0, beta=0.5, delta=1.5, mu=0.1, rng=np.random.default_rng(1234567890)
    )
    samples = instance.sample_at(np.array([0.0, 0.5, 2.0, 2.5]))

    np.testing.assert_allclose(
        samples, [0.00000000, -0.05800341, 1.64193190, 1.69073783], rtol=1e-6
    )


def test_sample_at_without_zero() -> None:
    samples = ProcessType().sample_at(np.array([0.5, 2.0, 2.5]))

    assert samples.shape == (3,)


@pytest.mark.parametrize("n", [4, 64])
def test_moments(n: int) -> None:
    # The value at time t has mean t (mu + delta beta / gamma) and variance
    # t delta alpha^2 / gamma^3, whatever the number of increments.
    alpha, beta, delta, mu, t = 2.0, 0.5, 1.5, 0.1, 2.0
    gamma = math.sqrt(alpha**2 - beta**2)

    instance = ProcessType(
        alpha=alpha, beta=beta, delta=delta, mu=mu, t=t, rng=np.random.default_rng(1)
    )
    final = instance.sample_paths(n, 200_000)[:, -1]

    np.testing.assert_allclose(final.mean(), t * (mu + delta * beta / gamma), rtol=0.01)
    np.testing.assert_allclose(final.var(), t * delta * alpha**2 / gamma**3, rtol=0.02)


def test_asymmetry_sets_the_sign_of_the_skewness() -> None:
    def skewness(beta: float) -> float:
        instance = ProcessType(alpha=2.0, beta=beta, rng=np.random.default_rng(1))
        final = instance.sample_paths(16, 100_000)[:, -1]
        centered = final - final.mean()
        return float((centered**3).mean() / final.std() ** 3)

    assert skewness(1.5) > 0.3
    assert skewness(-1.5) < -0.3
    assert abs(skewness(0.0)) < 0.05
