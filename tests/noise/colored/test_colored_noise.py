from typing import Any

import numpy as np
import pytest

from samplepaths.noise import ColoredNoise as ProcessType


@pytest.mark.parametrize(
    "beta, error_type",
    [
        (
            "invalid",
            TypeError,
        ),
    ],
)
def test_colored_noise_invalid_beta(
    beta: Any,
    error_type: type,
) -> None:
    with pytest.raises(error_type):
        _ = ProcessType(beta=beta)


def test_colored_noise_str_repr() -> None:
    instance = ProcessType(beta=1.5, t=2.0)

    assert str(instance) == "Colored noise generator with exponent 1.5 on interval [0, 2.0]"
    assert repr(instance) == "ColoredNoise(beta=1.5, t=2.0)"


def test_colored_noise_sample_shape() -> None:
    instance = ProcessType(beta=0.5, t=1.0)
    n = 1000
    samples = instance.sample(n=n)

    assert samples.shape == (n,)


@pytest.mark.parametrize(
    "beta, t, n, expected",
    [
        (
            2,
            2.0,
            5,
            [-1.01341075, -0.08917693, 1.23199803, 0.41792037, -0.54733071],
        ),
    ],
)
def test_colored_noise_sample_values(
    beta: float,
    t: float,
    n: int,
    expected: list[float],
) -> None:
    rng = np.random.default_rng(1234567890)

    instance = ProcessType(
        beta=beta,
        t=t,
        rng=rng,
    )

    samples = instance.sample(n=n)

    assert np.allclose(samples, expected, atol=1e-5)


def test_colored_noise_parameter_change_after_sampling() -> None:
    # The frequencies and their scale are cached between realizations and
    # must be recomputed when beta or t changes.
    instance = ProcessType(beta=0.0, t=1.0, rng=np.random.default_rng(7))
    instance.sample(n=16)

    instance.beta = 2.0
    instance.t = 3.0
    instance.rng = np.random.default_rng(11)
    samples = instance.sample(n=16)

    fresh = ProcessType(beta=2.0, t=3.0, rng=np.random.default_rng(11))
    np.testing.assert_allclose(samples, fresh.sample(n=16))
