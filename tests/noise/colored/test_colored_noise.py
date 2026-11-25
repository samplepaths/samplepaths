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
            [-0.71658961, -0.06305761, 0.87115416, 0.29551432, -0.38702126],
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
