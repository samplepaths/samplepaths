import numpy as np
import pytest

from samplepaths.noise import VioletNoise as ProcessType


def test_violet_noise_str_repr() -> None:
    instance = ProcessType(t=2.0)

    assert str(instance) == "Colored noise generator with exponent -2 on interval [0, 2.0]"
    assert repr(instance) == "ColoredNoise(beta=-2, t=2.0)"


def test_violet_noise_sample_shape() -> None:
    instance = ProcessType(t=1.0)
    n = 1000
    samples = instance.sample(n=n)

    assert samples.shape == (n,)


@pytest.mark.parametrize(
    "t, n, expected",
    [
        (
            2.0,
            5,
            [-0.74954518, -0.34216913, 1.35888880, -0.11555393, -0.15162057],
        ),
    ],
)
def test_violet_noise_sample_values(
    t: float,
    n: int,
    expected: list[float],
) -> None:
    rng = np.random.default_rng(1234567890)

    instance = ProcessType(
        t=t,
        rng=rng,
    )

    samples = instance.sample(n=n)

    assert np.allclose(samples, expected, atol=1e-5)
