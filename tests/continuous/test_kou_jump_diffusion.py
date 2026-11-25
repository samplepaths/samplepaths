import math
from typing import Any

import numpy as np
import pytest

from samplepaths.continuous import KouJumpDiffusion as ProcessType

KWARGS: dict[str, Any] = {
    "drift": 0.1,
    "volatility": 0.2,
    "rate": 3.0,
    "up_probability": 0.4,
    "up_rate": 8.0,
    "down_rate": 5.0,
    "t": 2.0,
}


@pytest.mark.parametrize(
    "name, value, error_type",
    [
        ("drift", "a", TypeError),
        ("volatility", -1, ValueError),
        ("rate", -1, ValueError),
        ("up_probability", -0.1, ValueError),
        ("up_probability", 1.1, ValueError),
        ("up_probability", "a", TypeError),
        # the price has no mean unless upward jumps have a mean size below 1
        ("up_rate", 1.0, ValueError),
        ("up_rate", 0.5, ValueError),
        ("down_rate", 0, ValueError),
        ("t", 0, ValueError),
    ],
)
def test_invalid_parameter(name: str, value: Any, error_type: type) -> None:
    with pytest.raises(error_type):
        ProcessType(**{name: value})


def test_str_repr() -> None:
    instance = ProcessType(**KWARGS)

    assert str(instance) == (
        "Kou jump diffusion with drift 0.1, volatility 0.2, rate 3.0, up probab"
        "ility 0.4, up rate 8.0 and down rate 5.0 on [0, 2.0]."
    )
    assert repr(instance) == (
        "KouJumpDiffusion(drift=0.1, volatility=0.2, rate=3.0, up_probability=0"
        ".4, up_rate=8.0, down_rate=5.0, t=2.0)"
    )


@pytest.mark.parametrize("n", [1, 2, 100])
def test_sample_shape(n: int) -> None:
    samples = ProcessType(**KWARGS).sample(n)

    assert samples.shape == (n + 1,)
    assert samples[0] == 1.0
    assert np.all(samples > 0)


def test_sample_values() -> None:
    instance = ProcessType(**KWARGS, rng=np.random.default_rng(1234567890))

    np.testing.assert_allclose(
        instance.sample(5),
        [1.00000000, 0.83985319, 0.83608719, 0.91590055, 0.73370021, 0.88624158],
        rtol=1e-6,
    )


def test_sample_at_values() -> None:
    instance = ProcessType(**KWARGS, rng=np.random.default_rng(1234567890))
    samples = instance.sample_at(np.array([0.0, 0.5, 2.0, 2.5]))

    np.testing.assert_allclose(samples, [1.00000000, 0.78564569, 2.44551809, 2.46137975], rtol=1e-6)


def test_mean_relative_jump() -> None:
    instance = ProcessType(**KWARGS)

    assert instance.mean_relative_jump == pytest.approx(0.4 * 8 / 7 + 0.6 * 5 / 6 - 1)


@pytest.mark.parametrize("n", [4, 64])
def test_moments(n: int) -> None:
    # The expected value is exp(drift t). A jump has mean p / eta1 - q / eta2
    # and second moment 2 p / eta1^2 + 2 q / eta2^2 in the logarithm.
    instance = ProcessType(**KWARGS, rng=np.random.default_rng(1))
    final = instance.sample_paths(n, 400_000)[:, -1]

    kappa = instance.mean_relative_jump
    jump_mean = 0.4 / 8.0 - 0.6 / 5.0
    jump_second = 2 * 0.4 / 8.0**2 + 2 * 0.6 / 5.0**2
    log_mean = (0.1 - 0.2**2 / 2 - 3.0 * kappa + 3.0 * jump_mean) * 2.0
    log_variance = (0.2**2 + 3.0 * jump_second) * 2.0

    np.testing.assert_allclose(final.mean(), math.exp(0.1 * 2.0), rtol=0.01)
    np.testing.assert_allclose(np.log(final).mean(), log_mean, atol=0.01)
    np.testing.assert_allclose(np.log(final).var(), log_variance, rtol=0.02)


def test_downward_jumps_give_a_negative_skewness() -> None:
    instance = ProcessType(
        drift=0.0,
        volatility=0.1,
        rate=2.0,
        up_probability=0.1,
        up_rate=20.0,
        down_rate=3.0,
        rng=np.random.default_rng(1),
    )
    logs = np.log(instance.sample_paths(16, 100_000)[:, -1])
    centered = logs - logs.mean()

    assert (centered**3).mean() / logs.std() ** 3 < -0.5
