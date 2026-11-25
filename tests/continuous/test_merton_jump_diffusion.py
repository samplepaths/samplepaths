import math
from typing import Any

import numpy as np
import pytest

from samplepaths.continuous import GeometricBrownianMotion
from samplepaths.continuous import MertonJumpDiffusion as ProcessType

KWARGS: dict[str, Any] = {
    "drift": 0.1,
    "volatility": 0.2,
    "rate": 3.0,
    "jump_mean": -0.1,
    "jump_volatility": 0.3,
    "t": 2.0,
}


@pytest.mark.parametrize(
    "name, value, error_type",
    [
        ("drift", "a", TypeError),
        ("volatility", -1, ValueError),
        ("rate", -1, ValueError),
        ("rate", "a", TypeError),
        ("jump_mean", "a", TypeError),
        ("jump_volatility", -1, ValueError),
        ("t", 0, ValueError),
    ],
)
def test_invalid_parameter(name: str, value: Any, error_type: type) -> None:
    with pytest.raises(error_type):
        ProcessType(**{name: value})


def test_str_repr() -> None:
    instance = ProcessType(**KWARGS)

    assert str(instance) == (
        "Merton jump diffusion with drift 0.1, volatility 0.2, rate 3.0, jump m"
        "ean -0.1 and jump volatility 0.3 on [0, 2.0]."
    )
    assert repr(instance) == (
        "MertonJumpDiffusion(drift=0.1, volatility=0.2, rate=3.0, jump_mean=-0."
        "1, jump_volatility=0.3, t=2.0)"
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
        [1.00000000, 0.85066327, 0.87053407, 0.81946274, 1.57975008, 1.43201627],
        rtol=1e-6,
    )


def test_sample_at_values() -> None:
    instance = ProcessType(**KWARGS, rng=np.random.default_rng(1234567890))
    samples = instance.sample_at_with_initial(np.array([0.0, 0.5, 2.0, 2.5]), initial=10.0)

    np.testing.assert_allclose(
        samples, [10.00000000, 7.28162372, 17.57058476, 11.50621966], rtol=1e-6
    )


def test_sample_at_without_zero() -> None:
    samples = ProcessType(**KWARGS).sample_at(np.array([0.5, 2.0, 2.5]))

    assert samples.shape == (3,)


@pytest.mark.parametrize("initial", [0, -1.0])
def test_sample_with_initial_invalid(initial: float) -> None:
    with pytest.raises(ValueError):
        ProcessType(**KWARGS).sample_with_initial(5, initial=initial)


def test_mean_relative_jump() -> None:
    instance = ProcessType(**KWARGS)

    assert instance.mean_relative_jump == pytest.approx(math.exp(-0.1 + 0.3**2 / 2) - 1)


@pytest.mark.parametrize("n", [4, 64])
def test_moments(n: int) -> None:
    # The expected value is initial exp(drift t), whatever the jumps and the
    # number of increments. The logarithm has mean
    # (drift - volatility^2 / 2 - rate kappa + rate jump_mean) t and variance
    # (volatility^2 + rate (jump_mean^2 + jump_volatility^2)) t.
    instance = ProcessType(**KWARGS, rng=np.random.default_rng(1))
    final = instance.sample_paths_with_initial(n, 400_000, initial=5.0)[:, -1]

    kappa = instance.mean_relative_jump
    log_mean = math.log(5.0) + (0.1 - 0.2**2 / 2 - 3.0 * kappa + 3.0 * -0.1) * 2.0
    log_variance = (0.2**2 + 3.0 * (0.1**2 + 0.3**2)) * 2.0

    np.testing.assert_allclose(final.mean(), 5.0 * math.exp(0.1 * 2.0), rtol=0.01)
    np.testing.assert_allclose(np.log(final).mean(), log_mean, atol=0.01)
    np.testing.assert_allclose(np.log(final).var(), log_variance, rtol=0.02)


def test_without_jumps_is_a_geometric_brownian_motion() -> None:
    jumps = ProcessType(drift=0.1, volatility=0.2, rate=0.0, t=2.0, rng=np.random.default_rng(3))
    plain = GeometricBrownianMotion(drift=0.1, volatility=0.2, t=2.0, rng=np.random.default_rng(3))

    np.testing.assert_allclose(jumps.sample_paths(16, 50), plain.sample_paths(16, 50))
