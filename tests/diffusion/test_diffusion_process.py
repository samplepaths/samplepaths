from typing import Any

import numpy as np
import pytest

from samplepaths.diffusion import DiffusionProcess as ProcessType


@pytest.mark.parametrize(
    "speed, error_type",
    [
        (
            "invalid",
            TypeError,
        ),
        (
            lambda x, y: x + y,
            ValueError,
        ),
    ],
)
def test_diffusion_process_invalid_speed(
    speed: Any,
    error_type: type,
) -> None:
    with pytest.raises(error_type):
        ProcessType(speed=speed)


def test_diffusion_process_str_repr() -> None:
    instance = ProcessType(speed=1.0, mean=0.0, vol=1.0, t=2.0)

    assert "Diffusion process with speed=" in str(instance)
    assert "DiffusionProcess(speed=" in repr(instance)


def test_diffusion_process_sample_shape() -> None:
    instance = ProcessType(
        speed=1.0,
        mean=0.0,
        vol=1.0,
        volexp=0.0,
        t=1.0,
    )
    n = 1000
    samples = instance.sample(n=n)

    assert samples.shape == (n + 1,)


@pytest.mark.parametrize(
    "speed,mean,vol,volexp, t, n, expected",
    [
        (
            1,
            1.0,
            1.0,
            0.0,
            1.0,
            5,
            [1.0, 0.08798519, 0.27855002, 0.09520856, 0.63807354, 0.97671214],
        ),
        (
            2,
            1.0,
            5,
            0.0,
            1.0,
            5,
            [1.0, -3.56007407, -1.69523508, -2.25529831, 0.85635447, 2.24507924],
        ),
    ],
)
def test_diffusion_process_sample_values(
    speed: float,
    mean: float,
    vol: float,
    volexp: float,
    t: float,
    n: int,
    expected: list[float],
) -> None:
    rng = np.random.default_rng(1234567890)

    instance = ProcessType(
        speed=speed,
        mean=mean,
        vol=vol,
        volexp=volexp,
        t=t,
        rng=rng,
    )

    samples = instance.sample(n=n)
    np.testing.assert_allclose(samples, expected, rtol=1e-6)


def _euler_maruyama(
    instance: ProcessType,
    noise: Any,
    initial: float,
) -> list[float]:
    """Apply the Euler-Maruyama scheme step by step with NumPy scalars."""
    n = len(noise)
    delta_t = instance.t / n
    x = np.float64(initial)
    values = [float(x)]
    t = 0.0
    with np.errstate(all="ignore"):
        for k in range(n):
            t += delta_t
            x = x + (
                instance.speed(t) * (instance.mean(t) - x) * delta_t
                + instance.vol(t) * x ** instance.volexp(float(x)) * noise[k]
            )
            values.append(float(x))
    return values


@pytest.mark.parametrize(
    "kwargs, initial",
    [
        # constant parameters, one case per branch of the volatility exponent
        ({"speed": 1.5, "mean": 0.5, "vol": 0.3, "volexp": 0.0}, 2.0),
        ({"speed": -0.3, "mean": 1.0, "vol": 0.4, "volexp": 1.0}, 2.0),
        ({"speed": 1.0, "mean": 1.0, "vol": 0.3, "volexp": 0.5}, 2.0),
        ({"speed": 2, "mean": 1, "vol": 1, "volexp": 2}, 1),
        # functions for some or all of the parameters
        ({"speed": lambda t: 1 + t, "mean": lambda t: np.cos(t), "vol": 0.3, "volexp": 0.0}, 1.2),
        (
            {
                "speed": lambda t: 1 + t,
                "mean": lambda t: np.float64(t * t),
                "vol": lambda t: 0.2 + 0.1 * t,
                "volexp": lambda x: 0.5 if x > 1 else 1.0,
            },
            1.2,
        ),
        # the state becomes negative, after which a fractional power is nan
        ({"speed": 1.0, "mean": 0.1, "vol": 3.0, "volexp": 0.5}, 0.2),
    ],
)
def test_diffusion_process_matches_euler_maruyama(
    kwargs: dict[str, Any],
    initial: float,
) -> None:
    n = 200
    seed = 2468

    instance = ProcessType(t=2.0, rng=np.random.default_rng(seed), **kwargs)
    with np.errstate(all="ignore"):
        samples = instance.sample_with_initial(n=n, initial=initial)

    # The realization draws its Gaussian noise first, in one call.
    noise = np.random.default_rng(seed).normal(scale=np.sqrt(2.0 / n), size=n)
    expected = _euler_maruyama(instance, noise, initial)

    assert samples.dtype == np.float64
    np.testing.assert_array_equal(samples, expected)


def test_diffusion_process_negative_state_with_fractional_exponent() -> None:
    instance = ProcessType(volexp=0.5, rng=np.random.default_rng(1))

    with pytest.warns(RuntimeWarning):
        samples = instance.sample_with_initial(n=4, initial=-1.0)

    assert samples[0] == -1.0
    assert np.all(np.isnan(samples[1:]))


def test_diffusion_process_zero_state_with_negative_exponent() -> None:
    instance = ProcessType(volexp=-1.0, rng=np.random.default_rng(1))

    with pytest.warns(RuntimeWarning):
        samples = instance.sample_with_initial(n=4, initial=0.0)

    assert samples[0] == 0.0
    assert not np.all(np.isfinite(samples[1:]))


def test_diffusion_process_parameter_change_after_construction() -> None:
    instance = ProcessType(speed=1.0, mean=0.0, vol=1.0, rng=np.random.default_rng(5))

    instance.speed = lambda t: 2.0 + t
    instance.mean = 0.5
    instance.vol = 0.2
    instance.volexp = 1.0
    samples = instance.sample(n=50)

    fresh = ProcessType(
        speed=lambda t: 2.0 + t,
        mean=0.5,
        vol=0.2,
        volexp=1.0,
        rng=np.random.default_rng(5),
    )
    np.testing.assert_array_equal(samples, fresh.sample(n=50))
