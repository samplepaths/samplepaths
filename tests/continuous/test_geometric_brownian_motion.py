from typing import Any

import numpy as np
import pytest

from samplepaths.continuous import (
    GeometricBrownianMotion as ProcessType,
)


@pytest.mark.parametrize(
    "drift, volatility, error_type",
    [
        (1.0, None, TypeError),
        (None, 1.0, TypeError),
        (1.0, -1.0, ValueError),
        (1.0, 0.0, ValueError),
    ],
)
def test_geometric_brownian_motion_invalid_parametrization(
    drift: Any,
    volatility: Any,
    error_type: type,
) -> None:
    with pytest.raises(error_type):
        _ = ProcessType(drift=drift, volatility=volatility)


def test_geometric_brownian_motion_str_repr() -> None:
    instance = ProcessType(drift=1.1, volatility=2.1, t=2.0)

    assert (
        str(instance) == "Geometric Brownian motion with drift 1.1 and volatility 2.1 on [0, 2.0]."
    )
    assert repr(instance) == "GeometricBrownianMotion(drift=1.1, volatility=2.1, t=2.0)"


def test_geometric_brownian_motion_sample_shape() -> None:
    instance = ProcessType(drift=1.1, volatility=0.5, t=1.0)
    n = 1000
    samples = instance.sample(n=n)

    assert samples.shape == (n + 1,)

    t = np.linspace(0, 1.0, n + 1)
    samples_at = instance.sample_at(times=t)

    assert samples_at.shape == (n + 1,)


@pytest.mark.parametrize(
    "drift, volatility, t, n, expected",
    [
        (
            1.1,
            0.5,
            1.0,
            5,
            [1.0, 0.77027521, 0.939952, 0.96972554, 1.41228809, 1.96077108],
        ),
        (
            1.1,
            0.5,
            2.0,
            5,
            [1.0, 0.7750003, 1.15128586, 1.34879154, 2.57311109, 4.58772946],
        ),
    ],
)
def test_geometric_brownian_motion_sample_values(
    drift: float,
    volatility: float,
    t: float,
    n: int,
    expected: list[float],
) -> None:
    rng = np.random.default_rng(1234567890)

    instance = ProcessType(
        drift=drift,
        volatility=volatility,
        t=t,
        rng=rng,
    )

    samples = instance.sample(n=n)
    np.testing.assert_allclose(samples, expected, rtol=1e-6)


@pytest.mark.parametrize(
    "drift, volatility, t, n, expected",
    [
        (
            1.1,
            0.5,
            1.0,
            5,
            [1.0, 0.77027521, 0.939952, 0.96972554, 1.41228809, 1.96077108],
        ),
        (
            1.1,
            0.5,
            2.0,
            5,
            [1.0, 0.7750003, 1.15128586, 1.34879154, 2.57311109, 4.58772946],
        ),
    ],
)
def test_geometric_brownian_motion_sample_at_values(
    drift: float,
    volatility: float,
    t: float,
    n: int,
    expected: list[float],
) -> None:
    rng = np.random.default_rng(1234567890)

    instance = ProcessType(
        drift=drift,
        volatility=volatility,
        t=t,
        rng=rng,
    )

    times = np.linspace(0, t, n + 1)
    samples_at = instance.sample_at(times=times)
    np.testing.assert_allclose(samples_at, expected, rtol=1e-6)


@pytest.mark.parametrize(
    "drift, t",
    [
        (2.0, 1.0),
        (2.0, 5.0),
        (-0.5, 3.0),
    ],
)
def test_geometric_brownian_motion_drift_scales_with_time(
    drift: float,
    t: float,
) -> None:
    # With a negligible volatility the process follows exp(drift * time).
    n = 10
    instance = ProcessType(drift=drift, volatility=1e-12, t=t)
    times = np.linspace(0, t, n + 1)
    expected = np.exp(drift * times)

    np.testing.assert_allclose(instance.sample(n=n), expected, rtol=1e-6)
    np.testing.assert_allclose(instance.sample_at(times=times), expected, rtol=1e-6)
    np.testing.assert_allclose(
        instance.sample_with_initial(n=n, initial=3.0),
        3.0 * expected,
        rtol=1e-6,
    )
