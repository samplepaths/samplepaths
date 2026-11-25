from typing import Any

import numpy as np
import pytest

import samplepaths
from samplepaths.base import BaseTimeProcess

# Processes without default parameters.
_REQUIRED_KWARGS: dict[str, dict[str, Any]] = {
    "GammaProcess": {"mean": 1.0, "variance": 1.0},
}

TIME_PROCESS_NAMES = [
    name
    for name in samplepaths.__all__
    if isinstance(getattr(samplepaths, name), type)
    and issubclass(getattr(samplepaths, name), BaseTimeProcess)
]


@pytest.mark.parametrize("name", TIME_PROCESS_NAMES)
@pytest.mark.parametrize("t", [1.0, 2.5])
def test_times_match_last_sample(name: str, t: float) -> None:
    n = 8
    process_type = getattr(samplepaths, name)
    instance = process_type(t=t, **_REQUIRED_KWARGS.get(name, {}))

    instance.sample(n)

    assert instance.n == n
    np.testing.assert_allclose(instance.times, np.linspace(0, t, n + 1))

    # Sampling again with a different number of increments updates the times.
    instance.sample(2 * n)

    assert instance.n == 2 * n
    np.testing.assert_allclose(instance.times, np.linspace(0, t, 2 * n + 1))


@pytest.mark.parametrize(
    "name",
    [
        "ColoredNoise",
        "DiffusionProcess",
        "FractionalBrownianMotion",
        "FractionalGaussianNoise",
        "MultifractionalBrownianMotion",
        "SABRModel",
        "HullWhiteTwoFactorModel",
    ],
)
def test_sample_at_not_supported(name: str) -> None:
    instance = getattr(samplepaths, name)()

    with pytest.raises(NotImplementedError):
        instance.sample_at(np.array([0.0, 0.5, 2.0]))


def test_numpy_scalar_arguments() -> None:
    drift: Any = np.int64(1)
    scale: Any = np.float32(2.0)
    p: Any = np.float32(0.25)
    n: Any = np.int64(4)

    instance = samplepaths.BrownianMotion(drift=drift, scale=scale, t=np.float64(3.0))

    assert instance.sample(n).shape == (5,)
    assert samplepaths.BernoulliProcess(p=p).sample(n).shape == (4,)
    assert samplepaths.FractionalGaussianNoise(hurst=np.float64(0.7)).hurst == 0.7


@pytest.mark.parametrize(
    "name, kwargs",
    [
        ("ChineseRestaurantProcess", {}),
        ("MarkovChain", {}),
        ("MoranProcess", {"maximum": 10}),
    ],
)
def test_integer_valued_processes(name: str, kwargs: dict[str, Any]) -> None:
    instance = getattr(samplepaths, name)(**kwargs)

    assert instance.sample(8).dtype == np.int64
