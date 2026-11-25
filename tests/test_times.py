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
        "VasicekProcess",
    ],
)
def test_sample_at_not_supported(name: str) -> None:
    instance = getattr(samplepaths, name)()

    with pytest.raises(NotImplementedError):
        instance.sample_at(np.array([0.0, 0.5, 2.0]))
