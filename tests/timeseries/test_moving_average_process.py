from typing import Any

import numpy as np
import pytest

from samplepaths.timeseries import MovingAverageProcess as ProcessType

KWARGS: dict[str, Any] = {"coefficients": [0.6, 0.2], "scale": 1.5}


@pytest.mark.parametrize(
    "name, value, error_type",
    [
        ("coefficients", "a", TypeError),
        ("coefficients", [None], TypeError),
        ("scale", 0, ValueError),
        ("scale", "a", TypeError),
    ],
)
def test_invalid_parameter(name: str, value: Any, error_type: type) -> None:
    with pytest.raises(error_type):
        ProcessType(**{name: value})


def test_str_repr() -> None:
    instance = ProcessType(**KWARGS)

    assert str(instance) == (
        "Moving average process of order 2 with coefficients (0.6, 0.2) and scale 1.5."
    )
    assert repr(instance) == ("MovingAverageProcess(coefficients=(0.6, 0.2), scale=1.5)")


@pytest.mark.parametrize("n", [1, 2, 100])
def test_sample_shape(n: int) -> None:
    samples = ProcessType(**KWARGS).sample(n)

    assert samples.shape == (n,)
    assert samples.dtype == np.float64


@pytest.mark.parametrize(
    "n, error_type",
    [
        (0, ValueError),
        (-1, ValueError),
        (1.5, TypeError),
    ],
)
def test_sample_invalid_n(n: Any, error_type: type) -> None:
    with pytest.raises(error_type):
        ProcessType(**KWARGS).sample(n)


def test_sample_values() -> None:
    instance = ProcessType(**KWARGS, rng=np.random.default_rng(1234567890))

    np.testing.assert_allclose(
        instance.sample(6),
        [-3.05899068, -1.80801865, -1.69428198, 0.56000151, 1.40158206, 1.18893451],
        rtol=1e-6,
    )


def test_follows_its_definition() -> None:
    instance = ProcessType(**KWARGS, rng=np.random.default_rng(3))
    samples = instance.sample(50)
    noise = np.random.default_rng(3).normal(scale=1.5, size=50)

    expected = noise.copy()
    expected[1:] += 0.6 * noise[:-1]
    expected[2:] += 0.2 * noise[:-2]

    np.testing.assert_allclose(samples, expected)


def test_variance_and_covariances() -> None:
    # The variance is scale^2 (1 + theta1^2 + theta2^2), the covariance at lag
    # 1 is scale^2 (theta1 + theta1 theta2), and values more than 2 steps
    # apart are independent.
    paths = ProcessType(**KWARGS, rng=np.random.default_rng(1)).sample_paths(50, 100_000)

    np.testing.assert_allclose(paths[:, -1].var(), 2.25 * 1.4, rtol=0.02)
    np.testing.assert_allclose(np.cov(paths[:, -1], paths[:, -2])[0, 1], 2.25 * 0.72, rtol=0.03)
    np.testing.assert_allclose(np.cov(paths[:, -1], paths[:, -4])[0, 1], 0.0, atol=0.03)


@pytest.mark.parametrize("paths", [3, 5000])
def test_sample_paths_methods_agree(paths: int, monkeypatch: pytest.MonkeyPatch) -> None:
    # Many realizations advance together, few are filtered one after the
    # other. From the same noise, both give the same values.
    import samplepaths.timeseries._filters as module

    def moving_average_paths(process: Any, n: int, count: int, together: bool) -> Any:
        noise = process.rng.normal(scale=process.scale, size=(count, n))
        return module.apply_moving_average(noise.T, process._polynomials()[1]).T

    monkeypatch.setattr(ProcessType, "_moving_average_paths", moving_average_paths)
    results = []
    for together in (True, False):
        monkeypatch.setattr(ProcessType, "_advance_together", lambda self, paths, t=together: t)
        instance = ProcessType(**KWARGS, rng=np.random.default_rng(3))
        results.append(instance.sample_paths(40, paths))

    np.testing.assert_allclose(results[0], results[1], rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("paths", [3, 5000])
def test_sample_paths_matches_sample(paths: int) -> None:
    # A realization is the filtered noise, whichever method generates it.
    instance = ProcessType(**KWARGS, rng=np.random.default_rng(3))
    values = instance.sample_paths(300, paths, workers=2)

    assert values.shape == (paths, 300)
    single = ProcessType(**KWARGS, rng=np.random.default_rng(4))
    reference = np.array([single.sample(300) for _ in range(2000)])
    if paths >= 2000:
        np.testing.assert_allclose(values[:, 5].std(), reference[:, 5].std(), rtol=0.1)
        np.testing.assert_allclose(values[:, -1].std(), reference[:, -1].std(), rtol=0.1)
