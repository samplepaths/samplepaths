from typing import Any

import numpy as np
import pytest

from samplepaths.timeseries import AutoregressiveProcess as ProcessType

KWARGS: dict[str, Any] = {"coefficients": [0.5, -0.3], "scale": 2.0}


@pytest.mark.parametrize(
    "name, value, error_type",
    [
        ("coefficients", "a", TypeError),
        ("coefficients", ["a"], TypeError),
        ("coefficients", None, TypeError),
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
        "Autoregressive process of order 2 with coefficients (0.5, -0.3) and scale 2.0."
    )
    assert repr(instance) == ("AutoregressiveProcess(coefficients=(0.5, -0.3), scale=2.0)")


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
        [-4.07865424, -2.00282611, -1.24302918, 1.59782916, 2.36254434, 1.24903765],
        rtol=1e-6,
    )


def test_a_single_number_is_one_coefficient() -> None:
    assert ProcessType(coefficients=0.7).coefficients == (0.7,)


def test_follows_its_recurrence() -> None:
    instance = ProcessType(**KWARGS, rng=np.random.default_rng(3))
    samples = instance.sample(50)
    noise = np.random.default_rng(3).normal(scale=2.0, size=50)

    expected = np.zeros(50)
    for t in range(50):
        expected[t] = noise[t]
        if t >= 1:
            expected[t] += 0.5 * expected[t - 1]
        if t >= 2:
            expected[t] -= 0.3 * expected[t - 2]

    np.testing.assert_allclose(samples, expected)


def test_stationary_variance_and_correlation() -> None:
    # For an AR(2) process the variance is
    # scale^2 (1 - phi2) / ((1 + phi2) ((1 - phi2)^2 - phi1^2))
    # and the correlation of consecutive values is phi1 / (1 - phi2).
    paths = ProcessType(**KWARGS, rng=np.random.default_rng(1)).sample_paths(400, 40_000)

    variance = 4.0 * 1.3 / (0.7 * (1.3**2 - 0.5**2))
    np.testing.assert_allclose(paths[:, -1].var(), variance, rtol=0.03)
    np.testing.assert_allclose(np.corrcoef(paths[:, -1], paths[:, -2])[0, 1], 0.5 / 1.3, atol=0.02)


def test_a_unit_coefficient_is_a_random_walk() -> None:
    walk = ProcessType(coefficients=1.0, rng=np.random.default_rng(3)).sample(20)
    noise = np.random.default_rng(3).normal(size=20)

    np.testing.assert_allclose(walk, np.cumsum(noise))


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
