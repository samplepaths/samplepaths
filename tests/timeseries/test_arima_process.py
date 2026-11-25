from typing import Any

import numpy as np
import pytest

from samplepaths.timeseries import ARIMAProcess as ProcessType
from samplepaths.timeseries import AutoregressiveProcess, MovingAverageProcess

KWARGS: dict[str, Any] = {"ar": [0.5], "ma": [0.3], "d": 1, "scale": 1.0}


@pytest.mark.parametrize(
    "name, value, error_type",
    [
        ("ar", "a", TypeError),
        ("ma", ["a"], TypeError),
        ("d", -1, ValueError),
        ("d", 1.5, TypeError),
        ("scale", 0, ValueError),
    ],
)
def test_invalid_parameter(name: str, value: Any, error_type: type) -> None:
    with pytest.raises(error_type):
        ProcessType(**{name: value})


def test_str_repr() -> None:
    instance = ProcessType(**KWARGS)

    assert str(instance) == ("ARIMA(1, 1, 1) process with ar=(0.5,), ma=(0.3,) and scale 1.0.")
    assert repr(instance) == ("ARIMAProcess(ar=(0.5,), ma=(0.3,), d=1, scale=1.0)")


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
        [-2.03932712, -3.65253831, -5.18627496, -5.36367718, -4.61424340, -3.78736125],
        rtol=1e-6,
    )


def test_without_differencing_is_an_arma_process() -> None:
    instance = ProcessType(ar=0.5, ma=0.3, rng=np.random.default_rng(3))
    samples = instance.sample(50)
    noise = np.random.default_rng(3).normal(size=50)

    expected = np.zeros(50)
    for t in range(50):
        expected[t] = noise[t]
        if t >= 1:
            expected[t] += 0.5 * expected[t - 1] + 0.3 * noise[t - 1]

    np.testing.assert_allclose(samples, expected)


@pytest.mark.parametrize("d", [1, 2])
def test_differences_follow_the_arma_process(d: int) -> None:
    integrated = ProcessType(ar=0.5, ma=0.3, d=d, rng=np.random.default_rng(3)).sample(40)
    arma = ProcessType(ar=0.5, ma=0.3, rng=np.random.default_rng(3)).sample(40)

    differences = integrated
    for _ in range(d):
        differences = np.diff(differences, prepend=0.0)

    np.testing.assert_allclose(differences, arma, atol=1e-9)


def test_matches_the_autoregressive_and_moving_average_processes() -> None:
    ar = ProcessType(ar=[0.5, -0.3], rng=np.random.default_rng(3)).sample(30)
    ma = ProcessType(ma=[0.6, 0.2], rng=np.random.default_rng(3)).sample(30)

    reference_ar = AutoregressiveProcess(coefficients=[0.5, -0.3], rng=np.random.default_rng(3))
    reference_ma = MovingAverageProcess(coefficients=[0.6, 0.2], rng=np.random.default_rng(3))

    np.testing.assert_allclose(ar, reference_ar.sample(30))
    np.testing.assert_allclose(ma, reference_ma.sample(30))


def test_default_is_white_noise() -> None:
    samples = ProcessType(scale=2.0, rng=np.random.default_rng(3)).sample(20)

    np.testing.assert_allclose(samples, np.random.default_rng(3).normal(scale=2.0, size=20))


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
