from typing import Any

import numpy as np
import pytest

from samplepaths.timeseries import ARIMAProcess
from samplepaths.timeseries import SARIMAProcess as ProcessType

KWARGS: dict[str, Any] = {
    "ar": [0.5],
    "ma": [0.3],
    "seasonal_ar": [0.2],
    "seasonal_ma": [0.4],
    "d": 0,
    "seasonal_d": 1,
    "period": 4,
    "scale": 1.0,
}


@pytest.mark.parametrize(
    "name, value, error_type",
    [
        ("ar", "a", TypeError),
        ("seasonal_ma", ["a"], TypeError),
        ("d", -1, ValueError),
        ("seasonal_d", -1, ValueError),
        ("period", 0, ValueError),
        ("period", 1.5, TypeError),
        ("scale", 0, ValueError),
    ],
)
def test_invalid_parameter(name: str, value: Any, error_type: type) -> None:
    with pytest.raises(error_type):
        ProcessType(**{name: value})


def test_str_repr() -> None:
    instance = ProcessType(**KWARGS)

    assert str(instance) == ("SARIMA(1, 0, 1)(1, 1, 1) process with period 4 and scale 1.0.")
    assert repr(instance) == (
        "SARIMAProcess(ar=(0.5,), ma=(0.3,), seasonal_ar=(0.2,), s"
        "easonal_ma=(0.4,), d=0, seasonal_d=1, period=4, scale=1.0)"
    )


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
        [-2.03932712, -1.61321119, -1.53373665, -0.17740222, -2.51348961, -1.75425575],
        rtol=1e-6,
    )


def test_seasonal_differences_follow_the_remaining_model() -> None:
    kwargs: dict[str, Any] = {
        "ar": 0.5,
        "ma": 0.3,
        "seasonal_ar": 0.2,
        "seasonal_ma": 0.4,
        "period": 4,
    }
    integrated = ProcessType(seasonal_d=1, rng=np.random.default_rng(3), **kwargs).sample(40)
    stationary = ProcessType(rng=np.random.default_rng(3), **kwargs).sample(40)

    # x[t] - x[t - period], with zero before the first value
    differences = integrated.copy()
    differences[4:] -= integrated[:-4]

    np.testing.assert_allclose(differences, stationary, atol=1e-9)


def test_seasonal_and_plain_polynomials_are_multiplied() -> None:
    # (1 - 0.5 B) (1 - 0.3 B^3) X = noise
    instance = ProcessType(ar=0.5, seasonal_ar=0.3, period=3, rng=np.random.default_rng(3))
    samples = instance.sample(30)
    noise = np.random.default_rng(3).normal(size=30)

    expected = np.zeros(30)
    for t in range(30):
        expected[t] = noise[t]
        if t >= 1:
            expected[t] += 0.5 * expected[t - 1]
        if t >= 3:
            expected[t] += 0.3 * expected[t - 3]
        if t >= 4:
            expected[t] -= 0.15 * expected[t - 4]

    np.testing.assert_allclose(samples, expected)


def test_without_seasonal_terms_is_an_arima_process() -> None:
    seasonal = ProcessType(ar=0.5, ma=0.3, d=1, rng=np.random.default_rng(3)).sample(30)
    plain = ARIMAProcess(ar=0.5, ma=0.3, d=1, rng=np.random.default_rng(3)).sample(30)

    np.testing.assert_allclose(seasonal, plain)


def test_seasonal_pattern_repeats() -> None:
    # With seasonal differencing the values one period apart stay close.
    instance = ProcessType(seasonal_d=1, period=12, scale=0.1, rng=np.random.default_rng(1))
    paths = instance.sample_paths(240, 2000)

    same_season = np.corrcoef(paths[:, -1], paths[:, -13])[0, 1]
    other_season = np.corrcoef(paths[:, -1], paths[:, -7])[0, 1]
    assert same_season > 0.9
    assert abs(other_season) < 0.1


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
