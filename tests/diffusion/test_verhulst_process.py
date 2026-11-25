from typing import Any

import numpy as np
import pytest

from samplepaths.diffusion import VerhulstProcess as ProcessType


@pytest.mark.parametrize(
    "name, value, error_type",
    [
        ("capacity", 0, ValueError),
        ("capacity", "a", TypeError),
        ("rate", "a", TypeError),
        ("vol", -1, ValueError),
        ("t", 0, ValueError),
    ],
)
def test_invalid_parameter(name: str, value: Any, error_type: type) -> None:
    with pytest.raises(error_type):
        ProcessType(**{name: value})


def test_str_repr() -> None:
    instance = ProcessType(rate=1.5, capacity=20.0, vol=0.2, t=2.0)

    assert str(instance) == "Verhulst process with rate=1.5, capacity=20.0, vol=0.2 on [0, 2.0]"
    assert repr(instance) == "VerhulstProcess(rate=1.5, capacity=20.0, vol=0.2, t=2.0)"


def test_parameters_are_validated_when_changed() -> None:
    instance = ProcessType(rate=1.5, capacity=20.0, vol=0.2, t=2.0)

    value: Any = 0
    with pytest.raises(ValueError):
        instance.capacity = value


@pytest.mark.parametrize("n", [1, 2, 100])
def test_sample_shape(n: int) -> None:
    samples = ProcessType(rate=1.5, capacity=20.0, vol=0.2, t=2.0).sample(n)

    assert samples.shape == (n + 1,)
    assert samples.dtype == np.float64
    assert np.all(np.isfinite(samples))


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
        ProcessType(rate=1.5, capacity=20.0, vol=0.2, t=2.0).sample(n)


def test_sample_values() -> None:
    instance = ProcessType(
        rate=1.5, capacity=20.0, vol=0.2, t=2.0, rng=np.random.default_rng(1234567890)
    )

    np.testing.assert_allclose(
        instance.sample(5),
        [1.00000000, 1.31204326, 2.05065437, 2.96486111, 4.78355683, 7.32745763],
        rtol=1e-6,
    )


def test_sample_with_initial_values() -> None:
    instance = ProcessType(
        rate=1.5, capacity=20.0, vol=0.2, t=2.0, rng=np.random.default_rng(1234567890)
    )
    samples = instance.sample_with_initial(5, initial=2.0)

    np.testing.assert_allclose(
        samples,
        [2.00000000, 2.56408651, 3.91122149, 5.43657905, 8.36833747, 11.91866895],
        rtol=1e-6,
    )


def test_sample_starts_from_the_default_initial_value() -> None:
    instance = ProcessType(rate=1.5, capacity=20.0, vol=0.2, t=2.0)

    assert instance.sample(3)[0] == 1.0


@pytest.mark.parametrize("initial", ["a", None])
def test_sample_with_initial_invalid(initial: Any) -> None:
    with pytest.raises(TypeError):
        ProcessType(rate=1.5, capacity=20.0, vol=0.2, t=2.0).sample_with_initial(5, initial)


def test_sample_at_is_not_supported() -> None:
    with pytest.raises(NotImplementedError):
        ProcessType(rate=1.5, capacity=20.0, vol=0.2, t=2.0).sample_at(np.array([0.0, 0.5, 1.0]))


def test_few_and_many_realizations_use_the_same_method(monkeypatch: pytest.MonkeyPatch) -> None:
    # Few realizations are advanced with Python floats, many with arrays.
    import samplepaths.diffusion.euler_maruyama_process as module

    paths = 2 * module._VECTORIZE_FROM_PATHS
    together = ProcessType(
        rate=1.5, capacity=20.0, vol=0.2, t=2.0, rng=np.random.default_rng(3)
    ).sample_paths(50, paths)

    monkeypatch.setattr(module, "_VECTORIZE_FROM_PATHS", 10 * paths)
    in_turn = ProcessType(
        rate=1.5, capacity=20.0, vol=0.2, t=2.0, rng=np.random.default_rng(3)
    ).sample_paths(50, paths)

    np.testing.assert_allclose(together, in_turn, rtol=1e-12, atol=1e-14)


def test_follows_the_logistic_curve_without_noise() -> None:
    instance = ProcessType(rate=1.5, capacity=20.0, vol=0.0, t=4.0)
    samples = instance.sample_with_initial(4000, initial=2.0)

    times = instance.times
    expected = 20.0 / (1.0 + (20.0 / 2.0 - 1.0) * np.exp(-1.5 * times))
    np.testing.assert_allclose(samples, expected, rtol=2e-3)


def test_stays_nonnegative() -> None:
    instance = ProcessType(rate=0.5, capacity=5.0, vol=3.0, t=2.0, rng=np.random.default_rng(1))

    assert instance.sample_paths(100, 2000).min() >= 0.0
