from typing import Any

import numpy as np
import pytest

from samplepaths.diffusion import JacobiProcess as ProcessType


@pytest.mark.parametrize(
    "name, value, error_type",
    [
        ("speed", 0, ValueError),
        ("speed", "a", TypeError),
        ("mean", 1.5, ValueError),
        ("mean", -0.1, ValueError),
        ("vol", -1, ValueError),
        ("t", 0, ValueError),
    ],
)
def test_invalid_parameter(name: str, value: Any, error_type: type) -> None:
    with pytest.raises(error_type):
        ProcessType(**{name: value})


def test_str_repr() -> None:
    instance = ProcessType(speed=2.0, mean=0.4, vol=0.5, t=2.0)

    assert str(instance) == "Jacobi process with speed=2.0, mean=0.4, vol=0.5 on [0, 2.0]"
    assert repr(instance) == "JacobiProcess(speed=2.0, mean=0.4, vol=0.5, t=2.0)"


def test_parameters_are_validated_when_changed() -> None:
    instance = ProcessType(speed=2.0, mean=0.4, vol=0.5, t=2.0)

    value: Any = 0
    with pytest.raises(ValueError):
        instance.speed = value


@pytest.mark.parametrize("n", [1, 2, 100])
def test_sample_shape(n: int) -> None:
    samples = ProcessType(speed=2.0, mean=0.4, vol=0.5, t=2.0).sample(n)

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
        ProcessType(speed=2.0, mean=0.4, vol=0.5, t=2.0).sample(n)


def test_sample_values() -> None:
    instance = ProcessType(
        speed=2.0, mean=0.4, vol=0.5, t=2.0, rng=np.random.default_rng(1234567890)
    )

    np.testing.assert_allclose(
        instance.sample(5),
        [0.50000000, 0.09755407, 0.34122323, 0.27840503, 0.49038192, 0.51219373],
        rtol=1e-6,
    )


def test_sample_with_initial_values() -> None:
    instance = ProcessType(
        speed=2.0, mean=0.4, vol=0.5, t=2.0, rng=np.random.default_rng(1234567890)
    )
    samples = instance.sample_with_initial(5, initial=0.3)

    np.testing.assert_allclose(
        samples, [0.30000000, 0.08447342, 0.33849967, 0.27807364, 0.49027368, 0.51217168], rtol=1e-6
    )


def test_sample_starts_from_the_default_initial_value() -> None:
    instance = ProcessType(speed=2.0, mean=0.4, vol=0.5, t=2.0)

    assert instance.sample(3)[0] == 0.5


@pytest.mark.parametrize("initial", ["a", None])
def test_sample_with_initial_invalid(initial: Any) -> None:
    with pytest.raises(TypeError):
        ProcessType(speed=2.0, mean=0.4, vol=0.5, t=2.0).sample_with_initial(5, initial)


def test_sample_at_is_not_supported() -> None:
    with pytest.raises(NotImplementedError):
        ProcessType(speed=2.0, mean=0.4, vol=0.5, t=2.0).sample_at(np.array([0.0, 0.5, 1.0]))


def test_few_and_many_realizations_use_the_same_method(monkeypatch: pytest.MonkeyPatch) -> None:
    # Few realizations are advanced with Python floats, many with arrays.
    import samplepaths.diffusion.euler_maruyama_process as module

    paths = 2 * module._VECTORIZE_FROM_PATHS
    together = ProcessType(
        speed=2.0, mean=0.4, vol=0.5, t=2.0, rng=np.random.default_rng(3)
    ).sample_paths(50, paths)

    monkeypatch.setattr(module, "_VECTORIZE_FROM_PATHS", 10 * paths)
    in_turn = ProcessType(
        speed=2.0, mean=0.4, vol=0.5, t=2.0, rng=np.random.default_rng(3)
    ).sample_paths(50, paths)

    np.testing.assert_allclose(together, in_turn, rtol=1e-12, atol=1e-14)


def test_stays_in_the_unit_interval_and_reverts_to_the_mean() -> None:
    instance = ProcessType(speed=3.0, mean=0.7, vol=1.5, t=4.0, rng=np.random.default_rng(1))
    paths = instance.sample_paths_with_initial(400, 4000, initial=0.2)

    assert paths.min() >= 0.0
    assert paths.max() <= 1.0
    # The mean of the process is mean + (initial - mean) exp(-speed t).
    expected = 0.7 + (0.2 - 0.7) * np.exp(-3.0 * instance.times)
    np.testing.assert_allclose(paths.mean(axis=0), expected, atol=0.02)
