from typing import Any

import numpy as np
import pytest

from samplepaths.diffusion import FellerLogisticProcess as ProcessType


@pytest.mark.parametrize(
    "name, value, error_type",
    [
        ("speed", -1, ValueError),
        ("speed", "a", TypeError),
        ("level", -1, ValueError),
        ("vol", -1, ValueError),
        ("t", 0, ValueError),
    ],
)
def test_invalid_parameter(name: str, value: Any, error_type: type) -> None:
    with pytest.raises(error_type):
        ProcessType(**{name: value})


def test_str_repr() -> None:
    instance = ProcessType(speed=1.0, level=2.0, vol=0.3, t=2.0)

    assert str(instance) == "Feller logistic process with speed=1.0, level=2.0, vol=0.3 on [0, 2.0]"
    assert repr(instance) == "FellerLogisticProcess(speed=1.0, level=2.0, vol=0.3, t=2.0)"


def test_parameters_are_validated_when_changed() -> None:
    instance = ProcessType(speed=1.0, level=2.0, vol=0.3, t=2.0)

    value: Any = -1
    with pytest.raises(ValueError):
        instance.speed = value


@pytest.mark.parametrize("n", [1, 2, 100])
def test_sample_shape(n: int) -> None:
    samples = ProcessType(speed=1.0, level=2.0, vol=0.3, t=2.0).sample(n)

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
        ProcessType(speed=1.0, level=2.0, vol=0.3, t=2.0).sample(n)


def test_sample_values() -> None:
    instance = ProcessType(
        speed=1.0, level=2.0, vol=0.3, t=2.0, rng=np.random.default_rng(1234567890)
    )

    np.testing.assert_allclose(
        instance.sample(5),
        [1.00000000, 1.01306488, 1.41648194, 1.58166407, 2.03943427, 2.16858405],
        rtol=1e-6,
    )


def test_sample_with_initial_values() -> None:
    instance = ProcessType(
        speed=1.0, level=2.0, vol=0.3, t=2.0, rng=np.random.default_rng(1234567890)
    )
    samples = instance.sample_with_initial(5, initial=1.0)

    np.testing.assert_allclose(
        samples, [1.00000000, 1.01306488, 1.41648194, 1.58166407, 2.03943427, 2.16858405], rtol=1e-6
    )


def test_sample_starts_from_the_default_initial_value() -> None:
    instance = ProcessType(speed=1.0, level=2.0, vol=0.3, t=2.0)

    assert instance.sample(3)[0] == 1.0


@pytest.mark.parametrize("initial", ["a", None])
def test_sample_with_initial_invalid(initial: Any) -> None:
    with pytest.raises(TypeError):
        ProcessType(speed=1.0, level=2.0, vol=0.3, t=2.0).sample_with_initial(5, initial)


def test_sample_at_is_not_supported() -> None:
    with pytest.raises(NotImplementedError):
        ProcessType(speed=1.0, level=2.0, vol=0.3, t=2.0).sample_at(np.array([0.0, 0.5, 1.0]))


def test_few_and_many_realizations_use_the_same_method(monkeypatch: pytest.MonkeyPatch) -> None:
    # Few realizations are advanced with Python floats, many with arrays.
    import samplepaths.diffusion.euler_maruyama_process as module

    paths = 2 * module._VECTORIZE_FROM_PATHS
    together = ProcessType(
        speed=1.0, level=2.0, vol=0.3, t=2.0, rng=np.random.default_rng(3)
    ).sample_paths(50, paths)

    monkeypatch.setattr(module, "_VECTORIZE_FROM_PATHS", 10 * paths)
    in_turn = ProcessType(
        speed=1.0, level=2.0, vol=0.3, t=2.0, rng=np.random.default_rng(3)
    ).sample_paths(50, paths)

    np.testing.assert_allclose(together, in_turn, rtol=1e-12, atol=1e-14)


def test_grows_towards_the_level_without_noise() -> None:
    instance = ProcessType(speed=1.0, level=2.0, vol=0.0, t=10.0)
    samples = instance.sample_with_initial(2000, initial=0.1)

    assert np.all(np.diff(samples) > 0)
    np.testing.assert_allclose(samples[-1], 2.0, rtol=1e-3)


def test_zero_is_absorbing() -> None:
    instance = ProcessType(speed=0.1, level=1.0, vol=3.0, t=5.0, rng=np.random.default_rng(1))
    paths = instance.sample_paths_with_initial(500, 200, initial=0.2)

    assert paths.min() >= 0.0
    extinct = paths[paths[:, -1] == 0.0]
    assert len(extinct) > 0
    for path in extinct:
        first = np.flatnonzero(path == 0.0)[0]
        assert np.all(path[first:] == 0.0)
