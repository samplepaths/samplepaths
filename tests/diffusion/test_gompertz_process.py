from typing import Any

import numpy as np
import pytest

from samplepaths.diffusion import GompertzProcess as ProcessType


@pytest.mark.parametrize(
    "name, value, error_type",
    [
        ("decay", -1, ValueError),
        ("decay", "a", TypeError),
        ("growth", "a", TypeError),
        ("vol", -1, ValueError),
        ("t", 0, ValueError),
    ],
)
def test_invalid_parameter(name: str, value: Any, error_type: type) -> None:
    with pytest.raises(error_type):
        ProcessType(**{name: value})


def test_str_repr() -> None:
    instance = ProcessType(growth=1.0, decay=0.5, vol=0.2, t=2.0)

    assert str(instance) == "Gompertz process with growth=1.0, decay=0.5, vol=0.2 on [0, 2.0]"
    assert repr(instance) == "GompertzProcess(growth=1.0, decay=0.5, vol=0.2, t=2.0)"


def test_parameters_are_validated_when_changed() -> None:
    instance = ProcessType(growth=1.0, decay=0.5, vol=0.2, t=2.0)

    value: Any = -1
    with pytest.raises(ValueError):
        instance.decay = value


@pytest.mark.parametrize("n", [1, 2, 100])
def test_sample_shape(n: int) -> None:
    samples = ProcessType(growth=1.0, decay=0.5, vol=0.2, t=2.0).sample(n)

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
        ProcessType(growth=1.0, decay=0.5, vol=0.2, t=2.0).sample(n)


def test_sample_values() -> None:
    instance = ProcessType(
        growth=1.0, decay=0.5, vol=0.2, t=2.0, rng=np.random.default_rng(1234567890)
    )

    np.testing.assert_allclose(
        instance.sample(5),
        [1.00000000, 1.14204326, 1.57115999, 1.91205299, 2.62472708, 3.36572014],
        rtol=1e-6,
    )


def test_sample_with_initial_values() -> None:
    instance = ProcessType(
        growth=1.0, decay=0.5, vol=0.2, t=2.0, rng=np.random.default_rng(1234567890)
    )
    samples = instance.sample_with_initial(5, initial=2.0)

    np.testing.assert_allclose(
        samples, [2.00000000, 2.00682764, 2.53461827, 2.84212633, 3.67615448, 4.46628640], rtol=1e-6
    )


def test_sample_starts_from_the_default_initial_value() -> None:
    instance = ProcessType(growth=1.0, decay=0.5, vol=0.2, t=2.0)

    assert instance.sample(3)[0] == 1.0


@pytest.mark.parametrize("initial", ["a", None])
def test_sample_with_initial_invalid(initial: Any) -> None:
    with pytest.raises(TypeError):
        ProcessType(growth=1.0, decay=0.5, vol=0.2, t=2.0).sample_with_initial(5, initial)


def test_sample_at_is_not_supported() -> None:
    with pytest.raises(NotImplementedError):
        ProcessType(growth=1.0, decay=0.5, vol=0.2, t=2.0).sample_at(np.array([0.0, 0.5, 1.0]))


def test_few_and_many_realizations_use_the_same_method(monkeypatch: pytest.MonkeyPatch) -> None:
    # Few realizations are advanced with Python floats, many with arrays.
    import samplepaths.diffusion.euler_maruyama_process as module

    paths = 2 * module._VECTORIZE_FROM_PATHS
    together = ProcessType(
        growth=1.0, decay=0.5, vol=0.2, t=2.0, rng=np.random.default_rng(3)
    ).sample_paths(50, paths)

    monkeypatch.setattr(module, "_VECTORIZE_FROM_PATHS", 10 * paths)
    in_turn = ProcessType(
        growth=1.0, decay=0.5, vol=0.2, t=2.0, rng=np.random.default_rng(3)
    ).sample_paths(50, paths)

    np.testing.assert_allclose(together, in_turn, rtol=1e-12, atol=1e-14)


def test_logarithm_is_an_ornstein_uhlenbeck_process() -> None:
    growth, decay, vol, t = 1.0, 0.5, 0.2, 4.0
    instance = ProcessType(growth=growth, decay=decay, vol=vol, t=t, rng=np.random.default_rng(1))
    logs = np.log(instance.sample_paths_with_initial(800, 20000, initial=2.0)[:, -1])

    # d ln X = (growth - vol^2 / 2 - decay ln X) dt + vol dW
    level = (growth - vol**2 / 2) / decay
    mean = level + (np.log(2.0) - level) * np.exp(-decay * t)
    variance = vol**2 / (2 * decay) * (1 - np.exp(-2 * decay * t))
    np.testing.assert_allclose(logs.mean(), mean, atol=0.01)
    np.testing.assert_allclose(logs.var(), variance, rtol=0.05)


def test_stays_positive() -> None:
    instance = ProcessType(growth=-5.0, decay=0.0, vol=3.0, t=2.0, rng=np.random.default_rng(1))

    assert instance.sample_paths(100, 2000).min() > 0.0
