from typing import Any

import numpy as np
import pytest

from samplepaths.diffusion import KimuraProcess as ProcessType


@pytest.mark.parametrize(
    "name, value, error_type",
    [
        ("selection", "a", TypeError),
        ("vol", -1, ValueError),
        ("vol", "a", TypeError),
        ("t", 0, ValueError),
    ],
)
def test_invalid_parameter(name: str, value: Any, error_type: type) -> None:
    with pytest.raises(error_type):
        ProcessType(**{name: value})


def test_str_repr() -> None:
    instance = ProcessType(selection=1.0, vol=0.5, t=2.0)

    assert str(instance) == "Kimura process with selection=1.0, vol=0.5 on [0, 2.0]"
    assert repr(instance) == "KimuraProcess(selection=1.0, vol=0.5, t=2.0)"


def test_parameters_are_validated_when_changed() -> None:
    instance = ProcessType(selection=1.0, vol=0.5, t=2.0)

    value: Any = "a"
    with pytest.raises(TypeError):
        instance.selection = value


@pytest.mark.parametrize("n", [1, 2, 100])
def test_sample_shape(n: int) -> None:
    samples = ProcessType(selection=1.0, vol=0.5, t=2.0).sample(n)

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
        ProcessType(selection=1.0, vol=0.5, t=2.0).sample(n)


def test_sample_values() -> None:
    instance = ProcessType(selection=1.0, vol=0.5, t=2.0, rng=np.random.default_rng(1234567890))

    np.testing.assert_allclose(
        instance.sample(5),
        [0.50000000, 0.27755407, 0.36034554, 0.34131908, 0.55258595, 0.74509253],
        rtol=1e-6,
    )


def test_sample_with_initial_values() -> None:
    instance = ProcessType(selection=1.0, vol=0.5, t=2.0, rng=np.random.default_rng(1234567890))
    samples = instance.sample_with_initial(5, initial=0.3)

    np.testing.assert_allclose(
        samples, [0.30000000, 0.08847342, 0.12237073, 0.08940766, 0.19499140, 0.33237057], rtol=1e-6
    )


def test_sample_starts_from_the_default_initial_value() -> None:
    instance = ProcessType(selection=1.0, vol=0.5, t=2.0)

    assert instance.sample(3)[0] == 0.5


@pytest.mark.parametrize("initial", ["a", None])
def test_sample_with_initial_invalid(initial: Any) -> None:
    with pytest.raises(TypeError):
        ProcessType(selection=1.0, vol=0.5, t=2.0).sample_with_initial(5, initial)


def test_sample_at_is_not_supported() -> None:
    with pytest.raises(NotImplementedError):
        ProcessType(selection=1.0, vol=0.5, t=2.0).sample_at(np.array([0.0, 0.5, 1.0]))


def test_few_and_many_realizations_use_the_same_method(monkeypatch: pytest.MonkeyPatch) -> None:
    # Few realizations are advanced with Python floats, many with arrays.
    import samplepaths.diffusion.euler_maruyama_process as module

    paths = 2 * module._VECTORIZE_FROM_PATHS
    together = ProcessType(
        selection=1.0, vol=0.5, t=2.0, rng=np.random.default_rng(3)
    ).sample_paths(50, paths)

    monkeypatch.setattr(module, "_VECTORIZE_FROM_PATHS", 10 * paths)
    in_turn = ProcessType(selection=1.0, vol=0.5, t=2.0, rng=np.random.default_rng(3)).sample_paths(
        50, paths
    )

    np.testing.assert_allclose(together, in_turn, rtol=1e-12, atol=1e-14)


def test_stays_in_the_unit_interval_and_is_a_martingale_without_selection() -> None:
    instance = ProcessType(selection=0.0, vol=1.0, t=2.0, rng=np.random.default_rng(1))
    paths = instance.sample_paths_with_initial(400, 20000, initial=0.3)

    assert paths.min() >= 0.0
    assert paths.max() <= 1.0
    np.testing.assert_allclose(paths.mean(axis=0), 0.3, atol=0.01)


def test_boundaries_are_absorbing() -> None:
    instance = ProcessType(selection=0.0, vol=3.0, t=5.0, rng=np.random.default_rng(1))
    paths = instance.sample_paths_with_initial(500, 200, initial=0.5)

    # Once lost or fixed, an allele stays lost or fixed.
    for path in paths:
        absorbed = np.flatnonzero((path == 0.0) | (path == 1.0))
        if len(absorbed):
            assert np.all(path[absorbed[0] :] == path[absorbed[0]])


def test_selection_raises_the_frequency() -> None:
    kwargs: dict[str, Any] = {"vol": 0.2, "t": 2.0}
    favored = ProcessType(selection=2.0, rng=np.random.default_rng(1), **kwargs)
    neutral = ProcessType(selection=0.0, rng=np.random.default_rng(1), **kwargs)

    assert favored.sample_paths(100, 2000)[:, -1].mean() > 0.6
    np.testing.assert_allclose(neutral.sample_paths(100, 2000)[:, -1].mean(), 0.5, atol=0.02)
