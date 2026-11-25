from typing import Any

import numpy as np
import pytest

from samplepaths.continuous import HawkesProcess as ProcessType
from samplepaths.continuous import PoissonProcess


@pytest.mark.parametrize(
    "kwargs, error_type",
    [
        ({"baseline": 0}, ValueError),
        ({"baseline": "a"}, TypeError),
        ({"excitation": -1}, ValueError),
        ({"decay": 0}, ValueError),
        # an arrival must trigger less than one arrival on average
        ({"excitation": 2.0, "decay": 2.0}, ValueError),
        ({"excitation": 3.0, "decay": 2.0}, ValueError),
    ],
)
def test_invalid_parameter(kwargs: dict[str, Any], error_type: type) -> None:
    with pytest.raises(error_type):
        ProcessType(**kwargs)


def test_parameters_are_validated_when_changed() -> None:
    instance = ProcessType(baseline=2.0, excitation=1.5, decay=3.0)

    with pytest.raises(ValueError):
        instance.excitation = 3.0
    with pytest.raises(ValueError):
        instance.decay = 1.0


def test_str_repr() -> None:
    instance = ProcessType(baseline=2.0, excitation=1.5, decay=3.0)

    assert str(instance) == "Hawkes process with baseline 2.0, excitation 1.5 and decay 3.0."
    assert repr(instance) == "HawkesProcess(baseline=2.0, excitation=1.5, decay=3.0)"


@pytest.mark.parametrize("n", [1, 2, 100])
def test_sample_shape(n: int) -> None:
    samples = ProcessType().sample(n)

    assert samples.shape == (n + 1,)
    assert samples[0] == 0.0
    assert np.all(np.diff(samples) > 0)


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
        ProcessType().sample(n)


def test_sample_values() -> None:
    instance = ProcessType(
        baseline=2.0, excitation=1.5, decay=3.0, rng=np.random.default_rng(1234567890)
    )

    np.testing.assert_allclose(
        instance.sample(5),
        [0.00000000, 0.94388138, 1.92999587, 3.07807137, 3.17390438, 4.35281293],
        rtol=1e-6,
    )


def test_sample_with_length() -> None:
    samples = ProcessType(rng=np.random.default_rng(1)).sample_with_length(20.0)

    # The arrivals start at zero and end with the first one after the length.
    assert samples[0] == 0.0
    assert samples[-1] > 20.0
    assert samples[-2] <= 20.0
    assert np.all(np.diff(samples) > 0)


@pytest.mark.parametrize("length", [0, -1.0])
def test_sample_with_length_invalid(length: float) -> None:
    with pytest.raises(ValueError):
        ProcessType().sample_with_length(length)


def test_long_run_rate() -> None:
    # The long-run rate of the arrivals is baseline / (1 - excitation / decay).
    instance = ProcessType(baseline=2.0, excitation=1.5, decay=3.0, rng=np.random.default_rng(1))
    counts = [len(instance.sample_with_length(50.0)) - 2 for _ in range(300)]

    np.testing.assert_allclose(np.mean(counts) / 50.0, 4.0, rtol=0.03)


def test_arrivals_are_more_clustered_than_those_of_a_poisson_process() -> None:
    # The number of arrivals in a window has a variance equal to its mean for
    # a Poisson process, and larger for a Hawkes process.
    hawkes = ProcessType(baseline=2.0, excitation=1.5, decay=3.0, rng=np.random.default_rng(1))
    counts = np.array([len(hawkes.sample_with_length(50.0)) - 2 for _ in range(300)])

    assert counts.var() > 2.0 * counts.mean()


def test_without_excitation_is_a_poisson_process() -> None:
    hawkes = ProcessType(baseline=2.0, excitation=0.0, rng=np.random.default_rng(1))
    poisson = PoissonProcess(rate=2.0, rng=np.random.default_rng(2))

    waits = np.diff(hawkes.sample(20_000))
    reference = np.diff(poisson.sample(20_000))

    np.testing.assert_allclose(waits.mean(), reference.mean(), rtol=0.03)
    np.testing.assert_allclose(waits.var(), reference.var(), rtol=0.06)
