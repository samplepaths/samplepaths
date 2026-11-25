from typing import Any

import numpy as np
import pytest

from samplepaths.multivariate import CorrelatedBrownianMotions as ProcessType

KWARGS: dict[str, Any] = {
    "correlation": [[1.0, 0.5, 0.2], [0.5, 1.0, -0.3], [0.2, -0.3, 1.0]],
    "t": 2.0,
}


@pytest.mark.parametrize(
    "kwargs, error_type",
    [
        ({"correlation": 1.5}, ValueError),
        ({"correlation": "a"}, TypeError),
        ({"correlation": [[1.0, 0.5]]}, ValueError),
        ({"correlation": [1.0, 0.5]}, ValueError),
        ({"correlation": [[1.0, 0.5], [0.2, 1.0]]}, ValueError),
        ({"correlation": [[2.0, 0.5], [0.5, 2.0]]}, ValueError),
        ({"correlation": [[1.0, 0.9, -0.9], [0.9, 1.0, 0.9], [-0.9, 0.9, 1.0]]}, ValueError),
        ({"correlation": [["a", "b"], ["c", "d"]]}, TypeError),
        ({"t": 0}, ValueError),
    ],
)
def test_invalid_parameters(kwargs: dict[str, Any], error_type: type) -> None:
    with pytest.raises(error_type):
        ProcessType(**{**KWARGS, **kwargs})


def test_str_repr() -> None:
    instance = ProcessType(**KWARGS)

    assert str(instance) == ("3 correlated Brownian motions on [0, 2.0]")
    assert repr(instance) == (
        "CorrelatedBrownianMotions(correlation=[[1.0, 0.5,"
        " 0.2], [0.5, 1.0, -0.3], [0.2, -0.3, 1.0]], t=2.0)"
    )


@pytest.mark.parametrize("n", [1, 2, 100])
def test_sample_shape(n: int) -> None:
    instance = ProcessType(**KWARGS)
    samples = instance.sample(n)

    assert instance.components == ("motion 1", "motion 2", "motion 3")
    assert samples.shape == (3, n + 1)
    assert instance.times.shape == (n + 1,)
    assert instance.sample_paths(n, 5).shape == (5, 3, n + 1)


@pytest.mark.parametrize("n, error_type", [(0, ValueError), (-1, ValueError), (1.5, TypeError)])
def test_sample_invalid_n(n: Any, error_type: type) -> None:
    with pytest.raises(error_type):
        ProcessType(**KWARGS).sample(n)


def test_sample_values() -> None:
    instance = ProcessType(**KWARGS, rng=np.random.default_rng(1234567890))
    samples = instance.sample(4)

    np.testing.assert_allclose(
        samples[0], [0.00000000, -0.83950293, -0.92858134, -1.72652789, -1.78859643], rtol=1e-6
    )
    np.testing.assert_allclose(
        samples[1], [0.00000000, 0.45345607, 0.49109755, -0.30703943, -0.50037253], rtol=1e-6
    )
    np.testing.assert_allclose(
        samples[2], [0.00000000, 0.01619953, -0.16961585, 0.52941546, 0.02122357], rtol=1e-6
    )


def test_sample_at_is_not_supported() -> None:
    with pytest.raises(NotImplementedError):
        ProcessType(**KWARGS).sample_at(np.array([0.0, 0.5, 1.0]))


def test_threads_give_reproducible_realizations() -> None:
    first = ProcessType(**KWARGS, rng=np.random.default_rng(5)).sample_paths(16, 20_000, workers=4)
    again = ProcessType(**KWARGS, rng=np.random.default_rng(5)).sample_paths(16, 20_000, workers=4)

    assert first.shape == (20_000, 3, 17)
    np.testing.assert_array_equal(first, again)


def test_a_number_gives_two_motions() -> None:
    instance = ProcessType(correlation=0.6)

    assert instance.components == ("motion 1", "motion 2")
    np.testing.assert_array_equal(instance.correlation, [[1.0, 0.6], [0.6, 1.0]])
    assert instance.sample(4).shape == (2, 5)
    assert (
        repr(instance) == "CorrelatedBrownianMotions(correlation=[[1.0, 0.6], [0.6, 1.0]], t=1.0)"
    )


def test_the_number_of_motions_follows_the_correlation() -> None:
    instance = ProcessType(correlation=0.6)
    instance.correlation = np.eye(4)

    assert len(instance.components) == 4
    assert instance.sample_paths(4, 3).shape == (3, 4, 5)


def test_every_motion_is_a_standard_brownian_motion() -> None:
    instance = ProcessType(**KWARGS, rng=np.random.default_rng(1))
    paths = instance.sample_paths(8, 200_000, workers=4)

    assert np.all(paths[:, :, 0] == 0)
    np.testing.assert_allclose(paths[:, :, -1].mean(axis=0), 0.0, atol=0.02)
    np.testing.assert_allclose(paths[:, :, -1].var(axis=0), 2.0, rtol=0.02)
    np.testing.assert_allclose(paths[:, :, 4].var(axis=0), 1.0, rtol=0.02)


def test_motions_have_the_given_correlation() -> None:
    instance = ProcessType(**KWARGS, rng=np.random.default_rng(1))
    paths = instance.sample_paths(8, 200_000, workers=4)

    np.testing.assert_allclose(np.corrcoef(paths[:, :, -1].T), KWARGS["correlation"], atol=0.01)
    increments = np.diff(paths, axis=2)
    np.testing.assert_allclose(np.corrcoef(increments[:, :, 3].T), KWARGS["correlation"], atol=0.01)
    # Increments over different steps are independent.
    assert abs(np.corrcoef(increments[:, 0, 2], increments[:, 1, 5])[0, 1]) < 0.01


@pytest.mark.parametrize("correlation", [1.0, -1.0])
def test_perfectly_correlated_motions(correlation: float) -> None:
    instance = ProcessType(correlation=correlation, rng=np.random.default_rng(1))
    first, second = instance.sample(50)

    np.testing.assert_allclose(second, correlation * first, atol=1e-12)
