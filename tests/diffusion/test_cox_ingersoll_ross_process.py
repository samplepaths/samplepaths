import math
from typing import Any

import numpy as np
import pytest

from samplepaths.diffusion import CoxIngersollRossProcess as ProcessType


def test_cox_ingersoll_ross_process_str_repr() -> None:
    instance = ProcessType(speed=1.0, mean=0.5, vol=0.2, t=2.0)

    assert "Cox-Ingersoll-Ross process with speed=" in str(instance)
    assert "CoxIngersollRossProcess(speed=" in repr(instance)


def test_sample_exact_shape() -> None:
    instance = ProcessType(speed=1.0, mean=1.0, vol=0.3)

    realization = instance.sample_exact(10, initial=1.5)
    assert realization.shape == (11,)
    assert realization[0] == 1.5
    assert instance.times.shape == (11,)
    paths = instance.sample_paths_exact(10, 40, initial=1.5, workers=2)
    assert paths.shape == (40, 11)
    assert instance.n == 10
    assert np.all(paths[:, 0] == 1.5)


@pytest.mark.parametrize("n", [1, 4, 64])
def test_sample_exact_has_the_exact_distribution_whatever_the_step(n: int) -> None:
    from scipy.stats import kstest, ncx2

    instance = ProcessType(speed=1.0, mean=1.0, vol=0.3, rng=np.random.default_rng(1))
    final = instance.sample_paths_exact(n, 50_000, initial=1.5)[:, -1]

    scale = 0.3**2 * (1 - math.exp(-1.0)) / 4
    exact = ncx2(4 / 0.3**2, 1.5 * math.exp(-1.0) / scale, scale=scale)
    assert kstest(final, exact.cdf).pvalue > 1e-3


def test_sample_exact_for_one_and_for_many_realizations_agree() -> None:
    kwargs = {"speed": 2.0, "mean": 0.5, "vol": 0.6}
    one = ProcessType(**kwargs, rng=np.random.default_rng(1))
    single = np.array([one.sample_exact(4)[-1] for _ in range(20_000)])
    many = ProcessType(**kwargs, rng=np.random.default_rng(2)).sample_paths_exact(4, 20_000)[:, -1]

    np.testing.assert_allclose(single.mean(), many.mean(), rtol=0.02)
    np.testing.assert_allclose(single.std(), many.std(), rtol=0.03)


def test_sample_exact_with_a_mean_of_zero_can_reach_zero_and_stay() -> None:
    instance = ProcessType(speed=2.0, vol=0.5, rng=np.random.default_rng(1))
    paths = instance.sample_paths_exact(16, 100_000)

    np.testing.assert_allclose(paths[:, -1].mean(), math.exp(-2.0), rtol=0.02)
    assert paths.min() == 0.0
    absorbed = paths[paths[:, 8] == 0]
    assert len(absorbed) > 0
    assert np.all(absorbed[:, 8:] == 0)


def test_sample_exact_without_volatility_follows_the_mean() -> None:
    instance = ProcessType(speed=2.0, mean=0.5, vol=0.0)

    np.testing.assert_allclose(instance.sample_exact(4), 0.5 + 0.5 * np.exp(-2.0 * instance.times))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"speed": lambda t: 1.0},
        {"mean": lambda t: 1.0},
        {"vol": lambda t: 1.0},
        {"speed": 0.0},
        {"speed": -1.0},
        {"mean": -1.0},
        {"vol": -1.0},
    ],
)
def test_sample_exact_needs_constant_parameters_in_range(kwargs: Any) -> None:
    instance = ProcessType(**{"speed": 1.0, "mean": 1.0, "vol": 0.3, **kwargs})

    with pytest.raises(ValueError):
        instance.sample_exact(4)
    with pytest.raises(ValueError):
        instance.sample_paths_exact(4, 3)


def test_sample_exact_invalid_arguments() -> None:
    instance = ProcessType(speed=1.0, mean=1.0, vol=0.3)

    with pytest.raises(ValueError):
        instance.sample_exact(0)
    with pytest.raises(ValueError):
        instance.sample_exact(4, initial=-1.0)
    with pytest.raises(ValueError):
        instance.sample_paths_exact(4, 0)
    with pytest.raises(ValueError):
        instance.sample_paths_exact(4, 3, workers=0)


def test_sample_exact_with_threads_is_reproducible_and_leaves_the_euler_method_alone() -> None:
    kwargs = {"speed": 1.0, "mean": 1.0, "vol": 0.3}
    first = ProcessType(**kwargs, rng=np.random.default_rng(5))
    again = ProcessType(**kwargs, rng=np.random.default_rng(5))

    np.testing.assert_array_equal(
        first.sample_paths_exact(8, 20_000, workers=4),
        again.sample_paths_exact(8, 20_000, workers=4),
    )
    assert not first._exact
    assert first.sample_paths(8, 5).shape == (5, 9)


def test_sample_at_has_the_exact_distribution() -> None:
    from scipy.stats import kstest, ncx2

    instance = ProcessType(speed=1.0, mean=1.0, vol=0.3, rng=np.random.default_rng(1))
    final = np.array(
        [instance.sample_at_with_initial([0.2, 0.7, 1.0], 1.5)[-1] for _ in range(20_000)]
    )

    scale = 0.3**2 * (1 - math.exp(-1.0)) / 4
    exact = ncx2(4 / 0.3**2, 1.5 * math.exp(-1.0) / scale, scale=scale)
    assert kstest(final, exact.cdf).pvalue > 1e-3


def test_sample_at_shape_and_arguments() -> None:
    instance = ProcessType(speed=1.0, mean=1.0, vol=0.3)

    with_zero = instance.sample_at(np.array([0.0, 0.5, 2.0]))
    assert with_zero.shape == (3,)
    assert with_zero[0] == 1.0
    assert instance.sample_at([0.5, 2.0]).shape == (2,)
    assert instance.sample_at_with_initial([0.0, 1.0], initial=0.0)[0] == 0.0
    for times in ([], [1.0, 0.5], [-1.0, 1.0]):
        with pytest.raises(ValueError):
            instance.sample_at(times)
    with pytest.raises(ValueError):
        instance.sample_at_with_initial([0.0, 1.0], initial=-1.0)
    with pytest.raises(ValueError):
        ProcessType(speed=0.0, mean=1.0, vol=0.3).sample_at([0.0, 1.0])
    with pytest.raises(NotImplementedError):
        kwargs: dict[str, Any] = {"speed": 1.0, "mean": lambda t: 1.0, "vol": 0.3}
        ProcessType(**kwargs).sample_at([0.0, 1.0])


def test_sample_at_without_volatility_and_with_a_mean_of_zero() -> None:
    times = np.array([0.0, 0.3, 1.0, 4.0])
    still = ProcessType(speed=2.0, mean=0.5, vol=0.0)
    np.testing.assert_allclose(still.sample_at(times), 0.5 + 0.5 * np.exp(-2.0 * times))

    falling = ProcessType(speed=2.0, vol=0.5, rng=np.random.default_rng(1))
    final = np.array([falling.sample_at(times)[-1] for _ in range(2000)])
    assert final.min() == 0.0
