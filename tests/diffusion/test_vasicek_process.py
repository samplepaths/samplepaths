import math

import numpy as np
import pytest

from samplepaths.diffusion import VasicekProcess as ProcessType


def test_vasicek_process_str_repr() -> None:
    instance = ProcessType(speed=1.0, mean=0.5, vol=0.2, t=2.0)

    assert "Vasicek process with speed=" in str(instance)
    assert "VasicekProcess(speed=" in repr(instance)


def test_sample_exact_shape() -> None:
    instance = ProcessType(speed=2.0, mean=0.5, vol=0.4)

    realization = instance.sample_exact(10, initial=1.5)
    assert realization.shape == (11,)
    assert realization[0] == 1.5
    assert instance.times.shape == (11,)
    paths = instance.sample_paths_exact(10, 40, initial=1.5, workers=2)
    assert paths.shape == (40, 11)
    assert np.all(paths[:, 0] == 1.5)


@pytest.mark.parametrize("n", [1, 4, 64])
def test_sample_exact_has_the_exact_distribution_whatever_the_step(n: int) -> None:
    from scipy.stats import kstest, norm

    instance = ProcessType(speed=2.0, mean=0.5, vol=0.4, t=3.0, rng=np.random.default_rng(1))
    final = instance.sample_paths_exact(n, 50_000, initial=1.5, workers=2)[:, -1]

    speed, mean, vol = 2.0, 0.5, 0.4
    expected_mean = mean + (1.5 - mean) * math.exp(-speed * 3.0)
    expected_deviation = vol * math.sqrt((1 - math.exp(-2 * speed * 3.0)) / (2 * speed))
    assert kstest(final, norm(expected_mean, expected_deviation).cdf).pvalue > 1e-3


def test_sample_at_has_the_exact_distribution() -> None:
    from scipy.stats import kstest, norm

    instance = ProcessType(speed=2.0, mean=0.5, vol=0.4, rng=np.random.default_rng(1))
    values = np.array(
        [instance.sample_at_with_initial([0.0, 0.1, 0.5, 3.0], 1.5) for _ in range(20_000)]
    )

    speed, mean, vol = 2.0, 0.5, 0.4
    assert np.all(values[:, 0] == 1.5)
    expected_mean = mean + (1.5 - mean) * math.exp(-speed * 3.0)
    expected_deviation = vol * math.sqrt((1 - math.exp(-2 * speed * 3.0)) / (2 * speed))
    assert kstest(values[:, -1], norm(expected_mean, expected_deviation).cdf).pvalue > 1e-3
    # Values at two times are correlated as those of the exact process.
    correlation = math.exp(-speed * 0.4) * math.sqrt(
        (1 - math.exp(-2 * speed * 0.1)) / (1 - math.exp(-2 * speed * 0.5))
    )
    np.testing.assert_allclose(
        np.corrcoef(values[:, 1], values[:, 2])[0, 1], correlation, atol=0.03
    )


def test_sample_at_shape_and_times() -> None:
    instance = ProcessType(speed=2.0, mean=0.5, vol=0.4)

    assert instance.sample_at(np.array([0.0, 0.5, 2.0])).shape == (3,)
    assert instance.sample_at([0.0, 0.5, 2.0])[0] == 1.0
    assert instance.sample_at([0.5, 2.0]).shape == (2,)
    for times in ([], [1.0, 0.5], [-1.0, 1.0]):
        with pytest.raises(ValueError):
            instance.sample_at(times)
    with pytest.raises(TypeError):
        instance.sample_at_with_initial([0.0, 1.0], initial="a")  # type: ignore[arg-type]


def test_sample_exact_invalid_arguments() -> None:
    instance = ProcessType(speed=2.0, mean=0.5, vol=0.4)

    with pytest.raises(ValueError):
        instance.sample_exact(0)
    with pytest.raises(ValueError):
        instance.sample_paths_exact(4, 0)
    with pytest.raises(TypeError):
        instance.sample_exact(4, initial="a")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        ProcessType(speed=0.0, mean=0.5, vol=0.4).sample_exact(4)


def test_sample_exact_leaves_the_euler_method_alone() -> None:
    first = ProcessType(speed=2.0, mean=0.5, vol=0.4, rng=np.random.default_rng(5))
    again = ProcessType(speed=2.0, mean=0.5, vol=0.4, rng=np.random.default_rng(5))

    np.testing.assert_array_equal(
        first.sample_paths_exact(8, 20_000, workers=4),
        again.sample_paths_exact(8, 20_000, workers=4),
    )
    np.testing.assert_array_equal(first.sample(8), again.sample(8))
