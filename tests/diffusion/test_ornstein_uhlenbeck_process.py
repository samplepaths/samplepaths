import numpy as np

from samplepaths.diffusion import OrnsteinUhlenbeckProcess as ProcessType


def test_ornstein_uhlenbeck_process_str_repr() -> None:
    instance = ProcessType(speed=1.0, vol=0.5, t=2.0)

    assert "Ornstein-Uhlenbeck process with speed=" in str(instance)
    assert "OrnsteinUhlenbeckProcess(speed=" in repr(instance)


def test_exact_methods_revert_to_zero() -> None:
    instance = ProcessType(speed=2.0, vol=0.5, rng=np.random.default_rng(1))

    assert instance.sample_exact(4, initial=3.0)[0] == 3.0
    assert instance.sample_at(np.array([0.0, 0.5, 1.0])).shape == (3,)
    final = instance.sample_paths_exact(1, 100_000, initial=3.0)[:, -1]
    np.testing.assert_allclose(final.mean(), 3.0 * np.exp(-2.0), atol=0.01)
    np.testing.assert_allclose(final.var(), 0.25 * (1 - np.exp(-4.0)) / 4.0, rtol=0.03)
