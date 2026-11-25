import numpy as np
import pytest

from samplepaths.diffusion import ExtendedVasicekProcess as ProcessType


def test_extended_vasicek_process_str_repr() -> None:
    instance = ProcessType(speed=1.0, mean=0.5, vol=0.2, t=2.0)

    assert "Extended Vasicek process with speed=" in str(instance)
    assert "ExtendedVasicekProcess(speed=" in repr(instance)


def test_exact_methods_need_constant_parameters() -> None:
    instance = ProcessType(speed=1.0, mean=lambda t: t, vol=0.3)

    with pytest.raises(ValueError):
        instance.sample_exact(4)
    with pytest.raises(ValueError):
        instance.sample_paths_exact(4, 3)
    with pytest.raises(NotImplementedError):
        instance.sample_at(np.array([0.0, 1.0]))
    # The Euler-Maruyama method handles them.
    assert instance.sample(4).shape == (5,)


def test_exact_methods_with_constant_parameters() -> None:
    instance = ProcessType(speed=1.0, mean=0.5, vol=0.3, rng=np.random.default_rng(1))

    assert instance.sample_exact(4).shape == (5,)
    assert instance.sample_at(np.array([0.5, 1.0])).shape == (2,)
    final = instance.sample_paths_exact(2, 50_000, initial=2.0)[:, -1]
    np.testing.assert_allclose(final.mean(), 0.5 + 1.5 * np.exp(-1.0), atol=0.01)
