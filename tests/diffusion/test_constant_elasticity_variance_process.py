import numpy as np

from samplepaths.diffusion import (
    ConstantElasticityVarianceProcess as ProcessType,
)


def test_constant_elasticity_variance_process_str_repr() -> None:
    instance = ProcessType(drift=1.0, vol=0.5, volexp=0.5, t=2.0)

    assert "Constant elasticity of variance process with drift=" in str(instance)
    assert "ConstantElasticityVarianceProcess(drift=" in repr(instance)


def test_constant_elasticity_variance_process_drift_change_after_construction() -> None:
    instance = ProcessType(drift=0.1, vol=0.2, volexp=0.5, rng=np.random.default_rng(5))

    instance.drift = 0.4
    samples = instance.sample(n=50)

    fresh = ProcessType(drift=0.4, vol=0.2, volexp=0.5, rng=np.random.default_rng(5))
    np.testing.assert_array_equal(samples, fresh.sample(n=50))
    assert instance.drift(0.0) == 0.4
