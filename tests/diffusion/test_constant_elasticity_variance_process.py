from samplepaths.diffusion import (
    ConstantElasticityVarianceProcess as ProcessType,
)


def test_constant_elasticity_variance_process_str_repr() -> None:
    instance = ProcessType(drift=1.0, vol=0.5, volexp=0.5, t=2.0)

    assert "Constant elasticity of variance process with drift=" in str(instance)
    assert "ConstantElasticityVarianceProcess(drift=" in repr(instance)
