from samplepaths.diffusion import VasicekProcess as ProcessType


def test_vasicek_process_str_repr() -> None:
    instance = ProcessType(speed=1.0, mean=0.5, vol=0.2, t=2.0)

    assert "Vasicek process with speed=" in str(instance)
    assert "VasicekProcess(speed=" in repr(instance)
