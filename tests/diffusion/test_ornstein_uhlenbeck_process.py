from samplepaths.diffusion import OrnsteinUhlenbeckProcess as ProcessType


def test_ornstein_uhlenbeck_process_str_repr() -> None:
    instance = ProcessType(speed=1.0, vol=0.5, t=2.0)

    assert "Ornstein-Uhlenbeck process with speed=" in str(instance)
    assert "OrnsteinUhlenbeckProcess(speed=" in repr(instance)
