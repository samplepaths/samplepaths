from samplepaths.continuous import WienerProcess as ProcessType


def test_wiener_str_repr() -> None:
    instance = ProcessType(t=1.0)

    assert str(instance) == "Wiener process on [0, 1.0]"

    assert repr(instance) == "WienerProcess(t=1.0)"

    instance_standard = ProcessType()

    assert str(instance_standard) == "Wiener process on [0, 1.0]"
    assert repr(instance_standard) == "WienerProcess(t=1.0)"
