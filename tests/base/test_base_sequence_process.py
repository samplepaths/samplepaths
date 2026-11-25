import pytest

from samplepaths.base import BaseSequenceProcess


def test_base_sequence_process_abstract() -> None:
    with pytest.raises(TypeError):
        _ = BaseSequenceProcess()  # type: ignore[abstract]
