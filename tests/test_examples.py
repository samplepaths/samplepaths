"""Tests of the examples of the gallery."""

import inspect
from pathlib import Path

import pytest

import samplepaths

EXAMPLES = Path(__file__).parent.parent / "docs" / "examples"

PROCESSES = [name for name in samplepaths.__all__ if isinstance(getattr(samplepaths, name), type)]


@pytest.mark.parametrize("name", PROCESSES)
def test_every_process_has_an_example(name: str) -> None:
    # The example of a process is named after its module and lies in the
    # folder of its family: docs/examples/<family>/plot_<module>.py.
    module = inspect.getmodule(getattr(samplepaths, name))
    assert module is not None
    family, stem = module.__name__.split(".")[1], module.__name__.split(".")[-1]

    example = EXAMPLES / family / f"plot_{stem}.py"
    assert example.is_file(), f"{name} has no example; expected {example}"
    assert name in example.read_text()
