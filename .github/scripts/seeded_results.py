#!/usr/bin/env python3
"""Record seeded results of every process, or compare them with a record.

Realizations of ``sample()`` and ``sample_at()`` for a given seed must not
change from one version to the next. This script records them with one
version and compares them with another::

    python .github/scripts/seeded_results.py record results.npz
    python .github/scripts/seeded_results.py compare results.npz

``make check-seeded`` records with a released version installed from PyPI
and compares with the working tree. The processes and their parameters are
the cases of ``tests/test_sample_paths.py``; a process which the recording
version does not have is left out.

The methods with their own interface, ``sample_exact()`` and the variants
with an initial value, are recorded too, as are small batches of
``sample_paths()``. A difference there is reported but does not fail the
comparison, because only ``sample()`` and ``sample_at()`` are promised.
"""

import argparse
import importlib.util
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np


def _load_cases() -> Any:
    """Load the module of the tests which lists the processes to sample."""
    path = Path(__file__).resolve().parents[2] / "tests" / "test_sample_paths.py"
    spec = importlib.util.spec_from_file_location("cases_of_the_tests", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_TESTS = _load_cases()

#: Times for sample_at, unequally spaced and starting after zero.
TIMES = np.array([0.1, 0.25, 0.7, 1.0, 2.5])

#: The methods whose seeded results are promised to users.
PROMISED = ("sample", "sample_at")

#: Other methods, with their arguments, recorded to notice changes.
OTHERS: dict[str, tuple[Any, ...]] = {
    "sample_with_initial": (24, 0.7),
    "sample_exact": (24,),
    "sample_paths": (24, 5),
    "sample_paths_many": (24, 40),
}


def results() -> dict[str, Any]:
    """Return the seeded results of every case, by a name of the case."""
    recorded: dict[str, Any] = {}

    def record(key: str, name: str, kwargs: dict[str, Any], method: str, *args: Any) -> None:
        try:
            process = _TESTS.make(name, kwargs, seed=11)
            recorded[key] = np.asarray(getattr(process, method)(*args))
        except (AttributeError, NotImplementedError, TypeError, ValueError):
            # The process or the method does not exist in this version, or
            # the method does not apply to these parameters.
            pass

    for index, (name, kwargs) in enumerate(_TESTS.CASES):
        case = f"{name}[{index}]"
        for n in (1, 2, 37, 256):
            record(f"{case}.sample({n})", name, kwargs, "sample", n)
        record(f"{case}.sample_at", name, kwargs, "sample_at", TIMES)
        for label, args in OTHERS.items():
            method = "sample_paths" if label.startswith("sample_paths") else label
            record(f"{case}.{label}", name, kwargs, method, *args)

    return recorded


def compare(before: dict[str, Any], now: dict[str, Any]) -> int:
    """Print the differences between two records and return the exit status."""
    shared = sorted(set(before) & set(now))
    failures: list[str] = []
    notices: list[str] = []
    for key in shared:
        old, new = before[key], now[key]
        if old.shape == new.shape and np.array_equal(old, new, equal_nan=True):
            continue
        promised = key.split(".", 1)[1].split("(")[0] in PROMISED
        (failures if promised else notices).append(key)

    print(f"{len(shared)} seeded results compared, {len(failures) + len(notices)} differ.")
    for key in notices:
        print(f"  changed, not promised to users: {key}")
    for key in failures:
        print(f"  CHANGED: {key}")
    if failures:
        print(
            "\nResults of sample() or sample_at() changed for a given seed. Keep them, or "
            "say in the changelog which process changes and why."
        )
    return 1 if failures else 0


def main() -> int:
    """Record or compare, and return the exit status."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("action", choices=("record", "compare"))
    parser.add_argument("file", type=Path, help="the file of recorded results")
    args = parser.parse_args()

    warnings.simplefilter("ignore")
    if args.action == "record":
        recorded = results()
        np.savez_compressed(args.file, **recorded)
        print(f"{len(recorded)} seeded results recorded in {args.file}.")
        return 0

    with np.load(args.file) as archive:
        before = {key: archive[key] for key in archive.files}
    return compare(before, results())


if __name__ == "__main__":
    sys.exit(main())
