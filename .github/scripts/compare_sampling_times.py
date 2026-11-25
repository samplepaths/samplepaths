#!/usr/bin/env python3
"""Measure the sampling times of two checkouts of the repository and compare them.

Both checkouts are measured in one run, so on the same machine::

    python .github/scripts/compare_sampling_times.py BASE_DIRECTORY HEAD_DIRECTORY

The script fails when a sampling method of the head is slower than in the
base by more than ``--max-slowdown`` times. In a GitHub Actions job the
comparison is added to the job summary.

It needs ``uv`` and the standard library only.
"""

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def benchmark(directory: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    """Run the benchmark of a checkout with only the package installed.

    ``uv`` is called directly instead of ``make``: the base checkout may not
    have the make targets of the head.

    :param directory: the root of the checkout
    :param arguments: the arguments of ``docs/scripts/benchmark.py``
    :returns: the finished benchmark, with its output
    """
    subprocess.run(["uv", "sync", "--locked", "--no-default-groups"], cwd=directory, check=True)
    return subprocess.run(
        ["uv", "run", "--no-sync", "python", "docs/scripts/benchmark.py", *arguments],
        cwd=directory,
        stdout=subprocess.PIPE,
        text=True,
        check=False,
    )


def write_summary(comparison: str) -> None:
    """Add the comparison to the summary of a GitHub Actions job, if any."""
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if not summary:
        return

    with Path(summary).open("a", encoding="utf-8") as file:
        file.write("### Sampling times: base branch and pull request (milliseconds)\n\n")
        file.write(f"```\n{comparison}```\n")


def main() -> int:
    """Measure both checkouts and return the exit status of the comparison."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("base", type=Path, help="the checkout to compare with")
    parser.add_argument("head", type=Path, help="the checkout to check")
    parser.add_argument(
        "--max-slowdown",
        type=float,
        default=2.0,
        help="fail when a method is slower than in the base by this factor (default: 2)",
    )
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as results:
        saved = str(Path(results) / "base.json")

        print(f"Measuring the base: {args.base.resolve()}", flush=True)
        base = benchmark(args.base.resolve(), "--save", saved)
        if base.returncode != 0:
            print(base.stdout)
            return base.returncode

        print(f"Measuring the head: {args.head.resolve()}", flush=True)
        head = benchmark(
            args.head.resolve(), "--compare", saved, "--max-slowdown", f"{args.max_slowdown:g}"
        )

    print(head.stdout)
    write_summary(head.stdout)
    return head.returncode


if __name__ == "__main__":
    sys.exit(main())
