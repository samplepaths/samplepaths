"""Measure the time of the sampling methods of every process.

Run from the repository root::

    make benchmark

or directly, optionally saving the results or comparing with saved ones::

    uv run python docs/scripts/benchmark.py --save results.json
    uv run python docs/scripts/benchmark.py --compare results.json
"""

import argparse
import json
import time
import warnings
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np

import samplepaths
from samplepaths.base import BaseProcess, BaseTimeProcess

#: Number of increments of a realization.
STEPS = 4096

#: Smaller numbers of increments for the algorithms with a quadratic cost.
QUADRATIC_STEPS = {"MultifractionalBrownianMotion": 512, "sample_hosking": 1024}

#: Arguments of the processes without default parameters.
REQUIRED_KWARGS: dict[str, dict[str, Any]] = {
    "GammaProcess": {"mean": 1.0, "variance": 1.0},
    "MoranProcess": {"maximum": 2000},
}


def make(name: str) -> BaseProcess:
    """Create a process with a seeded random number generator."""
    rng = np.random.default_rng(0)
    kwargs = dict(REQUIRED_KWARGS.get(name, {}))
    if name == "MixedPoissonProcess":
        kwargs = {"rate_func": rng.uniform, "rate_args": (1, 5)}
    process: BaseProcess = getattr(samplepaths, name)(rng=rng, **kwargs)
    return process


def measure(function: Callable[[], object], minimum_time: float = 0.2) -> float:
    """Return the mean time of a call in milliseconds."""
    function()  # warm up, including the caches of the process
    calls = 0
    start = time.perf_counter()
    while True:
        function()
        calls += 1
        elapsed = time.perf_counter() - start
        if elapsed > minimum_time and calls >= 3:
            return elapsed / calls * 1e3


def run() -> dict[str, float]:
    """Measure every sampling method and return the times in milliseconds."""
    results: dict[str, float] = {}
    names = [name for name in samplepaths.__all__ if isinstance(getattr(samplepaths, name), type)]

    for name in names:
        n = QUADRATIC_STEPS.get(name, STEPS)
        process = make(name)
        results[f"{name}.sample({n})"] = measure(partial(process.sample, n))

        if isinstance(process, BaseTimeProcess):
            times = np.linspace(0, 1, n + 1)
            try:
                process.sample_at(times)
            except NotImplementedError:
                continue
            results[f"{name}.sample_at({n + 1} times)"] = measure(partial(process.sample_at, times))

    n = QUADRATIC_STEPS["sample_hosking"]
    noise = samplepaths.FractionalGaussianNoise(hurst=0.7, rng=np.random.default_rng(0))
    results[f"FractionalGaussianNoise.sample_hosking({n})"] = measure(
        lambda: noise.sample_hosking(n)
    )
    results[f"FractionalGaussianNoise.sample_davies_harte({STEPS})"] = measure(
        lambda: noise.sample_davies_harte(STEPS)
    )
    restaurant = samplepaths.ChineseRestaurantProcess(rng=np.random.default_rng(0))
    results[f"ChineseRestaurantProcess.sample_partition({STEPS})"] = measure(
        lambda: restaurant.sample_partition(STEPS)
    )

    return results


def main() -> None:
    """Run the benchmark and print the results."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--save", type=Path, help="write the results to a JSON file")
    parser.add_argument("--compare", type=Path, help="compare with a saved JSON file")
    args = parser.parse_args()

    warnings.simplefilter("ignore")
    results = run()

    if args.save:
        args.save.write_text(json.dumps(results, indent=1) + "\n")

    if args.compare:
        before = json.loads(args.compare.read_text())
        print(f"{'before ms':>10} {'now ms':>10} {'speedup':>9}  case")
        for case in sorted(results, key=lambda c: before.get(c, 0) / results[c], reverse=True):
            if case in before:
                ratio = before[case] / results[case]
                print(f"{before[case]:10.3f} {results[case]:10.3f} {ratio:8.1f}x  {case}")
    else:
        print(f"{'ms':>10}  case")
        for case in sorted(results, key=lambda c: results[c], reverse=True):
            print(f"{results[case]:10.3f}  {case}")


if __name__ == "__main__":
    main()
