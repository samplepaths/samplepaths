"""Compare the results of samplepaths and of stochastic-rs.

The two packages use different random number generators, so the same seed
does not give the same realization. What can be compared is the distribution
of the results. For every process which both packages provide, this script
generates many realizations with each and compares a summary of each
realization (its final value, for most processes) with a two-sample
Kolmogorov-Smirnov test::

    uv run --with stochastic-rs python docs/scripts/compare_rust_outputs.py

A p-value which is not small means that the test finds no difference between
the two distributions. With 43 comparisons, one p-value below 0.05 is
expected by chance.
"""

import warnings

import numpy as np
from _rust.models import add_model_cases
from _rust.processes import add_process_cases
from _rust.series import add_series_cases
from _rust.shared import Array
from scipy.stats import ks_2samp


def cases(rng: np.random.Generator, count: int) -> list[tuple[str, str, Array, Array]]:
    """Return, for every shared process, a summary from each package.

    :returns: the name of the process, what is compared, the values of
        stochastic-rs and the values of samplepaths
    """
    results: list[tuple[str, str, Array, Array]] = []

    def add(name: str, what: str, theirs: Array, ours: Array) -> None:
        results.append((name, what, np.asarray(theirs, dtype=float), np.asarray(ours, dtype=float)))

    # The jump sizes which stochastic-rs asks for come from one generator.
    jump_rng = np.random.default_rng(1)

    add_process_cases(add, rng, count, jump_rng)
    add_model_cases(add, rng, count, jump_rng)
    add_series_cases(add, rng, count)
    return results


def main() -> None:
    """Run the comparison and print the results."""
    warnings.simplefilter("ignore")
    rng = np.random.default_rng(2026)

    print(
        f"{'process':28s} {'compared':20s} {'rust mean':>10s} {'ours mean':>10s} "
        f"{'rust std':>9s} {'ours std':>9s} {'p-value':>8s}"
    )
    for name, what, theirs, ours in cases(rng, count=40_000):
        p_value = ks_2samp(theirs, ours).pvalue
        print(
            f"{name:28s} {what:20s} {theirs.mean():10.4f} {ours.mean():10.4f} "
            f"{theirs.std():9.4f} {ours.std():9.4f} {p_value:8.3f}"
        )


if __name__ == "__main__":
    main()
