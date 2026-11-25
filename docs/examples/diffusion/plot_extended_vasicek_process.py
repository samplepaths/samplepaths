"""
Extended Vasicek process
========================

A Vasicek process whose parameters are functions of time. With a mean which
changes over time it is the Hull-White model of interest rates, which can
follow any path expected for the rates.
"""

import matplotlib.pyplot as plt
import numpy as np

from samplepaths import ExtendedVasicekProcess


def expected_rate(time: float) -> float:
    """Rates expected to rise from 2% to 5% over the first five years."""
    return 0.02 + 0.03 * min(time, 5.0) / 5.0


process = ExtendedVasicekProcess(
    speed=1.5, mean=expected_rate, vol=0.01, t=10.0, rng=np.random.default_rng(114)
)
paths = process.sample_paths_with_initial(500, paths=2000, initial=0.02)

fig, ax = plt.subplots(figsize=(8, 4))
low, high = np.quantile(paths, [0.05, 0.95], axis=0)
ax.fill_between(process.times, low, high, alpha=0.3, label="90% of the realizations")
for path in paths[:4]:
    ax.plot(process.times, path, linewidth=0.8)
ax.plot(
    process.times,
    [expected_rate(time) for time in process.times],
    linestyle="--",
    color="gray",
    label="mean parameter",
)
ax.set_xlabel("Years")
ax.set_ylabel("Rate")
ax.legend()
plt.show()
