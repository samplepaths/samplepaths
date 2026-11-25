"""
Ho-Lee process
==============

A model of the short interest rate without reversion to a mean: a drift,
which can change over time, plus a Brownian motion.
"""

import matplotlib.pyplot as plt
import numpy as np

from samplepaths import HoLeeProcess


def drift(time: float) -> float:
    """Rates expected to rise for three years, then to stay."""
    return 0.01 if time < 3.0 else 0.0


rates = HoLeeProcess(drift=drift, vol=0.008, t=10.0, rng=np.random.default_rng(123))
paths = rates.sample_paths_with_initial(500, paths=5000, initial=0.02)

fig, ax = plt.subplots(figsize=(8, 4))
low, high = np.quantile(paths, [0.05, 0.95], axis=0)
ax.fill_between(rates.times, low, high, alpha=0.3, label="90% of the realizations")
for path in paths[:4]:
    ax.plot(rates.times, path, linewidth=0.8)
ax.plot(rates.times, paths.mean(axis=0), linestyle="--", color="gray", label="mean")
ax.set_xlabel("Years")
ax.set_ylabel("Rate")
ax.legend()
plt.show()

# %%
# Nothing pulls the rate back, so its spread keeps growing and the rate can
# become negative.

print("share of negative rates after 10 years:", np.mean(paths[:, -1] < 0))
