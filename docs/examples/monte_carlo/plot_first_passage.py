"""
First passage times
===================

Whole paths make path-dependent quantities easy. This example estimates how
long a Brownian motion with drift takes to reach a level, and compares the
mean with the exact value ``level / drift``.
"""

import matplotlib.pyplot as plt
import numpy as np

from samplepaths import BrownianMotion

drift, level = 1.0, 2.0

bm = BrownianMotion(drift=drift, scale=1.0, t=12.0, rng=np.random.default_rng(27))
paths = bm.sample_paths(n=6000, paths=4000)

reached = paths >= level
first = np.where(reached.any(axis=1), bm.times[reached.argmax(axis=1)], np.nan)
print("share of realizations which reach the level:", np.mean(~np.isnan(first)))
print("mean passage time:", round(float(np.nanmean(first)), 3), "exact:", level / drift)

fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(first[~np.isnan(first)], bins=60, density=True)
ax.axvline(level / drift, linestyle="--", color="gray", label="exact mean")
ax.set_xlabel("First time at which the level is reached")
ax.set_ylabel("Density")
ax.legend()
plt.show()
