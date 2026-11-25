"""
Mixed Poisson process
=====================

A Poisson process whose rate is itself random: it is drawn anew for every
realization. The number of arrivals then varies more than that of a Poisson
process with the same mean rate.
"""

import matplotlib.pyplot as plt
import numpy as np

from samplepaths import MixedPoissonProcess, PoissonProcess

rng = np.random.default_rng(107)
mixed = MixedPoissonProcess(rate_func=rng.uniform, rate_args=(1.0, 5.0), rng=rng)

fig, ax = plt.subplots(figsize=(8, 4))
for _ in range(6):
    arrivals = mixed.sample_with_length(10.0)
    ax.step(arrivals, np.arange(len(arrivals)), where="post")
ax.set_xlim(0, 10)
ax.set_xlabel("Time")
ax.set_ylabel("Arrivals")
plt.show()

# %%
# The rate is uniform between 1 and 5, with a mean of 3. Compare the number
# of arrivals in 10 units of time with that of a Poisson process of rate 3.

poisson = PoissonProcess(rate=3.0, rng=rng)
mixed_counts = [len(mixed.sample_with_length(10.0)) - 1 for _ in range(5000)]
poisson_counts = [len(poisson.sample_with_length(10.0)) - 1 for _ in range(5000)]

fig, ax = plt.subplots(figsize=(8, 3.5))
bins = np.arange(0, 71, 2)
ax.hist(mixed_counts, bins=bins, histtype="step", label="mixed Poisson process")
ax.hist(poisson_counts, bins=bins, histtype="step", label="Poisson process")
ax.set_xlabel("Arrivals in 10 units of time")
ax.legend()
plt.show()
