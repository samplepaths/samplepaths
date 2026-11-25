"""
Normal inverse Gaussian process
===============================

A process with independent increments which have a normal inverse Gaussian
distribution: heavier tails than the normal distribution, and a skew set by
``beta``. It moves by jumps only.
"""

import matplotlib.pyplot as plt
import numpy as np

from samplepaths import NormalInverseGaussianProcess

nig = NormalInverseGaussianProcess(
    alpha=8.0, beta=-3.0, delta=0.5, t=2.0, rng=np.random.default_rng(108)
)

fig, ax = plt.subplots(figsize=(8, 4))
for _ in range(5):
    path = nig.sample(2000)
    ax.plot(nig.times, path, linewidth=0.9)
ax.set_xlabel("Time")
plt.show()

# %%
# The increments over a short time are far from normal. The dashed line is
# the normal density with the same mean and variance.

increments = np.diff(nig.sample_paths(20, paths=20_000), axis=1).ravel()
grid = np.linspace(
    increments.mean() - 6 * increments.std(), increments.mean() + 6 * increments.std(), 300
)
normal = np.exp(-((grid - increments.mean()) ** 2) / (2 * increments.var())) / np.sqrt(
    2 * np.pi * increments.var()
)

fig, ax = plt.subplots(figsize=(8, 3.5))
ax.hist(increments, bins=grid[::3], density=True)
ax.plot(grid, normal, linestyle="--", color="gray")
ax.set_yscale("log")
ax.set_ylim(1e-3, None)
ax.set_xlabel("Increment over a time step of 0.1")
plt.show()
