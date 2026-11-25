"""
Asymmetric GARCH process
========================

A GARCH process in which the previous value is shifted before it is squared.
The variance is then lowest after a value equal to the shift, instead of
zero.
"""

import matplotlib.pyplot as plt
import numpy as np

from samplepaths import AGARCHProcess

model = AGARCHProcess(omega=0.02, alpha=0.08, shift=0.8, beta=0.85, rng=np.random.default_rng(159))
values = model.sample(2000)

fig, axes = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
axes[0].plot(values, linewidth=0.6)
axes[0].set_ylabel("Value")
axes[1].plot(model.volatility, color="C1")
axes[1].set_ylabel("Volatility")
axes[1].set_xlabel("Step")
plt.show()

# %%
# The mean volatility which follows a value is lowest to the right of zero,
# near the shift.

paths = model.sample_paths(300, paths=100_000)
previous = paths[:, -2]
following = model.volatility[:, -1]
edges = np.linspace(-3.0, 3.0, 25)
bins = np.digitize(previous, edges)
means = [following[bins == index].mean() for index in range(1, len(edges))]

fig, ax = plt.subplots(figsize=(8, 3.5))
ax.plot((edges[:-1] + edges[1:]) / 2, means, marker="o")
ax.axvline(0.8, linestyle="--", color="gray", label="shift")
ax.set_xlabel("Previous value")
ax.set_ylabel("Mean volatility")
ax.legend()
plt.show()
