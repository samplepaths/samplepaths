"""
Threshold GARCH process
=======================

A GARCH process in which a negative value raises the variance more than a
positive value of the same size: the squares of negative values get the
additional weight ``gamma``.
"""

import matplotlib.pyplot as plt
import numpy as np

from samplepaths import TGARCHProcess

model = TGARCHProcess(omega=0.02, alpha=0.03, gamma=0.12, beta=0.88, rng=np.random.default_rng(158))
values = model.sample(2000)

fig, axes = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
axes[0].plot(values, linewidth=0.6)
axes[0].set_ylabel("Value")
axes[1].plot(model.volatility, color="C1")
axes[1].set_ylabel("Volatility")
axes[1].set_xlabel("Step")
plt.show()

# %%
# The mean volatility which follows a value, against that value: a fall is
# followed by more volatility than a rise.

paths = model.sample_paths(300, paths=100_000)
previous = paths[:, -2]
following = model.volatility[:, -1]
edges = np.linspace(-2.5, 2.5, 21)
bins = np.digitize(previous / previous.std(), edges)
means = [following[bins == index].mean() for index in range(1, len(edges))]

fig, ax = plt.subplots(figsize=(8, 3.5))
ax.plot((edges[:-1] + edges[1:]) / 2, means, marker="o")
ax.set_xlabel("Previous value, in standard deviations")
ax.set_ylabel("Mean volatility")
plt.show()
