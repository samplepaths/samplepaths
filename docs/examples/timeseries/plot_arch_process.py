"""
ARCH process
============

The variance of a value depends on the squares of the previous values, so a
large value is followed by a large variance. The values themselves are
uncorrelated.
"""

import matplotlib.pyplot as plt
import numpy as np

from samplepaths import ARCHProcess

model = ARCHProcess(omega=0.2, alpha=[0.5, 0.3], rng=np.random.default_rng(156))
values = model.sample(1000)

fig, axes = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
axes[0].plot(values, linewidth=0.6)
axes[0].set_ylabel("Value")
axes[1].plot(model.volatility, color="C1")
axes[1].set_ylabel("Volatility")
axes[1].set_xlabel("Step")
plt.show()

# %%
# The values are uncorrelated, but their squares are not.

paths = model.sample_paths(200, paths=50_000)
lags = np.arange(1, 9)
of_values = [np.corrcoef(paths[:, -1], paths[:, -1 - lag])[0, 1] for lag in lags]
of_squares = [np.corrcoef(paths[:, -1] ** 2, paths[:, -1 - lag] ** 2)[0, 1] for lag in lags]

fig, ax = plt.subplots(figsize=(8, 3.5))
ax.bar(lags - 0.2, of_values, width=0.4, label="values")
ax.bar(lags + 0.2, of_squares, width=0.4, label="squared values")
ax.set_xlabel("Lag")
ax.set_ylabel("Correlation")
ax.legend()
plt.show()
