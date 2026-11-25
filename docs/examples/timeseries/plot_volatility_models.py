"""
Volatility clustering and the leverage effect
=============================================

In a GARCH process, a large value raises the variance of the values which
follow, so calm and agitated periods alternate, as in the returns of
financial assets. After sampling, ``volatility`` holds the standard
deviation of every value.
"""

import matplotlib.pyplot as plt
import numpy as np

from samplepaths import AGARCHProcess, EGARCHProcess, GARCHProcess, TGARCHProcess

garch = GARCHProcess(omega=0.05, alpha=0.15, beta=0.8, rng=np.random.default_rng(8))
returns = garch.sample(1000)

fig, axes = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
axes[0].plot(returns, linewidth=0.6)
axes[0].set_ylabel("Value")
axes[1].plot(garch.volatility, color="C1")
axes[1].set_ylabel("Volatility")
axes[1].set_xlabel("Step")
plt.show()

# %%
# Stock markets are more agitated after a fall than after a rise of the same
# size. The threshold, asymmetric and exponential models reproduce this, each
# in its own way. The curves show the mean volatility which follows a value,
# estimated from 20,000 realizations.

models = {
    "GARCH": GARCHProcess(omega=0.1, alpha=0.1, beta=0.8),
    "threshold GARCH": TGARCHProcess(omega=0.1, alpha=0.03, gamma=0.14, beta=0.8),
    "asymmetric GARCH": AGARCHProcess(omega=0.05, alpha=0.1, shift=0.7, beta=0.8),
    "exponential GARCH": EGARCHProcess(omega=0.0, alpha=0.25, gamma=-0.15, beta=0.9),
}
edges = np.linspace(-2.5, 2.5, 21)
centers = (edges[:-1] + edges[1:]) / 2

fig, ax = plt.subplots(figsize=(8, 4))
for name, model in models.items():
    model.rng = np.random.default_rng(9)
    paths = model.sample_paths(200, paths=20_000)
    previous = paths[:, -2] / paths[:, -2].std()
    following = model.volatility[:, -1]
    bins = np.digitize(previous, edges)
    means = [following[bins == index].mean() for index in range(1, len(edges))]
    ax.plot(centers, means / following.mean(), label=name)
ax.set_xlabel("Previous value, in standard deviations")
ax.set_ylabel("Volatility, relative to its mean")
ax.legend()
plt.show()
