"""
Exponential GARCH process
=========================

A GARCH process for the logarithm of the variance. The variance is positive
whatever the signs of the coefficients, and ``gamma`` sets how the sign of a
value affects it.
"""

import matplotlib.pyplot as plt
import numpy as np

from samplepaths import EGARCHProcess

model = EGARCHProcess(
    omega=-0.1, alpha=0.25, gamma=-0.15, beta=0.95, rng=np.random.default_rng(160)
)
values = model.sample(2000)

fig, axes = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
axes[0].plot(values, linewidth=0.6)
axes[0].set_ylabel("Value")
axes[1].plot(model.volatility, color="C1")
axes[1].set_ylabel("Volatility")
axes[1].set_xlabel("Step")
plt.show()

# %%
# The logarithm of the variance is close to normal, so the volatility is
# close to lognormal.

model.sample_paths(500, paths=50_000)
log_variance = 2 * np.log(model.volatility[:, -1])
print("mean of the logarithm of the variance:", round(log_variance.mean(), 2))
print("expected, omega / (1 - beta):          ", round(-0.1 / (1 - 0.95), 2))

fig, ax = plt.subplots(figsize=(8, 3.5))
ax.hist(log_variance, bins=80, density=True)
ax.set_xlabel("Logarithm of the variance")
plt.show()
