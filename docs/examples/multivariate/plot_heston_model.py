"""
Random volatility and the shape of returns
==========================================

In a Heston model the variance of a price is random and reverts to a mean.
A realization has two components, the price and the variance.
"""

import matplotlib.pyplot as plt
import numpy as np

from samplepaths import GeometricBrownianMotion, HestonModel

heston = HestonModel(
    drift=0.05, speed=3.0, mean=0.04, vol=0.5, correlation=-0.7, rng=np.random.default_rng(21)
)
price, variance = heston.sample(1000)

fig, axes = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
axes[0].plot(heston.times, price)
axes[0].set_ylabel("Price")
axes[1].plot(heston.times, np.sqrt(variance), color="C1")
axes[1].set_ylabel("Volatility")
axes[1].set_xlabel("Time")
plt.show()

# %%
# With a negative correlation the volatility rises when the price falls, so
# large falls are more likely than large rises: the distribution of the
# returns is skewed, unlike that of a geometric Brownian motion with the same
# mean variance.

returns = np.log(heston.sample_paths(64, paths=200_000)[:, 0, -1])
reference = GeometricBrownianMotion(drift=0.05, volatility=0.2, rng=np.random.default_rng(22))
lognormal = np.log(reference.sample_paths(64, paths=200_000)[:, -1])

fig, ax = plt.subplots(figsize=(8, 4))
bins = np.linspace(-1.0, 0.8, 91)
ax.hist(returns, bins=bins, density=True, histtype="step", label="Heston model")
ax.hist(lognormal, bins=bins, density=True, histtype="step", label="geometric Brownian motion")
ax.set_yscale("log")
ax.set_xlabel("Logarithm of the final price")
ax.legend()
plt.show()
