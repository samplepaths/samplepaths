"""
A basket of correlated assets
=============================

Correlated Brownian motions drive the prices of assets which tend to move
together. The exponential of each motion, with a volatility and a correction
for the drift, is a geometric Brownian motion.
"""

import matplotlib.pyplot as plt
import numpy as np

from samplepaths import CorrelatedBrownianMotions

correlation = np.array(
    [
        [1.0, 0.8, 0.3],
        [0.8, 1.0, 0.1],
        [0.3, 0.1, 1.0],
    ]
)
volatility = np.array([[0.2], [0.25], [0.4]])
motions = CorrelatedBrownianMotions(correlation=correlation, rng=np.random.default_rng(23))

prices = 100 * np.exp(
    -(volatility**2) / 2 * np.linspace(0, 1, 253) + volatility * motions.sample(252)
)

fig, ax = plt.subplots(figsize=(8, 4))
for index, asset in enumerate(prices):
    ax.plot(motions.times, asset, label=f"asset {index + 1}")
ax.set_xlabel("Time")
ax.set_ylabel("Price")
ax.legend()
plt.show()

# %%
# Correlation matters for a basket: the value of an equal mix of the three
# assets is less spread out when they are independent.

fig, ax = plt.subplots(figsize=(8, 4))
for label, matrix in (("correlated", correlation), ("independent", np.eye(3))):
    motions = CorrelatedBrownianMotions(correlation=matrix, rng=np.random.default_rng(24))
    final = motions.sample_paths(16, paths=100_000)[:, :, -1]
    basket = (100 * np.exp(-(volatility.T**2) / 2 + volatility.T * final)).mean(axis=1)
    ax.hist(basket, bins=np.linspace(40, 200, 81), density=True, histtype="step", label=label)
    print(f"{label}: standard deviation of the basket {basket.std():.1f}")
ax.set_xlabel("Value of the basket after one year")
ax.legend()
plt.show()
