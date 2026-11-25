"""
Geometric Brownian motion
=========================

A geometric Brownian motion is the exponential of a Brownian motion with
drift. It stays positive, and its mean grows at the rate ``drift``.
"""

import matplotlib.pyplot as plt
import numpy as np

from samplepaths import GeometricBrownianMotion

gbm = GeometricBrownianMotion(drift=0.08, volatility=0.25, t=5.0, rng=np.random.default_rng(102))

fig, ax = plt.subplots(figsize=(8, 4))
for _ in range(6):
    path = gbm.sample_with_initial(1000, initial=100.0)
    ax.plot(gbm.times, path)
ax.plot(gbm.times, 100.0 * np.exp(0.08 * gbm.times), linestyle="--", color="gray", label="mean")
ax.set_xlabel("Time")
ax.set_ylabel("Price")
ax.legend()
plt.show()

# %%
# The final values have a lognormal distribution: their logarithm is normal.

final = gbm.sample_paths_with_initial(64, paths=50_000, initial=100.0)[:, -1]
print("mean:", round(final.mean(), 1), "expected:", round(100.0 * np.exp(0.08 * 5.0), 1))

fig, ax = plt.subplots(figsize=(8, 3.5))
ax.hist(np.log(final), bins=80, density=True)
ax.set_xlabel("Logarithm of the final price")
plt.show()
