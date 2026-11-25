"""
Fractional geometric Brownian motion
====================================

The exponential of a fractional Brownian motion with a trend. The Hurst
parameter decides whether rises tend to follow rises, above 0.5, or falls,
below 0.5. The mean grows at the rate ``drift`` in every case.
"""

import matplotlib.pyplot as plt
import numpy as np

from samplepaths import FractionalGeometricBrownianMotion

fig, axes = plt.subplots(1, 3, figsize=(10, 3.5), sharey=True)
for ax, hurst in zip(axes, (0.3, 0.5, 0.8), strict=True):
    process = FractionalGeometricBrownianMotion(
        hurst=hurst, drift=0.05, volatility=0.3, rng=np.random.default_rng(103)
    )
    for _ in range(4):
        path = process.sample_with_initial(1000, initial=100.0)
        ax.plot(process.times, path)
    ax.set_title(f"hurst = {hurst}")
    ax.set_xlabel("Time")
axes[0].set_ylabel("Price")
plt.show()

# %%
# The mean does not depend on the Hurst parameter.

for hurst in (0.3, 0.5, 0.8):
    process = FractionalGeometricBrownianMotion(
        hurst=hurst, drift=0.05, volatility=0.3, rng=np.random.default_rng(104)
    )
    final = process.sample_paths_with_initial(32, paths=100_000, initial=100.0)[:, -1]
    print(f"hurst {hurst}: mean {final.mean():.2f}, expected {100 * np.exp(0.05):.2f}")
