"""
Fractional Vasicek process
==========================

A Vasicek process driven by a fractional Brownian motion. With a Hurst
parameter above 0.5 a rate which has risen tends to keep rising, so it
strays from its mean for longer.
"""

import matplotlib.pyplot as plt
import numpy as np

from samplepaths import FractionalVasicekProcess

fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharey=True)
for ax, hurst in zip(axes, (0.5, 0.85), strict=True):
    process = FractionalVasicekProcess(
        hurst=hurst, speed=1.0, mean=0.03, vol=0.01, t=10.0, rng=np.random.default_rng(126)
    )
    for _ in range(4):
        path = process.sample_with_initial(2000, initial=0.03)
        ax.plot(process.times, path, linewidth=0.8)
    ax.axhline(0.03, linestyle="--", color="gray")
    ax.set_title(f"hurst = {hurst}")
    ax.set_xlabel("Years")
axes[0].set_ylabel("Rate")
plt.show()

# %%
# The long memory widens the range of the rates over the ten years.

for hurst in (0.5, 0.85):
    process = FractionalVasicekProcess(
        hurst=hurst, speed=1.0, mean=0.03, vol=0.01, t=10.0, rng=np.random.default_rng(127)
    )
    final = process.sample_paths_with_initial(512, paths=20_000, initial=0.03)[:, -1]
    print(f"hurst {hurst}: standard deviation of the final rate {final.std():.4f}")
