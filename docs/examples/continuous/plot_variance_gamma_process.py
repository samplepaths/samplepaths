"""
Variance gamma process
======================

A variance gamma process is a Brownian motion with drift run on a random
clock given by a gamma process. A larger ``variance`` of the clock gives
larger jumps.
"""

import matplotlib.pyplot as plt
import numpy as np

from samplepaths import VarianceGammaProcess

fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), sharey=True)
for ax, variance in zip(axes, (0.05, 1.0), strict=True):
    process = VarianceGammaProcess(
        drift=0.0,
        variance=variance,
        scale=1.0,
        rng=np.random.default_rng(10),
    )
    for _ in range(3):
        path = process.sample(1000)
        ax.plot(process.times, path)
    ax.set_title(f"variance = {variance}")
    ax.set_xlabel("Time")
axes[0].set_ylabel("Value")
plt.show()
