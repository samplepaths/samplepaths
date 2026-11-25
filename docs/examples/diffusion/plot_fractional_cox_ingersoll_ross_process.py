"""
Fractional Cox-Ingersoll-Ross process
=====================================

A Cox-Ingersoll-Ross process driven by a fractional Brownian motion: it
reverts to a mean, stays nonnegative, and has a memory set by the Hurst
parameter. With a parameter above 0.5, used here, it strays from its mean
for longer.
"""

import matplotlib.pyplot as plt
import numpy as np

from samplepaths import FractionalCoxIngersollRossProcess

fig, axes = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
for ax, hurst in zip(axes, (0.5, 0.85), strict=True):
    process = FractionalCoxIngersollRossProcess(
        hurst=hurst, speed=2.0, mean=0.04, vol=0.2, t=5.0, rng=np.random.default_rng(128)
    )
    for _ in range(3):
        path = process.sample_with_initial(2500, initial=0.04)
        ax.plot(process.times, path, linewidth=0.7)
    ax.axhline(0.04, linestyle="--", color="gray")
    ax.set_ylabel(f"hurst = {hurst}")
axes[-1].set_xlabel("Time")
plt.show()

# %%
# A step which would make a value negative sets it to zero, so the values
# are never negative.

process = FractionalCoxIngersollRossProcess(
    hurst=0.7, speed=2.0, mean=0.04, vol=0.6, rng=np.random.default_rng(129)
)
paths = process.sample_paths_with_initial(256, paths=5000, initial=0.04)
print("smallest value:", paths.min())
