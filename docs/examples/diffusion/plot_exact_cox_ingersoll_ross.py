"""
Exact sampling of a Cox-Ingersoll-Ross process
==============================================

The Euler-Maruyama method approximates the process over each time step. The
exact method draws every value from its true distribution, so its result
does not depend on the number of steps.
"""

import math

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ncx2

from samplepaths import CoxIngersollRossProcess

speed, mean, vol, initial = 1.0, 0.5, 0.6, 0.5
process = CoxIngersollRossProcess(speed=speed, mean=mean, vol=vol, rng=np.random.default_rng(51))

# The exact distribution of the value at time 1.
scale = vol**2 * (1 - math.exp(-speed)) / (4 * speed)
exact = ncx2(4 * speed * mean / vol**2, initial * math.exp(-speed) / scale, scale=scale)
grid = np.linspace(0.0, 2.0, 400)

fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharey=True)
for ax, steps in zip(axes, (4, 64), strict=True):
    # A path of the Euler-Maruyama method which steps below zero has no
    # square root to continue with, and ends as nan.
    with np.errstate(invalid="ignore"):
        euler = process.sample_paths_with_initial(steps, paths=100_000, initial=initial)[:, -1]
    print(f"{steps} steps: {np.isnan(euler).mean():.1%} of the Euler-Maruyama paths end as nan")
    drawn = process.sample_paths_exact(steps, paths=100_000, initial=initial)[:, -1]
    bins = np.linspace(0.0, 2.0, 81)
    ax.hist(
        euler[np.isfinite(euler)], bins=bins, density=True, histtype="step", label="Euler-Maruyama"
    )
    ax.hist(drawn, bins=bins, density=True, histtype="step", label="exact method")
    ax.plot(
        grid, exact.pdf(grid), color="gray", linewidth=0.8, linestyle="--", label="true density"
    )
    ax.set_title(f"{steps} steps")
    ax.set_xlabel("Value at time 1")
axes[0].legend()
plt.show()

# %%
# With a larger volatility the process reaches zero now and then. The exact
# method stays there or leaves again.

process.vol = 1.5
path = process.sample_exact(1000, initial=0.05)

fig, ax = plt.subplots(figsize=(8, 3.5))
ax.plot(process.times, path)
ax.set_xlabel("Time")
ax.set_ylabel("Value")
plt.show()
