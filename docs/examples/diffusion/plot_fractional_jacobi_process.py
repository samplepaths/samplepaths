"""
Fractional Jacobi process
=========================

A Jacobi process driven by a fractional Brownian motion: a proportion, or a
correlation, between 0 and 1 with a memory set by the Hurst parameter.
"""

import matplotlib.pyplot as plt
import numpy as np

from samplepaths import FractionalJacobiProcess

fig, axes = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
for ax, hurst in zip(axes, (0.5, 0.85), strict=True):
    process = FractionalJacobiProcess(
        hurst=hurst, speed=1.5, mean=0.6, vol=0.5, t=5.0, rng=np.random.default_rng(130)
    )
    for _ in range(3):
        path = process.sample_with_initial(2500, initial=0.2)
        ax.plot(process.times, path, linewidth=0.7)
    ax.axhline(0.6, linestyle="--", color="gray")
    ax.set_ylim(0, 1)
    ax.set_ylabel(f"hurst = {hurst}")
axes[-1].set_xlabel("Time")
plt.show()

# %%
# A correlation which changes over time: map the process from the unit
# interval to the interval from -1 to 1.

process = FractionalJacobiProcess(
    hurst=0.8, speed=1.0, mean=0.75, vol=0.4, t=5.0, rng=np.random.default_rng(131)
)
correlation = 2 * process.sample_with_initial(1000, initial=0.5) - 1

fig, ax = plt.subplots(figsize=(8, 3.5))
ax.plot(process.times, correlation)
ax.set_ylim(-1, 1)
ax.set_xlabel("Time")
ax.set_ylabel("Correlation")
plt.show()
