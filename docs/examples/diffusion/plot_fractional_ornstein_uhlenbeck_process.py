"""
Fractional Ornstein-Uhlenbeck process
=====================================

An Ornstein-Uhlenbeck process driven by a fractional Brownian motion. With a
small Hurst parameter it is rough, and is the usual model of the logarithm
of a volatility.
"""

import matplotlib.pyplot as plt
import numpy as np

from samplepaths import FractionalOrnsteinUhlenbeckProcess

fig, axes = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
for ax, hurst in zip(axes, (0.1, 0.8), strict=True):
    process = FractionalOrnsteinUhlenbeckProcess(
        hurst=hurst, speed=1.0, vol=0.5, t=4.0, rng=np.random.default_rng(124)
    )
    path = process.sample_with_initial(2000, initial=0.0)
    ax.plot(process.times, path, linewidth=0.7)
    ax.set_ylabel(f"hurst = {hurst}")
axes[-1].set_xlabel("Time")
plt.show()

# %%
# A rough volatility: the exponential of a rough process.

process = FractionalOrnsteinUhlenbeckProcess(
    hurst=0.1, speed=0.5, vol=0.6, t=1.0, rng=np.random.default_rng(125)
)
volatility = 0.2 * np.exp(process.sample_with_initial(1000, initial=0.0))

fig, ax = plt.subplots(figsize=(8, 3.5))
ax.plot(process.times, volatility, linewidth=0.7)
ax.set_xlabel("Time")
ax.set_ylabel("Volatility")
plt.show()
