"""
Memory and roughness
====================

A fractional Ornstein-Uhlenbeck process is driven by a fractional Brownian
motion. Its Hurst parameter decides how the path looks: rough below 0.5,
smooth and persistent above.
"""

import matplotlib.pyplot as plt
import numpy as np

from samplepaths import FractionalOrnsteinUhlenbeckProcess

hursts = (0.1, 0.5, 0.9)

fig, axes = plt.subplots(len(hursts), 1, figsize=(8, 6), sharex=True)
for ax, hurst in zip(axes, hursts, strict=True):
    process = FractionalOrnsteinUhlenbeckProcess(
        hurst=hurst, speed=2.0, vol=1.0, t=4.0, rng=np.random.default_rng(41)
    )
    ax.plot(np.linspace(0, 4, 1025), process.sample_with_initial(1024, initial=0.0), linewidth=0.8)
    ax.set_ylabel(f"hurst = {hurst}")
axes[-1].set_xlabel("Time")
plt.show()

# %%
# The increments of the process over a short time step tell the three apart.
# Their spread grows as the step to the power of the Hurst parameter, so on a
# logarithmic scale the three processes give lines with these slopes. This is
# how the roughness of a volatility is measured from data.

lags = np.array([1, 2, 4, 8, 16, 32])

fig, ax = plt.subplots(figsize=(8, 4))
for hurst in hursts:
    process = FractionalOrnsteinUhlenbeckProcess(
        hurst=hurst, speed=0.1, vol=1.0, t=1.0, rng=np.random.default_rng(42)
    )
    paths = process.sample_paths_with_initial(1024, paths=2000, initial=0.0)
    spreads = [np.std(paths[:, lag:] - paths[:, :-lag]) for lag in lags]
    slope = np.polyfit(np.log(lags), np.log(spreads), 1)[0]
    ax.loglog(lags / 1024, spreads, marker="o", label=f"hurst = {hurst}, slope {slope:.2f}")
ax.set_xlabel("Time step")
ax.set_ylabel("Spread of the increments")
ax.legend()
plt.show()
