"""
Bessel process
==============

A Bessel process is the distance from the origin of a Brownian motion in
``dim`` dimensions. In one dimension it reaches zero; in three it does not.
"""

import matplotlib.pyplot as plt
import numpy as np

from samplepaths import BesselProcess

fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), sharey=True)
for ax, dim in zip(axes, (1, 3), strict=True):
    bessel = BesselProcess(dim=dim, t=2.0, rng=np.random.default_rng(5))
    for _ in range(4):
        path = bessel.sample(1000)
        ax.plot(bessel.times, path)
    ax.set_title(f"dim = {dim}")
    ax.set_xlabel("Time")
axes[0].set_ylabel("Distance")
plt.show()
