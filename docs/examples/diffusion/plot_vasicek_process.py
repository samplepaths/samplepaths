"""
Vasicek process
===============

A Vasicek process reverts to its mean. Started above or below it, the
realizations converge towards the mean at the speed of reversion.
"""

import matplotlib.pyplot as plt
import numpy as np

from samplepaths import VasicekProcess

vasicek = VasicekProcess(speed=3.0, mean=1.0, vol=0.4, t=2.0, rng=np.random.default_rng(13))

fig, ax = plt.subplots(figsize=(8, 4))
for initial in (3.0, 2.0, 0.0, -1.0):
    path = vasicek.sample_with_initial(1000, initial=initial)
    ax.plot(vasicek.times, path, label=f"initial = {initial:g}")
ax.axhline(1.0, linestyle="--", color="gray")
ax.set_xlabel("Time")
ax.set_ylabel("Value")
ax.legend()
plt.show()
