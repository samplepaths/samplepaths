"""
Fractional Brownian motion
==========================

The Hurst parameter controls the correlation of the increments: below 0.5 the
paths are rough, above 0.5 they are smooth, and 0.5 gives a standard Brownian
motion.
"""

import matplotlib.pyplot as plt
import numpy as np

from samplepaths import FractionalBrownianMotion

fig, axes = plt.subplots(1, 3, figsize=(9, 3), sharey=True)
for ax, hurst in zip(axes, (0.2, 0.5, 0.8), strict=True):
    fbm = FractionalBrownianMotion(hurst=hurst, rng=np.random.default_rng(6))
    path = fbm.sample(1024)
    ax.plot(fbm.times, path)
    ax.set_title(f"hurst = {hurst}")
    ax.set_xlabel("Time")
plt.show()
