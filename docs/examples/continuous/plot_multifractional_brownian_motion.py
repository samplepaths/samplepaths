"""
Multifractional Brownian motion
===============================

The Hurst parameter is a function of time, so the roughness of the path
changes along it. Here it rises from 0.2 to 0.9.
"""

import matplotlib.pyplot as plt
import numpy as np

from samplepaths import MultifractionalBrownianMotion


def hurst(t: float) -> float:
    return 0.2 + 0.7 * t


mbm = MultifractionalBrownianMotion(hurst=hurst, t=1.0, rng=np.random.default_rng(7))
path = mbm.sample(1024)

fig, (top, bottom) = plt.subplots(2, 1, figsize=(8, 5), sharex=True, height_ratios=(3, 1))
top.plot(mbm.times, path)
top.set_ylabel("Value")
bottom.plot(mbm.times, [hurst(t) for t in mbm.times], color="gray")
bottom.set_ylabel("Hurst")
bottom.set_xlabel("Time")
plt.show()
