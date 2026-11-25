"""
Gaussian noise
==============

Gaussian noise is the sequence of increments of a Wiener process. Its
cumulative sum is the Wiener process itself.
"""

import matplotlib.pyplot as plt
import numpy as np

from samplepaths import GaussianNoise

noise = GaussianNoise(t=1.0, rng=np.random.default_rng(22))
increments = noise.sample(1000)

fig, (top, bottom) = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
top.plot(noise.times[1:], increments, linewidth=0.6)
top.set_ylabel("Increment")
bottom.plot(noise.times, np.concatenate(([0.0], np.cumsum(increments))))
bottom.set_ylabel("Cumulative sum")
bottom.set_xlabel("Time")
plt.show()
