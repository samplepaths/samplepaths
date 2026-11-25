"""
Time-dependent parameters
=========================

The parameters of a diffusion process can be functions of time. Here the
mean follows a seasonal cycle, and the process tracks it.
"""

import math

import matplotlib.pyplot as plt
import numpy as np

from samplepaths import ExtendedVasicekProcess


def mean(t: float) -> float:
    return math.sin(2 * math.pi * t)


process = ExtendedVasicekProcess(
    speed=8.0, mean=mean, vol=0.3, t=3.0, rng=np.random.default_rng(15)
)

fig, ax = plt.subplots(figsize=(8, 4))
for _ in range(3):
    path = process.sample_with_initial(1500, initial=0.0)
    ax.plot(process.times, path)
ax.plot(process.times, [mean(t) for t in process.times], linestyle="--", color="gray", label="mean")
ax.set_xlabel("Time")
ax.set_ylabel("Value")
ax.legend()
plt.show()
