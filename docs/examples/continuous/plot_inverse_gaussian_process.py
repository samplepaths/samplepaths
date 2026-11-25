"""
Inverse Gaussian process
========================

An inverse Gaussian process only increases. Its mean is an increasing
function of time, here quadratic, which the realizations follow.
"""

import matplotlib.pyplot as plt
import numpy as np

from samplepaths import InverseGaussianProcess


def mean(t: float) -> float:
    return t + t * t


process = InverseGaussianProcess(mean=mean, scale=50.0, t=2.0, rng=np.random.default_rng(11))

fig, ax = plt.subplots(figsize=(8, 4))
for _ in range(5):
    path = process.sample(500)
    ax.plot(process.times, path)
ax.plot(process.times, [mean(t) for t in process.times], linestyle="--", color="gray", label="mean")
ax.set_xlabel("Time")
ax.set_ylabel("Value")
ax.legend()
plt.show()
