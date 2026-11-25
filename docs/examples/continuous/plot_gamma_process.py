"""
Gamma process
=============

A gamma process only increases. It is specified either by the mean and the
variance of its increase per unit time, or by the rate and the scale of its
jumps.
"""

import matplotlib.pyplot as plt
import numpy as np

from samplepaths import GammaProcess

gamma = GammaProcess(mean=1.0, variance=0.5, t=5.0, rng=np.random.default_rng(9))
print("rate:", gamma.rate, "scale:", gamma.scale)

fig, ax = plt.subplots(figsize=(8, 4))
for _ in range(5):
    path = gamma.sample(1000)
    ax.plot(gamma.times, path)
ax.plot(gamma.times, gamma.mean * gamma.times, linestyle="--", color="gray", label="expected value")
ax.set_xlabel("Time")
ax.set_ylabel("Value")
ax.legend()
plt.show()
