"""
Jacobi process
==============

A process which reverts to a mean and stays between 0 and 1: its noise
vanishes at both ends. It models a proportion, or a correlation once mapped
to the interval from -1 to 1.
"""

import matplotlib.pyplot as plt
import numpy as np

from samplepaths import JacobiProcess

jacobi = JacobiProcess(speed=2.0, mean=0.7, vol=0.6, t=5.0, rng=np.random.default_rng(117))

fig, ax = plt.subplots(figsize=(8, 4))
for initial in (0.05, 0.5, 0.95):
    path = jacobi.sample_with_initial(2500, initial=initial)
    ax.plot(jacobi.times, path, linewidth=0.9)
ax.axhline(0.7, linestyle="--", color="gray")
ax.set_ylim(0, 1)
ax.set_xlabel("Time")
plt.show()

# %%
# In the long run the values follow a beta distribution around the mean.

final = jacobi.sample_paths(512, paths=50_000)[:, -1]

fig, ax = plt.subplots(figsize=(8, 3.5))
ax.hist(final, bins=np.linspace(0, 1, 81), density=True)
ax.set_xlabel("Value at time 5")
plt.show()
