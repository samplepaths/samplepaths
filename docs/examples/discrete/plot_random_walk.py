"""
Random walk
===========

A random walk adds a random step at each time. The steps and their weights
are free; here the walk is biased upwards.
"""

import matplotlib.pyplot as plt
import numpy as np

from samplepaths import RandomWalk

walk = RandomWalk(steps=[-1, 0, 2], weights=[1, 1, 2], rng=np.random.default_rng(17))
print("step probabilities:", walk.p)

fig, ax = plt.subplots(figsize=(8, 4))
for _ in range(5):
    ax.step(np.arange(101), walk.sample(100), where="post")
ax.set_xlabel("Step")
ax.set_ylabel("Position")
plt.show()
