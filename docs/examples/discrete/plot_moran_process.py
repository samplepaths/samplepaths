"""
Moran process
=============

A Moran process follows the number of individuals of one type in a
population of fixed size. It ends when the type dies out or takes over.
"""

import matplotlib.pyplot as plt
import numpy as np

from samplepaths import MoranProcess

moran = MoranProcess(maximum=30, rng=np.random.default_rng(19))

fig, ax = plt.subplots(figsize=(8, 4))
for _ in range(6):
    sizes = moran.sample_with_start(2000, start=15)
    ax.step(np.arange(len(sizes)), sizes, where="post")
ax.axhline(0, color="gray", linestyle="--")
ax.axhline(30, color="gray", linestyle="--")
ax.set_xlabel("Step")
ax.set_ylabel("Individuals of the type")
plt.show()
