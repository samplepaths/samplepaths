"""
Verhulst process
================

Logistic growth with noise: a population grows at the rate ``rate`` while it
is small and levels off around the capacity of its environment.
"""

import matplotlib.pyplot as plt
import numpy as np

from samplepaths import VerhulstProcess

verhulst = VerhulstProcess(
    rate=1.5, capacity=1000.0, vol=0.15, t=10.0, rng=np.random.default_rng(120)
)

fig, ax = plt.subplots(figsize=(8, 4))
for _ in range(5):
    path = verhulst.sample_with_initial(2000, initial=10.0)
    ax.plot(verhulst.times, path, linewidth=0.9)
ax.axhline(1000.0, linestyle="--", color="gray", label="capacity")
ax.set_xlabel("Time")
ax.set_ylabel("Population")
ax.legend()
plt.show()

# %%
# The noise is proportional to the population, so it lowers the level around
# which the population settles: to about ``capacity * (1 - vol ** 2 / (2 * rate))``.

final = verhulst.sample_paths_with_initial(2000, paths=5000, initial=10.0)[:, -1]
print("mean population:", round(final.mean()), "expected:", round(1000 * (1 - 0.15**2 / 3.0)))
