"""
Feller logistic process
=======================

A population with logistic growth and the noise of births and deaths, which
is proportional to the square root of the population. A small population can
die out, and then stays extinct.
"""

import matplotlib.pyplot as plt
import numpy as np

from samplepaths import FellerLogisticProcess

process = FellerLogisticProcess(
    speed=1.0, level=3.0, vol=0.8, t=10.0, rng=np.random.default_rng(122)
)

fig, ax = plt.subplots(figsize=(8, 4))
for _ in range(6):
    path = process.sample_with_initial(4000, initial=0.3)
    ax.plot(process.times, path, linewidth=0.9)
ax.axhline(3.0, linestyle="--", color="gray", label="level")
ax.set_xlabel("Time")
ax.set_ylabel("Population")
ax.legend()
plt.show()

# %%
# The smaller the population at the start, the more often it dies out before
# it can grow.

for initial in (0.1, 0.3, 1.0):
    final = process.sample_paths_with_initial(4000, paths=4000, initial=initial)[:, -1]
    print(f"initial population {initial}: extinct in {np.mean(final == 0):.0%} of the cases")
