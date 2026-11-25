"""
Brownian excursion
==================

A Brownian excursion is a Brownian motion conditioned to stay nonnegative and
to return to zero at time ``t``.
"""

import matplotlib.pyplot as plt
import numpy as np

from samplepaths import BrownianExcursion

excursion = BrownianExcursion(t=1.0, rng=np.random.default_rng(3))

fig, ax = plt.subplots(figsize=(8, 4))
for _ in range(4):
    path = excursion.sample(1024)
    ax.plot(excursion.times, path)
ax.set_xlabel("Time")
ax.set_ylabel("Value")
plt.show()
