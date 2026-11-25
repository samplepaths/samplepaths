"""
Cauchy process
==============

A Cauchy process has heavy-tailed increments, so its paths show occasional
large jumps.
"""

import matplotlib.pyplot as plt
import numpy as np

from samplepaths import CauchyProcess

cauchy = CauchyProcess(t=1.0, rng=np.random.default_rng(8))

fig, ax = plt.subplots(figsize=(8, 4))
for _ in range(3):
    path = cauchy.sample(2000)
    ax.plot(cauchy.times, path)
ax.set_xlabel("Time")
ax.set_ylabel("Value")
plt.show()
