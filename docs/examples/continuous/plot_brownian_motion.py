"""
Brownian motion
===============

A Brownian motion with a drift and a scale. The dashed line is its expected
value, ``drift * time``.
"""

import matplotlib.pyplot as plt
import numpy as np

from samplepaths import BrownianMotion

bm = BrownianMotion(drift=1.0, scale=0.5, t=4.0, rng=np.random.default_rng(1))

fig, ax = plt.subplots(figsize=(8, 4))
for _ in range(5):
    path = bm.sample(1000)
    ax.plot(bm.times, path)
ax.plot(bm.times, bm.drift * bm.times, linestyle="--", color="gray", label="expected value")
ax.set_xlabel("Time")
ax.set_ylabel("Value")
ax.legend()
plt.show()

# %%
# The same process can be sampled at times of your choice, which do not need
# to be equally spaced.

times = np.array([0.0, 0.1, 0.5, 2.0, 2.2, 4.0])
print(bm.sample_at(times))
