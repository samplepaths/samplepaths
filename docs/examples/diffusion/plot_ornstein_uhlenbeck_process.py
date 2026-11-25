"""
Ornstein-Uhlenbeck process
==========================

A process pulled back to zero with a force proportional to its distance. It
models a velocity with friction, or any noisy quantity with a resting level.
"""

import matplotlib.pyplot as plt
import numpy as np

from samplepaths import OrnsteinUhlenbeckProcess

process = OrnsteinUhlenbeckProcess(speed=3.0, vol=1.0, t=4.0, rng=np.random.default_rng(113))

fig, ax = plt.subplots(figsize=(8, 4))
for initial in (3.0, 0.0, -2.0):
    path = process.sample_with_initial(2000, initial=initial)
    ax.plot(process.times, path, linewidth=0.9)
ax.axhline(0.0, linestyle="--", color="gray")
ax.set_xlabel("Time")
plt.show()

# %%
# After a while the values settle into a normal distribution with variance
# ``vol ** 2 / (2 * speed)``, whatever the initial value.

final = process.sample_paths_with_initial(512, paths=50_000, initial=3.0)[:, -1]
print("variance:", round(final.var(), 4), "expected:", round(1.0 / (2 * 3.0), 4))
