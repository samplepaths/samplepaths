"""
Wiener process
==============

The Wiener process is the standard Brownian motion: it starts at zero, and
its value at time ``t`` is normal with variance ``t``.
"""

import matplotlib.pyplot as plt
import numpy as np

from samplepaths import WienerProcess

wiener = WienerProcess(t=2.0, rng=np.random.default_rng(101))

fig, ax = plt.subplots(figsize=(8, 4))
for _ in range(5):
    path = wiener.sample(1000)
    ax.plot(wiener.times, path)
ax.plot(wiener.times, 2 * np.sqrt(wiener.times), linestyle="--", color="gray")
ax.plot(
    wiener.times,
    -2 * np.sqrt(wiener.times),
    linestyle="--",
    color="gray",
    label="two standard deviations",
)
ax.set_xlabel("Time")
ax.legend()
plt.show()

# %%
# The variance of many realizations grows in proportion to the time.

paths = wiener.sample_paths(64, paths=20_000)
print("variance at time 1:", round(paths[:, 32].var(), 3))
print("variance at time 2:", round(paths[:, 64].var(), 3))
