"""
Squared Bessel process
======================

The squared distance from the origin of a Brownian motion in ``dim``
dimensions. Its mean grows as ``dim`` times the time.
"""

import matplotlib.pyplot as plt
import numpy as np

from samplepaths import SquaredBesselProcess

fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharey=True)
for ax, dim in zip(axes, (1, 3), strict=True):
    process = SquaredBesselProcess(dim=dim, t=2.0, rng=np.random.default_rng(109))
    for _ in range(5):
        path = process.sample(1000)
        ax.plot(process.times, path, linewidth=0.9)
    ax.plot(process.times, dim * process.times, linestyle="--", color="gray", label="mean")
    ax.set_title(f"dim = {dim}")
    ax.set_xlabel("Time")
axes[0].legend()
plt.show()

# %%
# In one dimension the process returns to zero again and again; in three
# dimensions it leaves and does not come back.

for dim in (1, 3):
    process = SquaredBesselProcess(dim=dim, t=2.0, rng=np.random.default_rng(110))
    paths = process.sample_paths(1000, paths=2000)
    close = np.mean(paths[:, 500:].min(axis=1) < 0.01)
    print(f"dim {dim}: {close:.0%} of the paths come within 0.01 of zero after time 1")
