"""
The general diffusion process
=============================

All the processes of the general model are a
:py:class:`~samplepaths.diffusion.DiffusionProcess` with some parameters
fixed. Used directly, it gives models which have no name: here a process
which reverts to a mean, with a volatility proportional to its value to the
power 0.75.
"""

import matplotlib.pyplot as plt
import numpy as np

from samplepaths import DiffusionProcess

process = DiffusionProcess(
    speed=2.0, mean=1.0, vol=0.5, volexp=0.75, t=4.0, rng=np.random.default_rng(111)
)

fig, ax = plt.subplots(figsize=(8, 4))
for initial in (0.2, 1.0, 2.5):
    path = process.sample_with_initial(2000, initial=initial)
    ax.plot(process.times, path, linewidth=0.9, label=f"initial value {initial}")
ax.axhline(1.0, linestyle="--", color="gray")
ax.set_xlabel("Time")
ax.legend()
plt.show()

# %%
# The exponent decides how the spread of the process depends on its level.
# With an exponent of 0 the noise is the same everywhere; with 1 it vanishes
# near zero.

fig, ax = plt.subplots(figsize=(8, 3.5))
for volexp in (0.0, 0.5, 1.0):
    process.volexp = volexp
    process.rng = np.random.default_rng(112)
    final = process.sample_paths(256, paths=20_000)[:, -1]
    ax.hist(
        final,
        bins=np.linspace(-0.5, 3.0, 106),
        density=True,
        histtype="step",
        label=f"volexp = {volexp}",
    )
ax.set_xlabel("Value at time 4")
ax.legend()
plt.show()
