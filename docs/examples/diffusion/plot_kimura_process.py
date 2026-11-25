"""
Kimura process
==============

The frequency of one of two variants of a gene in a population. Chance alone
makes it wander until one variant is lost; selection pushes it towards one
side.
"""

import matplotlib.pyplot as plt
import numpy as np

from samplepaths import KimuraProcess

fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharey=True)
for ax, selection in zip(axes, (0.0, 2.0), strict=True):
    kimura = KimuraProcess(selection=selection, vol=1.0, t=4.0, rng=np.random.default_rng(118))
    for _ in range(6):
        path = kimura.sample_with_initial(2000, initial=0.3)
        ax.plot(kimura.times, path, linewidth=0.9)
    ax.set_title(f"selection = {selection}")
    ax.set_xlabel("Time")
axes[0].set_ylabel("Frequency")
plt.show()

# %%
# Without selection, a variant which starts at a frequency of 0.3 takes over
# the population with a probability of 0.3. Selection in its favor raises
# that probability.

for selection in (0.0, 2.0):
    kimura = KimuraProcess(selection=selection, vol=1.0, t=20.0, rng=np.random.default_rng(119))
    final = kimura.sample_paths_with_initial(4000, paths=5000, initial=0.3)[:, -1]
    print(f"selection {selection}: fixed in {np.mean(final == 1.0):.1%} of the populations")
