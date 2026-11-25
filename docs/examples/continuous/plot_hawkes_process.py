"""
Hawkes process
==============

In a Hawkes process every arrival raises the rate of the following ones for a
while, so arrivals come in clusters. A Poisson process with the same average
rate spreads its arrivals evenly.
"""

import matplotlib.pyplot as plt
import numpy as np

from samplepaths import HawkesProcess, PoissonProcess

hawkes = HawkesProcess(baseline=1.0, excitation=3.0, decay=4.0, rng=np.random.default_rng(30))
average_rate = hawkes.baseline / (1 - hawkes.excitation / hawkes.decay)
poisson = PoissonProcess(rate=average_rate, rng=np.random.default_rng(31))
print("average rate of both processes:", average_rate)

length = 40.0
arrivals = {
    "Hawkes": hawkes.sample_with_length(length)[1:-1],
    "Poisson": poisson.sample_with_length(length)[1:-1],
}

fig, axes = plt.subplots(2, 1, figsize=(9, 3.5), sharex=True)
for ax, (name, times) in zip(axes, arrivals.items(), strict=True):
    ax.vlines(times, 0, 1, linewidth=0.8)
    ax.set_yticks([])
    ax.set_ylabel(name)
    ax.grid(False)
axes[-1].set_xlim(0, length)
axes[-1].set_xlabel("Time")
plt.show()

# %%
# The rate of the Hawkes process jumps at every arrival and decays between
# them.

times = arrivals["Hawkes"]
grid = np.linspace(0, 10, 2000)
elapsed = grid[:, np.newaxis] - times[np.newaxis, :]
kernel = np.where(elapsed > 0, np.exp(-hawkes.decay * np.clip(elapsed, 0, None)), 0.0)
rate = hawkes.baseline + hawkes.excitation * kernel.sum(axis=1)

fig, ax = plt.subplots(figsize=(9, 3.5))
ax.plot(grid, rate)
ax.axhline(hawkes.baseline, linestyle="--", color="gray", label="baseline")
ax.set_xlabel("Time")
ax.set_ylabel("Rate")
ax.legend()
plt.show()
