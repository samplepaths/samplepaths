"""
Quantile bands
==============

Simulating many realizations at once gives the distribution of a process at
every time. Here the bands hold 50% and 90% of the realizations of a
Cox-Ingersoll-Ross interest rate model.
"""

import matplotlib.pyplot as plt
import numpy as np

from samplepaths import CoxIngersollRossProcess

cir = CoxIngersollRossProcess(speed=1.5, mean=0.05, vol=0.1, t=5.0, rng=np.random.default_rng(26))
rates = cir.sample_paths_with_initial(n=500, paths=5000, initial=0.02)
times = cir.times

fig, ax = plt.subplots(figsize=(8, 4))
for low, high, alpha in ((0.05, 0.95, 0.15), (0.25, 0.75, 0.3)):
    ax.fill_between(
        times,
        np.quantile(rates, low, axis=0),
        np.quantile(rates, high, axis=0),
        alpha=alpha,
        color="C0",
        linewidth=0,
    )
ax.plot(times, np.median(rates, axis=0), color="C0", label="median")
for path in rates[:3]:
    ax.plot(times, path, linewidth=0.7)
ax.set_xlabel("Time (years)")
ax.set_ylabel("Rate")
ax.legend()
plt.show()
