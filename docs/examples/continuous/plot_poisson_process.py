"""
Poisson process
===============

A realization of a Poisson process is the sequence of its arrival times.
Plotting the number of arrivals against time gives the counting process.
"""

import matplotlib.pyplot as plt
import numpy as np

from samplepaths import PoissonProcess

poisson = PoissonProcess(rate=2.0, rng=np.random.default_rng(12))

fig, ax = plt.subplots(figsize=(8, 4))
for _ in range(4):
    arrivals = poisson.sample_with_length(10.0)
    ax.step(arrivals, np.arange(len(arrivals)), where="post")
ax.plot([0, 10], [0, 20], linestyle="--", color="gray", label="rate * time")
ax.set_xlim(0, 10)
ax.set_xlabel("Time")
ax.set_ylabel("Arrivals")
ax.legend()
plt.show()

# %%
# ``sample(n)`` generates the times of exactly ``n`` arrivals instead.

print(poisson.sample(5))
