"""
Cox-Ingersoll-Ross process
==========================

The volatility of a Cox-Ingersoll-Ross process shrinks as the process
approaches zero, which keeps it positive when the mean reversion is strong
enough. It is a common model for interest rates.
"""

import matplotlib.pyplot as plt
import numpy as np

from samplepaths import CoxIngersollRossProcess

cir = CoxIngersollRossProcess(speed=2.0, mean=0.05, vol=0.1, t=5.0, rng=np.random.default_rng(14))

fig, ax = plt.subplots(figsize=(8, 4))
for _ in range(5):
    path = cir.sample_with_initial(2000, initial=0.02)
    ax.plot(cir.times, path)
ax.axhline(0.05, linestyle="--", color="gray")
ax.set_xlabel("Time (years)")
ax.set_ylabel("Rate")
plt.show()
