"""
Two-factor Cox-Ingersoll-Ross model
===================================

An interest rate which is the sum of two independent Cox-Ingersoll-Ross
processes, a fast one and a slow one. The rate is never negative.
"""

import matplotlib.pyplot as plt
import numpy as np

from samplepaths import TwoFactorCoxIngersollRossModel

model = TwoFactorCoxIngersollRossModel(
    speed=(3.0, 0.2),
    mean=(0.01, 0.03),
    vol=(0.1, 0.05),
    t=20.0,
    rng=np.random.default_rng(168),
)
rate, fast, slow = model.sample(2000)

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(model.times, rate, label="rate")
ax.plot(model.times, fast, linewidth=0.8, label="fast factor")
ax.plot(model.times, slow, linewidth=0.8, label="slow factor")
ax.set_xlabel("Years")
ax.legend()
plt.show()

# %%
# The factors are drawn from their exact distributions, so a few steps are
# enough for the rate after 20 years.

final = model.sample_paths(4, paths=100_000)[:, 0, -1]
print("mean rate:", round(final.mean(), 4), "lowest rate:", round(final.min(), 5))

fig, ax = plt.subplots(figsize=(8, 3.5))
ax.hist(final, bins=80, density=True)
ax.set_xlabel("Rate after 20 years")
plt.show()
