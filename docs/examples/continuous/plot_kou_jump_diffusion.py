"""
Kou jump diffusion
==================

A jump diffusion whose jumps are exponential in size and go up with a
probability of your choice. Rare, large falls and frequent, small rises give
the skew observed in the returns of stocks.
"""

import matplotlib.pyplot as plt
import numpy as np

from samplepaths import KouJumpDiffusion

kou = KouJumpDiffusion(
    drift=0.05,
    volatility=0.15,
    rate=4.0,
    up_probability=0.7,
    up_rate=40.0,
    down_rate=6.0,
    t=3.0,
    rng=np.random.default_rng(106),
)

fig, ax = plt.subplots(figsize=(8, 4))
for _ in range(5):
    path = kou.sample_with_initial(3000, initial=100.0)
    ax.plot(kou.times, path, linewidth=0.9)
ax.set_xlabel("Time")
ax.set_ylabel("Price")
plt.show()

# %%
# The distribution of the returns over a short time has a long left tail.

returns = np.log(kou.sample_paths(1, paths=200_000)[:, -1])

fig, ax = plt.subplots(figsize=(8, 3.5))
ax.hist(returns, bins=np.linspace(-2.5, 1.5, 121), density=True)
ax.set_yscale("log")
ax.set_xlabel("Logarithm of the final value")
plt.show()
