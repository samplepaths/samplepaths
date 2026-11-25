"""
Merton jump diffusion
=====================

A geometric Brownian motion whose value gaps at the arrivals of a Poisson
process. The logarithm of a jump is normal. The mean effect of the jumps is
taken out of the drift, so the mean still grows at the rate ``drift``.
"""

import matplotlib.pyplot as plt
import numpy as np

from samplepaths import MertonJumpDiffusion

merton = MertonJumpDiffusion(
    drift=0.05,
    volatility=0.15,
    rate=2.0,
    jump_mean=-0.1,
    jump_volatility=0.15,
    t=3.0,
    rng=np.random.default_rng(105),
)

fig, ax = plt.subplots(figsize=(8, 4))
for _ in range(5):
    path = merton.sample_with_initial(3000, initial=100.0)
    ax.plot(merton.times, path, linewidth=0.9)
ax.set_xlabel("Time")
ax.set_ylabel("Price")
plt.show()

# %%
# The jumps give the returns heavier tails than a normal distribution.

returns = np.log(merton.sample_paths(1, paths=200_000)[:, -1])
standardized = (returns - returns.mean()) / returns.std()
print("mean relative jump:", round(merton.mean_relative_jump, 4))
print("share of returns beyond 4 standard deviations:", np.mean(np.abs(standardized) > 4))
