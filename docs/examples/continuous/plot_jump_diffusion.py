"""
Jump diffusions
===============

A jump diffusion is a geometric Brownian motion whose value gaps at the jumps
of a Poisson process. The Merton model has symmetric jumps; the Kou model
can make downward jumps larger than upward ones.
"""

import matplotlib.pyplot as plt
import numpy as np

from samplepaths import GeometricBrownianMotion, KouJumpDiffusion, MertonJumpDiffusion

processes = {
    "Geometric Brownian motion": GeometricBrownianMotion(drift=0.05, volatility=0.2),
    "Merton": MertonJumpDiffusion(
        drift=0.05, volatility=0.2, rate=3.0, jump_mean=-0.05, jump_volatility=0.1
    ),
    "Kou": KouJumpDiffusion(
        drift=0.05, volatility=0.2, rate=3.0, up_probability=0.3, up_rate=25.0, down_rate=8.0
    ),
}

fig, axes = plt.subplots(1, 3, figsize=(10, 3.5), sharey=True)
for ax, (name, process) in zip(axes, processes.items(), strict=True):
    process.rng = np.random.default_rng(28)
    for _ in range(4):
        path = process.sample(1000)
        ax.plot(process.times, path)
    ax.set_title(name)
    ax.set_xlabel("Time")
axes[0].set_ylabel("Price")
plt.show()

# %%
# The drift is the expected rate of return including the jumps, so the three
# models have the same mean. The jumps change the shape of the distribution
# of the returns: they make its tails heavier, and those of the Kou model
# heavier on the downside.

fig, ax = plt.subplots(figsize=(8, 4))
bins = np.linspace(-1.0, 1.0, 120)
for name, process in processes.items():
    process.rng = np.random.default_rng(29)
    final = process.sample_paths(8, paths=200_000)[:, -1]
    print(f"{name}: mean {final.mean():.4f} (expected {np.exp(0.05):.4f})")
    ax.hist(np.log(final), bins=bins, density=True, histtype="step", label=name)
ax.set_yscale("log")
ax.set_xlabel("Logarithm of the final price")
ax.set_ylabel("Density")
ax.legend()
plt.show()
