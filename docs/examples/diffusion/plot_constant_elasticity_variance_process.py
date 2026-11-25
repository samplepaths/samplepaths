"""
Constant elasticity of variance process
=======================================

A price whose volatility depends on its level: the noise is proportional to
the price to the power ``volexp``. Below 1, the relative volatility rises
when the price falls, as observed for stocks.
"""

import matplotlib.pyplot as plt
import numpy as np

from samplepaths import ConstantElasticityVarianceProcess

fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharey=True)
for ax, volexp in zip(axes, (0.5, 1.0), strict=True):
    process = ConstantElasticityVarianceProcess(
        drift=0.05, vol=0.3, volexp=volexp, t=3.0, rng=np.random.default_rng(115)
    )
    for _ in range(5):
        path = process.sample(1500)
        ax.plot(process.times, path, linewidth=0.9)
    ax.set_title(f"volexp = {volexp}")
    ax.set_xlabel("Time")
plt.show()

# %%
# With ``volexp=1`` the process is a geometric Brownian motion and the
# logarithm of its value is symmetric. A smaller exponent skews it to the
# left.

fig, ax = plt.subplots(figsize=(8, 3.5))
for volexp in (0.5, 1.0):
    process = ConstantElasticityVarianceProcess(
        drift=0.05, vol=0.3, volexp=volexp, rng=np.random.default_rng(116)
    )
    final = process.sample_paths(256, paths=50_000)[:, -1]
    ax.hist(
        np.log(final),
        bins=np.linspace(-1.5, 1.0, 101),
        density=True,
        histtype="step",
        label=f"volexp = {volexp}",
    )
ax.set_xlabel("Logarithm of the value at time 1")
ax.legend()
plt.show()
