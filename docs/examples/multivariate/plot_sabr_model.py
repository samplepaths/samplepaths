"""
SABR model
==========

A forward price or rate with a volatility which is itself random. The
exponent ``beta`` sets how the size of the moves depends on the level of the
forward.
"""

import matplotlib.pyplot as plt
import numpy as np

from samplepaths import SABRModel

sabr = SABRModel(
    volvol=0.6,
    beta=0.5,
    correlation=-0.4,
    initial_volatility=0.03,
    t=5.0,
    rng=np.random.default_rng(165),
)
forward, volatility = sabr.sample_with_initial(2000, initial=0.03)

fig, axes = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
axes[0].plot(sabr.times, forward)
axes[0].set_ylabel("Forward rate")
axes[1].plot(sabr.times, volatility, color="C1")
axes[1].set_ylabel("Volatility")
axes[1].set_xlabel("Years")
plt.show()

# %%
# The distribution of the forward after five years, for three exponents with
# volatilities chosen to give moves of the same size at the start. With
# ``beta=0`` the forward can become negative; with ``beta=1`` it is
# lognormal.

fig, ax = plt.subplots(figsize=(8, 3.5))
for beta in (0.0, 0.5, 1.0):
    model = SABRModel(
        volvol=0.3,
        beta=beta,
        correlation=0.0,
        initial_volatility=0.005 / 0.03**beta,
        t=5.0,
        rng=np.random.default_rng(166),
    )
    final = model.sample_paths_with_initial(128, paths=100_000, initial=0.03)[:, 0, -1]
    ax.hist(
        final,
        bins=np.linspace(-0.02, 0.1, 121),
        density=True,
        histtype="step",
        label=f"beta = {beta}",
    )
ax.set_xlabel("Forward rate after five years")
ax.legend()
plt.show()
