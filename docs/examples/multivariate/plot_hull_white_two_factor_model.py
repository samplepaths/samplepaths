"""
Two-factor Hull-White model
===========================

An interest rate which reverts to a level that itself moves at random. The
second factor is slow, so it carries the lasting changes of the rate.
"""

import matplotlib.pyplot as plt
import numpy as np

from samplepaths import HullWhiteTwoFactorModel

model = HullWhiteTwoFactorModel(
    speed=1.5,
    drift=0.045,
    vol=0.008,
    factor_speed=0.1,
    factor_vol=0.01,
    correlation=-0.3,
    t=20.0,
    rng=np.random.default_rng(167),
)
rate, factor = model.sample_with_initial(2000, initial=0.03)

fig, axes = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
axes[0].plot(model.times, rate)
axes[0].plot(
    model.times,
    (0.045 + factor) / 1.5,
    linestyle="--",
    color="gray",
    label="level the rate reverts to",
)
axes[0].set_ylabel("Rate")
axes[0].legend()
axes[1].plot(model.times, factor, color="C1")
axes[1].set_ylabel("Factor")
axes[1].set_xlabel("Years")
plt.show()

# %%
# The rate is normal at every time, and can be negative.

final = model.sample_paths_with_initial(256, paths=50_000, initial=0.03)[:, 0, -1]
print("mean rate after 20 years:", round(final.mean(), 4))
print("share of negative rates: ", round(np.mean(final < 0), 3))
