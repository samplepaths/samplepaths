"""
An interest rate with two factors
=================================

In a two-factor Cox-Ingersoll-Ross model the rate is the sum of a fast and a
slow factor. The fast one gives the day-to-day moves, the slow one the level
around which they happen.
"""

import matplotlib.pyplot as plt
import numpy as np

from samplepaths import HullWhiteTwoFactorModel, TwoFactorCoxIngersollRossModel

model = TwoFactorCoxIngersollRossModel(
    speed=(3.0, 0.15),
    mean=(0.01, 0.03),
    vol=(0.12, 0.04),
    t=20,
    rng=np.random.default_rng(61),
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
# In a two-factor Hull-White model the second factor moves the level to
# which the rate reverts. The bands hold 90% of 5000 realizations of the
# rate: they are wider than those of the same model without the factor.

fig, ax = plt.subplots(figsize=(8, 4))
for label, factor_vol in (("with the factor", 0.02), ("without", 0.0)):
    model = HullWhiteTwoFactorModel(
        speed=1.0,
        drift=0.03,
        vol=0.01,
        factor_speed=0.2,
        factor_vol=factor_vol,
        t=10,
        rng=np.random.default_rng(62),
    )
    rates = model.sample_paths_with_initial(200, paths=5000, initial=0.03)[:, 0]
    low, high = np.quantile(rates, [0.05, 0.95], axis=0)
    ax.fill_between(model.times, low, high, alpha=0.35, label=label)
ax.set_xlabel("Years")
ax.set_ylabel("Rate")
ax.legend()
plt.show()
