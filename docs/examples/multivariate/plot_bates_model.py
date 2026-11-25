"""
Bates model
===========

A Heston model whose price also jumps. The random variance gives the
returns a skew over months; the jumps add sudden falls over days.
"""

import matplotlib.pyplot as plt
import numpy as np

from samplepaths import BatesModel, HestonModel

parameters = {"drift": 0.05, "speed": 3.0, "mean": 0.04, "vol": 0.4, "correlation": -0.7}
bates = BatesModel(
    **parameters,
    rate=1.5,
    jump_mean=-0.08,
    jump_volatility=0.06,
    t=2.0,
    rng=np.random.default_rng(163),
)
price, variance = bates.sample_with_initial(2000, initial=100.0)

fig, axes = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
axes[0].plot(bates.times, price)
axes[0].set_ylabel("Price")
axes[1].plot(bates.times, np.sqrt(variance), color="C1")
axes[1].set_ylabel("Volatility")
axes[1].set_xlabel("Time")
plt.show()

# %%
# Over a week, the returns of a Heston model are close to normal. The jumps
# of the Bates model give them a left tail.

week = 1 / 52
bins = np.linspace(-0.3, 0.15, 91)

fig, ax = plt.subplots(figsize=(8, 3.5))
for label, model in (
    (
        "Bates model",
        BatesModel(**parameters, rate=1.5, jump_mean=-0.08, jump_volatility=0.06, t=week),
    ),
    ("Heston model", HestonModel(**parameters, t=week)),
):
    model.rng = np.random.default_rng(164)
    returns = np.log(model.sample_paths(4, paths=400_000)[:, 0, -1])
    ax.hist(returns, bins=bins, density=True, histtype="step", label=label)
ax.set_yscale("log")
ax.set_xlabel("Return over a week")
ax.legend()
plt.show()
