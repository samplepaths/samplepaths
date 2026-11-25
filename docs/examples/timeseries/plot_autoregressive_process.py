"""
Autoregressive process
======================

Every value is a combination of the previous ones plus noise. The
coefficients decide whether the series is smooth, or swings back and forth.
"""

import matplotlib.pyplot as plt
import numpy as np

from samplepaths import AutoregressiveProcess

models = {
    "coefficient 0.9": AutoregressiveProcess(coefficients=0.9),
    "coefficient -0.9": AutoregressiveProcess(coefficients=-0.9),
    "coefficients 1.6 and -0.9": AutoregressiveProcess(coefficients=[1.6, -0.9]),
}

fig, axes = plt.subplots(len(models), 1, figsize=(8, 6), sharex=True)
for ax, (label, model) in zip(axes, models.items(), strict=True):
    model.rng = np.random.default_rng(151)
    ax.plot(model.sample(200), linewidth=0.9)
    ax.set_ylabel(label, fontsize=8)
axes[-1].set_xlabel("Step")
plt.show()

# %%
# Two coefficients can give a series which oscillates with a period of its
# own: here about 11 steps, visible in the correlation between values a
# number of steps apart.

paths = models["coefficients 1.6 and -0.9"].sample_paths(300, paths=20_000)
lags = np.arange(30)
correlation = [np.corrcoef(paths[:, -1], paths[:, -1 - lag])[0, 1] for lag in lags]

fig, ax = plt.subplots(figsize=(8, 3.5))
ax.bar(lags, correlation)
ax.set_xlabel("Lag")
ax.set_ylabel("Correlation")
plt.show()
