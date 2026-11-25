"""
Autoregressive and moving average processes
===========================================

An autoregressive process remembers its past values, and its correlation
with them fades gradually. A moving average process remembers the past noise
only, and its correlation stops after as many steps as it has coefficients.
"""

import matplotlib.pyplot as plt
import numpy as np

from samplepaths import ARIMAProcess, AutoregressiveProcess, MovingAverageProcess

rng = np.random.default_rng(31)
processes = {
    "AR(1)": AutoregressiveProcess(coefficients=0.8, rng=rng),
    "MA(2)": MovingAverageProcess(coefficients=[0.8, 0.4], rng=rng),
}

fig, axes = plt.subplots(2, 2, figsize=(9, 5), width_ratios=[2, 1])
for row, (name, process) in zip(axes, processes.items(), strict=True):
    row[0].plot(process.sample(300), linewidth=0.9)
    row[0].set_ylabel(name)

    # Correlation between values a number of steps apart, over many
    # realizations, once the start is forgotten.
    paths = process.sample_paths(200, paths=20_000)
    lags = np.arange(8)
    correlation = [np.corrcoef(paths[:, -1], paths[:, -1 - lag])[0, 1] for lag in lags]
    row[1].bar(lags, correlation)
axes[1, 0].set_xlabel("Step")
axes[1, 1].set_xlabel("Lag")
axes[0, 1].set_title("Correlation")
plt.show()

# %%
# Differencing turns a stationary series into one which wanders. An ARIMA
# process with ``d=1`` is the running sum of an ARMA process.

fig, ax = plt.subplots(figsize=(8, 3.5))
for order in (0, 1):
    process = ARIMAProcess(ar=0.5, ma=0.3, d=order, rng=np.random.default_rng(32))
    ax.plot(process.sample(300), label=f"d = {order}")
ax.set_xlabel("Step")
ax.legend()
plt.show()
