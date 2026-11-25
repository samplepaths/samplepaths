"""
Seasonal ARIMA process
======================

An ARIMA process with a second set of coefficients acting at multiples of a
season length: twelve steps for monthly data with a yearly pattern.
"""

import matplotlib.pyplot as plt
import numpy as np

from samplepaths import SARIMAProcess

model = SARIMAProcess(ar=0.4, seasonal_ar=0.9, period=12, rng=np.random.default_rng(155))
series = model.sample(240)

fig, ax = plt.subplots(figsize=(8, 3.5))
ax.plot(series, linewidth=0.9)
ax.set_xlabel("Month")
plt.show()

# %%
# The seasonal coefficient shows in the correlation between values: it peaks
# again every twelve steps.

paths = model.sample_paths(400, paths=20_000)
lags = np.arange(40)
correlation = [np.corrcoef(paths[:, -1], paths[:, -1 - lag])[0, 1] for lag in lags]

fig, ax = plt.subplots(figsize=(8, 3.5))
ax.bar(lags, correlation)
ax.set_xlabel("Lag")
ax.set_ylabel("Correlation")
plt.show()
