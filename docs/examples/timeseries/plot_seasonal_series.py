"""
A seasonal series
=================

Monthly figures often repeat a yearly pattern which itself drifts. A
seasonal ARIMA process with one seasonal difference and a period of 12
produces such series: every month starts from its value of a year before.
"""

import matplotlib.pyplot as plt
import numpy as np

from samplepaths import SARIMAProcess

process = SARIMAProcess(
    ar=0.5,
    seasonal_ma=-0.4,
    seasonal_d=1,
    period=12,
    rng=np.random.default_rng(12),
)
series = process.sample(120)  # ten years of months

fig, ax = plt.subplots(figsize=(8, 3.5))
ax.plot(series)
for year in range(0, 121, 12):
    ax.axvline(year, linewidth=0.5, alpha=0.4)
ax.set_xlabel("Month")
plt.show()

# %%
# Laying the years over each other shows the pattern they share.

fig, ax = plt.subplots(figsize=(8, 3.5))
for year in series.reshape(10, 12)[2:]:
    ax.plot(np.arange(1, 13), year, linewidth=1.0)
ax.set_xlabel("Month of the year")
plt.show()
