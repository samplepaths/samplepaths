"""
Moving average process
======================

Every value is a combination of the current and of a few previous values of
a white noise. Values further apart than the number of coefficients are
independent.
"""

import matplotlib.pyplot as plt
import numpy as np

from samplepaths import MovingAverageProcess

model = MovingAverageProcess(coefficients=[0.8, 0.6, 0.4], rng=np.random.default_rng(152))

fig, ax = plt.subplots(figsize=(8, 3.5))
ax.plot(model.sample(300), linewidth=0.9)
ax.set_xlabel("Step")
plt.show()

# %%
# With three coefficients the correlation stops after three steps.

paths = model.sample_paths(50, paths=50_000)
lags = np.arange(8)
correlation = [np.corrcoef(paths[:, -1], paths[:, -1 - lag])[0, 1] for lag in lags]

fig, ax = plt.subplots(figsize=(8, 3.5))
ax.bar(lags, correlation)
ax.set_xlabel("Lag")
ax.set_ylabel("Correlation")
plt.show()
