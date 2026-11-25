"""
Fractional Gaussian noise
=========================

The increments of a fractional Brownian motion are correlated. The sample
autocorrelation is negative for a Hurst parameter below 0.5 and positive
above it.
"""

import matplotlib.pyplot as plt
import numpy as np

from samplepaths import FractionalGaussianNoise

lags = np.arange(1, 11)

fig, ax = plt.subplots(figsize=(8, 4))
for hurst in (0.2, 0.5, 0.8):
    noise = FractionalGaussianNoise(hurst=hurst, rng=np.random.default_rng(23))
    samples = noise.sample_paths(1024, paths=200)
    centered = samples - samples.mean(axis=1, keepdims=True)
    variance = (centered**2).mean()
    correlation = [(centered[:, lag:] * centered[:, :-lag]).mean() / variance for lag in lags]
    ax.plot(lags, correlation, marker="o", label=f"hurst = {hurst}")
ax.axhline(0, color="gray", linestyle="--")
ax.set_xlabel("Lag")
ax.set_ylabel("Autocorrelation")
ax.legend()
plt.show()

# %%
# Two exact algorithms are available. The Davies-Harte method is the default
# and the faster one.

noise = FractionalGaussianNoise(hurst=0.7, rng=np.random.default_rng(23))
print(noise.sample_davies_harte(4))
print(noise.sample_hosking(4))
