"""
Blue noise
==========

Blue noise has a power spectral density which rises in proportion to the frequency: colored noise
with an exponent ``beta`` of -1. Its values avoid staying on the same side for long, which spreads them evenly; it is used for dithering.
"""

import matplotlib.pyplot as plt
import numpy as np

from samplepaths import BlueNoise

n = 2048
noise = BlueNoise(rng=np.random.default_rng(144))
signal = noise.sample(n)

fig, ax = plt.subplots(figsize=(8, 3.5))
ax.plot(noise.times[1:], signal, linewidth=0.6)
ax.set_xlabel("Time")
plt.show()

# %%
# The spectrum, averaged over many realizations, follows the power law. The
# dashed line has the slope 1 of the law.

signals = noise.sample_paths(n, paths=400)
power = (np.abs(np.fft.rfft(signals, axis=1)) ** 2).mean(axis=0)[1:-1]
frequencies = np.fft.rfftfreq(n, d=1.0 / n)[1:-1]

fig, ax = plt.subplots(figsize=(8, 3.5))
ax.loglog(frequencies, power)
ax.loglog(
    frequencies, power[9] * (frequencies / frequencies[9]) ** 1.0, linestyle="--", color="gray"
)
ax.set_xlabel("Frequency")
ax.set_ylabel("Power")
plt.show()
