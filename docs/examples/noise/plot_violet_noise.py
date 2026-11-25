"""
Violet noise
============

Violet noise has a power spectral density which rises as the square of the frequency: colored noise
with an exponent ``beta`` of -2. It is the derivative of white noise, and changes sign at almost every step.
"""

import matplotlib.pyplot as plt
import numpy as np

from samplepaths import VioletNoise

n = 2048
noise = VioletNoise(rng=np.random.default_rng(146))
signal = noise.sample(n)

fig, ax = plt.subplots(figsize=(8, 3.5))
ax.plot(noise.times[1:], signal, linewidth=0.6)
ax.set_xlabel("Time")
plt.show()

# %%
# The spectrum, averaged over many realizations, follows the power law. The
# dashed line has the slope 2 of the law.

signals = noise.sample_paths(n, paths=400)
power = (np.abs(np.fft.rfft(signals, axis=1)) ** 2).mean(axis=0)[1:-1]
frequencies = np.fft.rfftfreq(n, d=1.0 / n)[1:-1]

fig, ax = plt.subplots(figsize=(8, 3.5))
ax.loglog(frequencies, power)
ax.loglog(
    frequencies, power[9] * (frequencies / frequencies[9]) ** 2.0, linestyle="--", color="gray"
)
ax.set_xlabel("Frequency")
ax.set_ylabel("Power")
plt.show()
