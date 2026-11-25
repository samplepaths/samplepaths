"""
Colored noise
=============

Colored noise has a power spectral density proportional to ``1 / f**beta``.
A positive ``beta`` gives slowly varying signals and a negative one rapidly
varying signals.
"""

import matplotlib.pyplot as plt
import numpy as np

from samplepaths import ColoredNoise

colors = {"violet": -2, "blue": -1, "white": 0, "pink": 1, "red": 2}
n = 2048

fig, axes = plt.subplots(len(colors), 1, figsize=(8, 7), sharex=True)
for ax, (name, beta) in zip(axes, colors.items(), strict=True):
    noise = ColoredNoise(beta=beta, rng=np.random.default_rng(24))
    ax.plot(noise.sample(n), linewidth=0.6)
    ax.set_ylabel(name)
    ax.set_yticks([])
axes[-1].set_xlabel("Sample")
plt.show()

# %%
# The average spectrum of many realizations shows the power law.

frequencies = np.fft.rfftfreq(n, d=1.0 / n)[1:]

fig, ax = plt.subplots(figsize=(8, 4))
for name, beta in colors.items():
    samples = ColoredNoise(beta=beta, rng=np.random.default_rng(24)).sample_paths(n, paths=100)
    power = (np.abs(np.fft.rfft(samples, axis=1))[:, 1:] ** 2).mean(axis=0)
    ax.loglog(frequencies, power / power[len(power) // 8], label=name)
ax.set_xlabel("Frequency")
ax.set_ylabel("Relative power")
ax.legend(ncols=5, loc="lower center", bbox_to_anchor=(0.5, 1.0))
plt.show()
