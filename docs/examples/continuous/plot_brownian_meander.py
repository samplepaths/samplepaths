"""
Brownian meander
================

A Brownian meander is a Brownian motion conditioned to stay nonnegative. Its
endpoint is random, or fixed with ``sample_with_endpoint()``.
"""

import matplotlib.pyplot as plt
import numpy as np

from samplepaths import BrownianMeander

meander = BrownianMeander(t=1.0, rng=np.random.default_rng(4))

fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), sharey=True)
for _ in range(4):
    axes[0].plot(np.linspace(0, 1, 513), meander.sample(512))
    axes[1].plot(np.linspace(0, 1, 513), meander.sample_with_endpoint(512, b=1.5))
axes[0].set_title("Random endpoint")
axes[1].set_title("Endpoint fixed at 1.5")
for ax in axes:
    ax.set_xlabel("Time")
axes[0].set_ylabel("Value")
plt.show()
