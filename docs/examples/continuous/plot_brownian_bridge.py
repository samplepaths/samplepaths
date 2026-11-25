"""
Brownian bridge
===============

A Brownian bridge is a Brownian motion conditioned to end at the value ``b``
at time ``t``.
"""

import matplotlib.pyplot as plt
import numpy as np

from samplepaths import BrownianBridge

bridge = BrownianBridge(b=2.0, t=1.0, rng=np.random.default_rng(2))

fig, ax = plt.subplots(figsize=(8, 4))
for _ in range(5):
    ax.plot(np.linspace(0, bridge.t, 513), bridge.sample(512))
ax.scatter([0, bridge.t], [0, bridge.b], color="gray", zorder=3)
ax.set_xlabel("Time")
ax.set_ylabel("Value")
plt.show()
