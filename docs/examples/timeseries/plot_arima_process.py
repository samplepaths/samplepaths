"""
ARIMA process
=============

An autoregressive moving average process, integrated ``d`` times. Without
integration it settles around zero; integrated once it wanders; twice, its
slope wanders.
"""

import matplotlib.pyplot as plt
import numpy as np

from samplepaths import ARIMAProcess

fig, axes = plt.subplots(3, 1, figsize=(8, 6), sharex=True)
for ax, order in zip(axes, (0, 1, 2), strict=True):
    model = ARIMAProcess(ar=0.6, ma=0.3, d=order, rng=np.random.default_rng(153))
    for _ in range(3):
        ax.plot(model.sample(300), linewidth=0.9)
    ax.set_ylabel(f"d = {order}")
axes[-1].set_xlabel("Step")
plt.show()

# %%
# Differencing an integrated series gives back the stationary one.

integrated = ARIMAProcess(ar=0.6, ma=0.3, d=1, rng=np.random.default_rng(154)).sample(300)
stationary = ARIMAProcess(ar=0.6, ma=0.3, d=0, rng=np.random.default_rng(154)).sample(300)
print("largest difference:", np.abs(np.diff(integrated) - stationary[1:]).max())
