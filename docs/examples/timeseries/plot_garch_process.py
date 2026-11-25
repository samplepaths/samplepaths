"""
GARCH process
=============

The variance depends on the squares of the previous values and on the
previous variances, which makes agitated periods last. It is the standard
model of the volatility of financial returns.
"""

import matplotlib.pyplot as plt
import numpy as np

from samplepaths import GARCHProcess

model = GARCHProcess(omega=0.02, alpha=0.1, beta=0.88, rng=np.random.default_rng(157))
values = model.sample(2000)

fig, axes = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
axes[0].plot(values, linewidth=0.6)
axes[0].set_ylabel("Value")
axes[1].plot(model.volatility, color="C1")
axes[1].set_ylabel("Volatility")
axes[1].set_xlabel("Step")
plt.show()

# %%
# The values have a variance of ``omega / (1 - alpha - beta)`` in the long
# run, and heavier tails than a normal distribution.

final = model.sample_paths(500, paths=100_000)[:, -1]
print("variance:", round(final.var(), 3), "expected:", round(0.02 / (1 - 0.1 - 0.88), 3))
print("share beyond 4 standard deviations:", np.mean(np.abs(final) > 4 * final.std()))
print("for a normal distribution:          ", 6.3e-5)
