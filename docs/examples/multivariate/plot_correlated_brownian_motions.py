"""
Correlated Brownian motions
===========================

Standard Brownian motions whose increments are correlated. The correlation
goes from -1, where one mirrors the other, to 1, where they coincide.
"""

import matplotlib.pyplot as plt
import numpy as np

from samplepaths import CorrelatedBrownianMotions

fig, axes = plt.subplots(1, 3, figsize=(10, 3.5), sharey=True)
for ax, correlation in zip(axes, (-0.9, 0.0, 0.9), strict=True):
    motions = CorrelatedBrownianMotions(correlation=correlation, rng=np.random.default_rng(161))
    first, second = motions.sample(1000)
    ax.plot(motions.times, first)
    ax.plot(motions.times, second)
    ax.set_title(f"correlation = {correlation}")
    ax.set_xlabel("Time")
plt.show()

# %%
# A correlation matrix gives any number of motions. The correlation of the
# final values is that of the matrix.

matrix = [[1.0, 0.6, -0.4], [0.6, 1.0, 0.0], [-0.4, 0.0, 1.0]]
motions = CorrelatedBrownianMotions(correlation=matrix, rng=np.random.default_rng(162))
final = motions.sample_paths(16, paths=100_000)[:, :, -1]
print(motions.components)
print(np.round(np.corrcoef(final.T), 2))
