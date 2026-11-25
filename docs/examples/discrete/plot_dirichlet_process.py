"""
Dirichlet process
=================

Each value is either a new draw from a base distribution or a repeat of an
earlier value. A larger ``alpha`` gives more distinct values.
"""

import matplotlib.pyplot as plt
import numpy as np

from samplepaths import DirichletProcess

fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), sharey=True)
for ax, alpha in zip(axes, (1, 20), strict=True):
    rng = np.random.default_rng(21)
    process = DirichletProcess(base=rng.standard_normal, alpha=alpha, rng=rng)
    values = process.sample(300)
    ax.scatter(np.arange(len(values)), values, s=8)
    ax.set_title(f"alpha = {alpha}: {len(np.unique(values))} distinct values")
    ax.set_xlabel("Step")
axes[0].set_ylabel("Value")
plt.show()
