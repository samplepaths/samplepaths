"""
Compound Poisson process
========================

A compound Poisson process is constant between the jumps of a Poisson
process and moves by a random amount at each jump. Here it adds up the claims
of an insurer, whose sizes are exponential.
"""

import matplotlib.pyplot as plt
import numpy as np

from samplepaths import CompoundPoissonProcess

claims = CompoundPoissonProcess(
    rate=5.0,
    jumps=lambda rng, size: rng.exponential(1000.0, size),
    t=10.0,
    rng=np.random.default_rng(32),
)

fig, ax = plt.subplots(figsize=(8, 4))
for _ in range(4):
    total = claims.sample(4000)
    ax.step(claims.times, total, where="post")
ax.plot(claims.times, 5.0 * 1000.0 * claims.times, linestyle="--", color="gray", label="expected")
ax.set_xlabel("Time (years)")
ax.set_ylabel("Total claims")
ax.legend()
plt.show()
