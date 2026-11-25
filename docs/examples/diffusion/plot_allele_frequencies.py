"""
Frequencies and proportions
===========================

The Kimura and the Jacobi processes live on the interval from 0 to 1. The
Kimura process follows the frequency of an allele, which is eventually lost
or fixed; the Jacobi process reverts to a mean and is used for quantities
such as a correlation.
"""

import matplotlib.pyplot as plt
import numpy as np

from samplepaths import JacobiProcess, KimuraProcess

kimura = KimuraProcess(selection=0.0, vol=0.5, t=20.0, rng=np.random.default_rng(35))
jacobi = JacobiProcess(speed=1.0, mean=0.6, vol=0.3, t=20.0, rng=np.random.default_rng(36))

fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), sharey=True)
for _ in range(6):
    path = kimura.sample_with_initial(4000, initial=0.5)
    axes[0].plot(kimura.times, path)
    path = jacobi.sample_with_initial(4000, initial=0.1)
    axes[1].plot(jacobi.times, path)
axes[0].set_title("Kimura: lost or fixed")
axes[1].set_title("Jacobi: reverts to 0.6")
axes[1].axhline(0.6, linestyle="--", color="gray")
for ax in axes:
    ax.set_xlabel("Time")
axes[0].set_ylabel("Frequency")
plt.show()

# %%
# Without selection, an allele is fixed with a probability equal to its
# initial frequency.

for initial in (0.2, 0.5, 0.8):
    paths = KimuraProcess(vol=1.0, t=40.0, rng=np.random.default_rng(37)).sample_paths_with_initial(
        2000, paths=4000, initial=initial
    )
    print(
        f"initial frequency {initial}: fixed in {np.mean(paths[:, -1] == 1.0):.3f} of the realizations"
    )
