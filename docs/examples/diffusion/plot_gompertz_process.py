"""
Gompertz process
================

Growth which slows down with the logarithm of the size. It describes the
growth of tumors and of animals: fast at first, then ever slower towards a
limit.
"""

import matplotlib.pyplot as plt
import numpy as np

from samplepaths import GompertzProcess

gompertz = GompertzProcess(growth=2.0, decay=0.5, vol=0.1, t=10.0, rng=np.random.default_rng(121))

fig, ax = plt.subplots(figsize=(8, 4))
for _ in range(5):
    path = gompertz.sample_with_initial(2000, initial=1.0)
    ax.plot(gompertz.times, path, linewidth=0.9)
ax.axhline(np.exp(2.0 / 0.5), linestyle="--", color="gray", label="limit without noise")
ax.set_xlabel("Time")
ax.set_ylabel("Size")
ax.legend()
plt.show()

# %%
# The logarithm of the size is an Ornstein-Uhlenbeck process, so its final
# values are normal.

logs = np.log(gompertz.sample_paths_with_initial(2000, paths=5000, initial=1.0)[:, -1])

fig, ax = plt.subplots(figsize=(8, 3.5))
ax.hist(logs, bins=60, density=True)
ax.set_xlabel("Logarithm of the size at time 10")
plt.show()
