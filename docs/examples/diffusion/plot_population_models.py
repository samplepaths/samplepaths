"""
Population growth
=================

Three models of a population which grows towards a limit. The Verhulst and
the Gompertz processes have noise proportional to the size of the
population; the Feller logistic process has demographic noise, which matters
most for small populations and can drive them extinct.
"""

import matplotlib.pyplot as plt
import numpy as np

from samplepaths import FellerLogisticProcess, GompertzProcess, VerhulstProcess

processes = {
    "Verhulst": VerhulstProcess(rate=1.0, capacity=100.0, vol=0.15, t=12.0),
    # this Gompertz process settles around exp(growth / decay), about 100
    "Gompertz": GompertzProcess(growth=2.3, decay=0.5, vol=0.15, t=12.0),
    "Feller logistic": FellerLogisticProcess(speed=0.01, level=100.0, vol=1.5, t=12.0),
}

fig, axes = plt.subplots(1, 3, figsize=(10, 3.5), sharey=True)
for ax, (name, process) in zip(axes, processes.items(), strict=True):
    process.rng = np.random.default_rng(33)
    for _ in range(5):
        path = process.sample_with_initial(2400, initial=2.0)
        ax.plot(process.times, path)
    ax.set_title(name)
    ax.set_xlabel("Time")
axes[0].set_ylabel("Population")
plt.show()

# %%
# With demographic noise a small population can die out before it takes off.

process = processes["Feller logistic"]
process.rng = np.random.default_rng(34)
paths = process.sample_paths_with_initial(1200, paths=5000, initial=2.0)
print("share of populations which are extinct at the end:", np.mean(paths[:, -1] == 0.0))
