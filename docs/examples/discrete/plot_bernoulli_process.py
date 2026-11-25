"""
Bernoulli process
=================

A sequence of independent trials which succeed with probability ``p``. The
running share of successes converges to ``p``.
"""

import matplotlib.pyplot as plt
import numpy as np

from samplepaths import BernoulliProcess

bernoulli = BernoulliProcess(p=0.3, rng=np.random.default_rng(16))
trials = bernoulli.sample(400)

fig, (top, bottom) = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
top.vlines(np.flatnonzero(trials), 0, 1)
top.set_yticks([])
top.set_title("Successes")
bottom.plot(np.cumsum(trials) / np.arange(1, len(trials) + 1))
bottom.axhline(0.3, linestyle="--", color="gray")
bottom.set_xlabel("Trial")
bottom.set_ylabel("Share of successes")
plt.show()
