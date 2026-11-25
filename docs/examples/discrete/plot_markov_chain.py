"""
Markov chain
============

A two-state chain for the weather: dry days tend to follow dry days. The
share of time spent in each state converges to the stationary distribution.
"""

import matplotlib.pyplot as plt
import numpy as np

from samplepaths import MarkovChain

transition = np.array(
    [
        [0.9, 0.1],  # dry -> dry, wet
        [0.5, 0.5],  # wet -> dry, wet
    ]
)
chain = MarkovChain(transition=transition, initial=[1, 0], rng=np.random.default_rng(18))
states = chain.sample(365)

# The stationary distribution is the left eigenvector for the eigenvalue 1.
values, vectors = np.linalg.eig(transition.T)
stationary = np.real(vectors[:, np.argmax(np.real(values))])
stationary /= stationary.sum()
print("stationary share of wet days:", round(stationary[1], 3))
print("simulated share of wet days: ", round(states.mean(), 3))

fig, ax = plt.subplots(figsize=(8, 2.5))
ax.step(np.arange(120), states[:120], where="post")
ax.set_yticks([0, 1], ["dry", "wet"])
ax.set_xlabel("Day")
plt.show()
