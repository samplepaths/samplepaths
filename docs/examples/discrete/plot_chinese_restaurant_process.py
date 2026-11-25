"""
Chinese restaurant process
==========================

Customers join an occupied table with a probability which grows with its
size, or open a new one. A few tables end up holding most customers.
"""

import matplotlib.pyplot as plt
import numpy as np

from samplepaths import ChineseRestaurantProcess

restaurant = ChineseRestaurantProcess(discount=0.3, strength=2.0, rng=np.random.default_rng(20))
partition = restaurant.sample_partition(500)
sizes = sorted((len(table) for table in partition), reverse=True)
print("tables:", len(sizes), "largest:", sizes[:5])

fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(np.arange(1, len(sizes) + 1), sizes)
ax.set_xlabel("Table, by size")
ax.set_ylabel("Customers")
plt.show()
